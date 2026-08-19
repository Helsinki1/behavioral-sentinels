"""Offline selftest for experiment 6. No network: the LLM is mocked
(experiments5.selftest5.MockLLM answers from generator truth and degrades on
scripted turns).

  python -m experiments6.selftest6

Verifies:
  - the store reducer equals generator truth on EVERY turn of EVERY task in
    the full data3 pool (the R1 fidelity guarantee)
  - R1 resume messages contain exactly the store's state (recomputed
    independently) and R2 resumes are the verbatim prior user log
  - A_no_reset import from runs5
  - every arm x every domain runs end-to-end; resets fire where the policy
    says (clock cadence, dense cadence, zero-carry after slips, oracle
    placement, random budget-matched to Z_reground)
  - metrics aggregation incl. the cross-experiment operator contrasts
"""
import asyncio
import json
import re
import shutil
import tempfile
from pathlib import Path

from experiments5.selftest5 import BAD_TURNS, CURRENT_TASK, MockLLM

from . import config6, store6
from .config6 import ARMS

FAILURES = []


def check(name, ok, detail=""):
    print(("  ok  " if ok else "  FAIL") + f" {name}" +
          (f" -- {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append((name, detail))


def unit_tests():
    # the fidelity guarantee: store == generator truth over the full pool
    report = store6.verify_pool()
    for domain, r in report.items():
        check(f"store fidelity {domain}: {r['turns']} turns, 0 errors",
              not r["errors"], str(r["errors"][:3]))

    # resume builders are pure functions of user-issued content
    tasks = json.loads((config6.DATA_DIR / "tasks3_registers.json").read_text())
    task = tasks[0]
    store = store6.make_store("registers", task)
    for turn in task["turns"][:5]:
        store.apply(turn)
    resume = store6.resume_reground("registers", task, store, 6)
    for reg, val in store.regs.items():
        check(f"R1 resume carries current {reg}",
              f"- {reg} = {val}" in resume, resume[:200])
    replay = store6.resume_replay("registers", task, 6)
    from experiments3.canaries3 import build_turn_body
    for turn in task["turns"][1:5]:
        body = build_turn_body("registers", task, turn, "baseline")
        check(f"R2 replay carries turn {turn['turn']} verbatim", body in replay)
    check("R2 replay carries nothing beyond turn 5",
          f"Turn 6:" not in replay)


def integration(tmp):
    from experiments5 import run_all5
    from . import harness6, run_all6
    from experiments5.run_all5 import select_tasks

    pool = {d: select_tasks(d, 2) for d in config6.DOMAINS}
    mock = MockLLM(pool)
    harness6.chat = mock.chat
    import experiments5.harness5 as harness5
    harness5.chat = mock.chat          # judge_turn lives there
    config6.RUNS_DIR = run_all6.RUNS_DIR = tmp / "runs6"
    config6.RUNS5_DIR = run_all6.RUNS5_DIR = tmp / "runs5"
    config6.RESULTS_DIR = tmp / "results6"

    orig_run_one = run_all6.run_one

    async def tagged_run_one(cfg, model, domain, task, arm_name, extras):
        CURRENT_TASK.set((domain, task["task_id"]))
        return await orig_run_one(cfg, model, domain, task, arm_name, extras)

    run_all6.run_one = tagged_run_one

    # fabricate runs5: A_no_reset (for import) + exp-5 arms for the
    # cross-experiment plumbing (content = the same A trajectories)
    async def gen_A():
        for domain in config6.DOMAINS:
            for task in pool[domain]:
                CURRENT_TASK.set((domain, task["task_id"]))
                out = await harness6.run_arm(
                    {}, domain, task, "A_no_reset", "reground", lambda s: False)
                for arm5 in ("A_no_reset", "Z_routed", "C_clock", "C_ctx",
                             "C_judge", "F_oracle"):
                    p = (tmp / "runs5" / "gpt-oss-20b" / domain / arm5
                         / f"task_{task['task_id']:03d}.json")
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(json.dumps({**out, "arm": arm5}))
    asyncio.run(gen_A())

    missing = run_all6.import_A("gpt-oss-20b", pool)
    check("A_no_reset imported from runs5", missing == 0, str(missing))

    for arm in config6.ARM_ORDER:
        asyncio.run(run_all6.run_arm_all("gpt-oss-20b", arm, pool,
                                         progress_every=10**9))

    ids = {d: [t["task_id"] for t in pool[d]] for d in config6.DOMAINS}
    recs = {arm: {(d, tid): r for d in config6.DOMAINS
                  for tid, r in run_all6.load_arm(
                      "gpt-oss-20b", d, arm, ids[d]).items()}
            for arm in ARMS}
    n_expected = sum(len(v) for v in ids.values())
    for arm in ARMS:
        check(f"{arm}: all {n_expected} trajectories complete",
              len(recs[arm]) == n_expected, str(len(recs[arm])))

    tasks_by_key = {(d, t["task_id"]): t for d in config6.DOMAINS
                    for t in pool[d]}

    # every R1 reset injected exactly the store's true state at that turn
    for arm in ("C_clock", "Z_reground", "G_dense", "B_random", "F_oracle"):
        for key, r in recs[arm].items():
            domain, _ = key
            task = tasks_by_key[key]
            ref = store6.make_store(domain, task)
            audits = {a["turn"]: a["resume"] for a in r["resets"]}
            for turn in task["turns"]:
                if turn["turn"] in audits:
                    want = store6.resume_reground(domain, task, ref,
                                                  turn["turn"])
                    check(f"{arm} {key}: reset@{turn['turn']} resume == store",
                          audits[turn["turn"]] == want)
                ref.apply(turn)

    # replay arms: resume is the verbatim user log
    for key, r in recs["C_clock_replay"].items():
        domain, _ = key
        task = tasks_by_key[key]
        for a in r.get("resets", []):
            want = store6.resume_replay(domain, task, a["turn"])
            check(f"C_clock_replay {key}: reset@{a['turn']} resume == log",
                  a["resume"] == want)

    # cadences
    for key, r in recs["C_clock"].items():
        want = [t for t in range(6, r["horizon"] + 1, 6)][:config6.MAX_RESETS]
        check(f"C_clock {key}: resets every 6", r["reset_turns"] == want,
              str(r["reset_turns"]))
    for key, r in recs["G_dense"].items():
        want = [t for t in range(3, r["horizon"] + 1, 3)][:config6.MAX_RESETS]
        check(f"G_dense {key}: resets every 3", r["reset_turns"] == want,
              str(r["reset_turns"]))

    # zero-carry fires on scripted degradation
    for key, r in recs["Z_reground"].items():
        fired = {rec["turn"] for rec in r["records"] if rec["zerocarry_fired"]}
        check(f"Z_reground {key}: monitor fired on a scripted turn",
              any(b in fired for b in BAD_TURNS), f"fired {sorted(fired)}")

    # budget matching and oracle placement
    for key, r in recs["B_random"].items():
        z = recs["Z_reground"][key]
        check(f"B_random {key}: budget matches Z_reground",
              r["n_resets"] == min(z["n_resets"], config6.MAX_RESETS),
              f"random {r['n_resets']} vs Z {z['n_resets']}")
    a = recs["A_no_reset"]
    for key, r in recs["F_oracle"].items():
        fh = a[key]["first_hallucination"]
        if fh is None:
            check(f"F_oracle {key}: no reset when A never hallucinated",
                  r["n_resets"] == 0)
        else:
            check(f"F_oracle {key}: one reset at the oracle turn",
                  r["reset_turns"] == [fh], str(r["reset_turns"]))

    # judge plumbing
    for key, r in recs["C_judge"].items():
        check(f"C_judge {key}: verdicts recorded every turn",
              all(rec["judge_yes"] is not None for rec in r["records"]))

    # metrics + cross-experiment contrasts run on the produced files
    from . import metrics6
    metrics6.select_tasks = lambda d, n=None: pool[d]
    run_all5.RUNS_DIR = tmp / "runs5"
    import experiments5.run_all5 as ra5
    ra5.RUNS_DIR = tmp / "runs5"
    config6.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = metrics6.compute("gpt-oss-20b")
    check("metrics: pooled over all tasks", out["n_tasks"] == n_expected)
    check("metrics: gate contrast present",
          any("GATE" in c["contrast"] for c in out["contrasts"]))
    check("metrics: cross-experiment operator contrasts present",
          len(out["cross_experiment"]) > 0)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="exp6_selftest_"))
    try:
        print("== unit tests ==")
        unit_tests()
        print("== integration (mock LLM, all arms x all domains) ==")
        integration(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print()
    if FAILURES:
        print(f"SELFTEST FAILED: {len(FAILURES)} failure(s)")
        for name, detail in FAILURES:
            print(f"  - {name}: {detail}")
        raise SystemExit(1)
    print("SELFTEST PASSED")


if __name__ == "__main__":
    main()
