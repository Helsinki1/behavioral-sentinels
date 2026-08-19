"""Offline selftest for experiment 5. No network: the LLM is mocked.

  python -m experiments5.selftest5

Exercises every arm x every domain end-to-end on real task data with a mock
model that answers correctly except on scripted BAD turns, and verifies:
  - the router path (mock genre labels -> probes; fallback on garbage)
  - hallucination checkers accept correct replies and flag bad ones
  - carried-probe scoring and the probe reset policy (incl. grace + cap)
  - the zero-carry monitors fire on contradictions/malformed output only
  - the judge plumbing and policy
  - compaction: snapshot request routed to the right schema, validation
    accepts a good snapshot, rejects a broken one, retries, and the
    trajectory continues either way
  - budget matching (B_random gets D_routed's per-task reset count)
  - metrics aggregation runs on the produced files
"""
import asyncio
import contextvars
import re
import shutil
import tempfile
from pathlib import Path

from . import config5
from .config5 import ARMS

CURRENT_TASK = contextvars.ContextVar("exp5_selftest_task", default=None)

FAILURES = []


def check(name, ok, detail=""):
    print(("  ok  " if ok else "  FAIL") + f" {name}" +
          (f" -- {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append((name, detail))


# --------------------------------------------------------------- mock model

BAD_TURNS = {3, 9}          # scripted degraded turns (task turns, 1-based)
PROBE_BAD_TURNS = {5, 11}   # probe-only failures (task still clean)


class MockLLM:
    """Answers from ground truth; degrades on scripted turns. The task under
    test is read from a contextvar set by the wrapped run_one, which is
    concurrency-safe because each asyncio task owns a copy of the context."""

    def __init__(self, pool):
        self.turn_index = {}       # (domain, task_id, turn) -> turn dict
        self.task_index = {}
        for domain, tasks in pool.items():
            for t in tasks:
                self.task_index[(domain, t["task_id"])] = t
                for turn in t["turns"]:
                    self.turn_index[(domain, t["task_id"], turn["turn"])] = turn
        self.break_snapshot_once = set()   # (domain, task_id) -> fail 1st attempt
        self.calls = {"task": 0, "snapshot": 0, "judge": 0, "router": 0}

    def _probe_lines(self, domain, turn, text_so_far):
        c3 = turn["c3"]
        lines = []
        good = turn["turn"] not in PROBE_BAD_TURNS
        if "LEDGER" in text_so_far:
            lines.append("LEDGER: " + (" ".join(str(x) for x in c3["stair_expect"])
                                       if good else "999"))
        if "CHECK" in text_so_far:
            lines.append(f"CHECK: {c3['check'] if good else 0}")
        if "ECHO" in text_so_far:
            vals = [c3["lag1"], c3["lag3"], c3["lag6"]]
            if not good:
                vals = ["ENG-0000"] * 3
            lines.append("ECHO: " + " ".join(vals))
        if "SHADOW" in text_so_far:
            lines.append(f"SHADOW: {c3['shadow_truth'] if good else 'wrongling'}")
        if "TAG" in text_so_far and c3["tag_query"]:
            lines.append(f"TAG: {c3['tag_truth'] if good else 'zzz'}")
        return lines

    def _task_reply(self, domain, task, turn, carried_instructions):
        bad = turn["turn"] in BAD_TURNS
        if domain == "coding":
            if bad:
                body = ["```python", "def totally_wrong():", "    return None",
                        "```", "SIG nothing_here(x)"]
            else:
                defs = []
                wired = {w["fn"]: w["callee"] for w in turn["wire_calls"]}
                for fn, params in turn["defs_truth"].items():
                    ret = f"return {wired[fn]}({', '.join(params)})" if fn in wired \
                        else "return None"
                    defs.append(f"def {fn}({', '.join(params)}):\n    {ret}")
                sigs = []
                for q in turn["queries"]:
                    sigs.append(f"SIG {q['fn']} DELETED" if q["deleted"]
                                else f"SIG {q['fn']}({', '.join(q['params'])})")
                inner = ["```python"] + defs + ["```"] + sigs
                body = inner if turn["defs_truth"] else (
                    ["```python", "def keepalive_stub():", "    return None", "```"]
                    + sigs)
            # a deletion-only turn: emitting a stub def is harmless & non-empty
        elif domain == "registers":
            if bad:
                # omit the first queried register (deterministic monitor fire)
                # and answer the rest wrongly
                body = [f"VALUE {reg} = 424242" for reg in turn["queries"][1:]]
                body = body or ["I seem to have lost track of the registers."]
            else:
                body = [f"VALUE {reg} = {turn['truth'][reg]}"
                        for reg in turn["queries"]]
        else:
            body = [f"ANSWER: {'zzzgarbled' if bad else turn['answer']}"]
        return "\n".join(self._probe_lines(domain, turn, carried_instructions)
                         + body)

    # -- entry ---------------------------------------------------------
    async def chat(self, cfg, messages, max_tokens=None, temperature=0.2):
        last = messages[-1]["content"]
        joined = "\n".join(m["content"] for m in messages)
        if "GENRE" in messages[0]["content"]:
            self.calls["router"] += 1
            if "Python module" in last:
                return "GENRE: ARTIFACT_ACCUMULATION", {"prompt_tokens": 10, "completion_tokens": 5}
            if "integer registers" in last:
                return "GENRE: UPDATE_APPLICATION", {"prompt_tokens": 10, "completion_tokens": 5}
            return "GENRE: RETRIEVAL_DISTANCE", {"prompt_tokens": 10, "completion_tokens": 5}
        if "monitoring system" in messages[0]["content"]:
            self.calls["judge"] += 1
            yes = any(f"Turn {b}:" in last for b in BAD_TURNS)
            return ("YES" if yes else "NO"), {"prompt_tokens": 10, "completion_tokens": 2}
        if "compaction engine" in messages[0]["content"]:
            self.calls["snapshot"] += 1
            return self._snapshot(last), {"prompt_tokens": 20, "completion_tokens": 20}
        # ordinary task turn
        self.calls["task"] += 1
        key = CURRENT_TASK.get()
        assert key, "contextvar not set for a task turn"
        m = re.search(r"Turn (\d+):", last)
        assert m, f"unrecognised task message: {last[:120]}"
        task = self.task_index[key]
        turn = self.turn_index[(key[0], key[1], int(m.group(1)))]
        reply = self._task_reply(key[0], task, turn, joined)
        return reply, {"prompt_tokens": 100 * len(messages), "completion_tokens": 50}

    def _snapshot(self, request):
        domain = ("coding" if "CURRENT_MODULE" in request else
                  "registers" if "CURRENT_REGISTERS" in request else "babi")
        key = CURRENT_TASK.get()
        assert key, "contextvar not set for a snapshot request"
        last_turn = int(re.search(r"LAST_COMPLETED_TASK_TURN: (\d+)", request).group(1))
        next_turn = int(re.search(r"NEXT_TASK_TURN: (\d+)", request).group(1))
        if key in self.break_snapshot_once:
            self.break_snapshot_once.discard(key)
            return "THIS IS NOT A SNAPSHOT"
        task = self.task_index[key]
        if domain == "coding":
            turn = self.turn_index[(key[0], key[1], last_turn)]
            live = turn["live_symbols"]
            block = ("DELETED_FUNCTIONS:\n- NONE\nCURRENT_MODULE:\n```python\n"
                     + "\n".join(f"def {fn}(a):\n    return None" for fn in live)
                     + "\n```")
        elif domain == "registers":
            block = ("DELETED_REGISTERS:\n- NONE\nCURRENT_REGISTERS:\n```text\n"
                     + "\n".join(f"{k} = 1" for k in task["initial_state"])
                     + "\n```")
        else:
            block = "CURRENT_STORY:\n```text\nMary went to the kitchen.\n```"
        return (f"STATE_SNAPSHOT_V1\nPROGRESS:\nLAST_COMPLETED_TASK_TURN: {last_turn}\n"
                f"NEXT_TASK_TURN: {next_turn}\nLAST_REQUIRED_RESPONSE_STATE: NONE\n"
                "DURABLE_USER_RULES:\n- keep going\nACCUMULATED_RULE_FACTS:\n- NONE\n"
                "OPEN_ITEMS:\n- NONE\nUNCERTAINTIES:\n- NONE\n"
                f"{block}\nEND_STATE_SNAPSHOT")


# ------------------------------------------------------------------- tests

def unit_tests():
    from .harness5 import validate_snapshot
    from .routing5 import (BabiMonitor, CodingMonitor, RegistersMonitor,
                           GENRE_TO_PROBE, probe_for)

    good = ("STATE_SNAPSHOT_V1\nPROGRESS:\nLAST_COMPLETED_TASK_TURN: 5\n"
            "NEXT_TASK_TURN: 6\nLAST_REQUIRED_RESPONSE_STATE: NONE\n"
            "DURABLE_USER_RULES:\n- x\nACCUMULATED_RULE_FACTS:\n- NONE\n"
            "OPEN_ITEMS:\n- NONE\nUNCERTAINTIES:\n- NONE\n"
            "DELETED_REGISTERS:\n- NONE\nCURRENT_REGISTERS:\n```text\nalpha = 3\n```\n"
            "END_STATE_SNAPSHOT")
    check("registers snapshot valid", validate_snapshot("registers", good, 6) == [])
    check("registers snapshot wrong turn rejected",
          validate_snapshot("registers", good, 7) != [])
    bad = good.replace("alpha = 3", "alpha = banana")
    check("registers snapshot bad line rejected",
          validate_snapshot("registers", bad, 6) != [])

    cm = CodingMonitor()
    f1 = cm.check("Turn 1:\nChanges:\n- Add parse_cfg.\nReport: parse_cfg",
                  "```python\ndef parse_cfg(path):\n    return None\n```\nSIG parse_cfg(path)")
    check("coding monitor quiet on clean turn", f1 == [], str(f1))
    f2 = cm.check("Turn 2:\nChanges:\n- nothing\nReport: parse_cfg",
                  "```python\ndef stub():\n    return None\n```\nSIG parse_cfg(path, extra)")
    check("coding monitor fires on sig drift",
          any("sig_drift" in x for x in f2), str(f2))
    f3 = cm.check("Turn 3:\nChanges:\n- nothing\nReport: gone_fn", "no code at all")
    check("coding monitor fires on malformed", len(f3) >= 1, str(f3))

    rm = RegistersMonitor()
    r1 = rm.check("Turn 1:\nUpdates:\n- Set alpha to 3.\nReport: alpha",
                  "VALUE alpha = 3")
    check("registers monitor quiet on clean turn", r1 == [], str(r1))
    r2 = rm.check("Turn 2:\nUpdates:\n- Add 1 to beta.\nReport: alpha",
                  "VALUE alpha = 99")
    check("registers monitor fires on drift",
          any("value_drift" in x for x in r2), str(r2))
    r3 = rm.check("Turn 3:\nUpdates:\n- Add 2 to alpha.\nReport: alpha",
                  "VALUE alpha = 101")
    check("registers monitor quiet after mention", r3 == [], str(r3))

    bm = BabiMonitor()
    b1 = bm.check("Turn 1:\nMary went to the kitchen.\nQuestion: Where is Mary?",
                  "ANSWER: kitchen")
    check("babi monitor quiet on grounded answer", b1 == [], str(b1))
    b2 = bm.check("Turn 2:\nJohn moved to the garden.\nQuestion: Where is John?",
                  "ANSWER: spaceship")
    check("babi monitor fires on ungrounded answer",
          any("ungrounded" in x for x in b2), str(b2))
    b3 = bm.check("Turn 3:\n--- NEW STORY ---\nSandra grabbed the milk.\n"
                  "Question: Who has the milk?", "ANSWER: kitchen")
    check("babi monitor resets vocabulary on NEW STORY",
          any("ungrounded" in x for x in b3), str(b3))

    check("probe_for routed", probe_for("routed", "coding", "lag_span") == "lag_span")
    check("probe_for blanket", probe_for("blanket", "babi", None) == "staircase")
    check("probe_for rotated", probe_for("rotated", "registers", None) == "lag_span")
    check("genre map covers 5 genres", len(GENRE_TO_PROBE) == 5)


def integration(tmp):
    from . import harness5, routing5, run_all5
    from .run_all5 import load_arm, select_tasks

    pool = {d: select_tasks(d, 2) for d in config5.DOMAINS}
    mock = MockLLM(pool)
    harness5.chat = mock.chat
    routing5.chat = mock.chat
    config5.RUNS_DIR = run_all5.RUNS_DIR = tmp / "runs"
    config5.RESULTS_DIR = tmp / "results"

    orig_run_one = run_all5.run_one

    async def tagged_run_one(cfg, model, domain, task, arm_name, extras, routed):
        CURRENT_TASK.set((domain, task["task_id"]))
        return await orig_run_one(cfg, model, domain, task, arm_name, extras, routed)

    run_all5.run_one = tagged_run_one

    # break the first snapshot attempt for one task to test retry
    key = ("coding", pool["coding"][0]["task_id"])
    mock.break_snapshot_once.add(key)

    asyncio.run(run_all5.route_all("gpt-oss-20b", pool))
    for arm in config5.ARM_ORDER:
        asyncio.run(run_all5.run_arm_all("gpt-oss-20b", arm, pool,
                                         progress_every=10**9))

    ids = {d: [t["task_id"] for t in pool[d]] for d in config5.DOMAINS}
    recs = {arm: {(d, tid): r for d in config5.DOMAINS
                  for tid, r in load_arm("gpt-oss-20b", d, arm, ids[d]).items()}
            for arm in ARMS}
    n_expected = sum(len(v) for v in ids.values())
    for arm in ARMS:
        check(f"{arm}: all {n_expected} trajectories complete",
              len(recs[arm]) == n_expected, str(len(recs[arm])))

    # scripted degradation is detected on BAD turns in the no-reset arm
    a = recs["A_no_reset"]
    for key, r in a.items():
        bad_seen = {rec["turn"] for rec in r["records"] if rec["hallucination"]}
        want = {b for b in BAD_TURNS if b <= r["horizon"]}
        check(f"A_no_reset {key}: hallucinations exactly on scripted turns",
              bad_seen == want, f"got {sorted(bad_seen)} want {sorted(want)}")

    # D_routed fires: probe fails on scripted probe-bad turns, resets follow
    for key, r in recs["D_routed"].items():
        fails = {rec["turn"] for rec in r["records"] if rec["probe_fail"]}
        expect = {b for b in (BAD_TURNS | PROBE_BAD_TURNS) if b <= r["horizon"]}
        # BAD turns break the task reply, which usually breaks the probe line too
        check(f"D_routed {key}: probe failed at least on probe-bad turns",
              {b for b in PROBE_BAD_TURNS if b <= r["horizon"]} <= fails,
              f"got {sorted(fails)}")
        check(f"D_routed {key}: resets follow failures (grace {config5.RESET_GRACE})",
              r["n_resets"] >= 1, str(r["n_resets"]))

    # Z_routed fires on scripted bad turns (monitor sees malformed/ungrounded)
    for key, r in recs["Z_routed"].items():
        fired = {rec["turn"] for rec in r["records"] if rec["zerocarry_fired"]}
        check(f"Z_routed {key}: monitor fired on some scripted turn",
              any(b in fired for b in BAD_TURNS), f"fired {sorted(fired)}")

    # judge arm: verdicts recorded, resets follow YES
    for key, r in recs["C_judge"].items():
        checkv = [rec["judge_yes"] for rec in r["records"]]
        check(f"C_judge {key}: judge verdicts recorded on every turn",
              all(v is not None for v in checkv))

    # budget matching: B_random reset count == min(D_routed count, feasible)
    for key, r in recs["B_random"].items():
        d = recs["D_routed"][key]
        check(f"B_random {key}: budget matches D_routed",
              r["n_resets"] == min(d["n_resets"], config5.MAX_RESETS),
              f"random {r['n_resets']} vs routed {d['n_resets']}")

    # oracle resets once, before A's first hallucination turn
    for key, r in recs["F_oracle"].items():
        fh = a[key]["first_hallucination"]
        if fh is None:
            check(f"F_oracle {key}: no reset when A never hallucinated",
                  r["n_resets"] == 0)
        else:
            check(f"F_oracle {key}: exactly one reset at the oracle turn",
                  r["reset_turns"] == [fh], str(r["reset_turns"]))

    # compaction retry happened for the scripted broken snapshot
    dr = None
    for arm in ("C_clock",):
        rec = recs[arm][key2] if (key2 := ("coding", pool["coding"][0]["task_id"])) in recs[arm] else None
        if rec and rec["compactions"]:
            dr = rec["compactions"][0]
    check("compaction retry consumed the scripted bad snapshot",
          dr is not None and dr["applied"] and dr["attempts"] == 2,
          str(dr and {k: dr[k] for k in ('applied', 'attempts')}))

    # carried arms actually carry the right probes
    for key, r in recs["D_rotated"].items():
        check(f"D_rotated {key}: condition is the rotated probe",
              r["condition"] == config5.ROTATED_TABLE[key[0]], r["condition"])
    for key, r in recs["C_prime_routed"].items():
        check(f"C_prime_routed {key}: carries routed probe, clock resets",
              r["condition"] in routing5.GENRE_TO_PROBE.values()
              and r["reset_turns"] == recs["C_clock"][key]["reset_turns"],
              f"{r['condition']} resets {r['reset_turns']} vs clock "
              f"{recs['C_clock'][key]['reset_turns']}")

    # metrics pipeline runs on the mock output
    from . import metrics5
    metrics5.select_tasks = lambda d, n=None: pool[d]
    metrics5.RESULTS_DIR = config5.RESULTS_DIR
    config5.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    import importlib
    out = metrics5.compute("gpt-oss-20b")
    check("metrics: pooled over all tasks", out["n_tasks"] == n_expected)
    check("metrics: routing contrasts present",
          any("anti-routing" in c["contrast"] for c in out["contrasts"]))


def main():
    tmp = Path(tempfile.mkdtemp(prefix="exp5_selftest_"))
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
