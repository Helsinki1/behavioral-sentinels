"""Offline selftest for experiment 8. No network: task turns are answered by
experiments5.selftest5.MockLLM; quiz forks are answered by a wrapper that
rebuilds the store from generator truth and replies correctly except on
scripted checkpoint turns.

  python -m experiments8.selftest8

Verifies:
  - quiz question generation + deterministic grading (right/wrong/threshold)
  - the carried-arm reground resume: baseline state + standing rule +
    verbatim payload log, nothing else
  - the QUIZ arm quizzes on the dense cadence, discards the fork (agent
    messages never contain the quiz), and resets after a failed checkpoint
  - ACT_probe resets on probe failure; ACT_carry_clock resets on the clock
    while carrying the probe
  - the shadow pass reproduces prefix conversations and grades checkpoints
  - metrics8 aggregation over new (runs8) + reused (runs6) arms, incl. the
    observer-cost contrast and the same-trajectory prediction table
"""
import asyncio
import json
import re
import shutil
import tempfile
from pathlib import Path

from experiments3.canaries3 import instruction_text
from experiments5.selftest5 import BAD_TURNS, CURRENT_TASK, MockLLM
from experiments5.run_all5 import select_tasks
from experiments6 import store6

from . import config8, harness8, metrics8, quiz8, run_all8, shadow8
from .config8 import ARMS, DOMAINS
from .quiz8 import (QUIZ_MARKER, build_quiz_message, grade_quiz,
                    quiz_questions)

FAILURES = []
QUIZ_BAD_TURNS = {6}       # checkpoints answered wrongly by the mock


def check(name, ok, detail=""):
    print(("  ok  " if ok else "  FAIL") + f" {name}" +
          (f" -- {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append((name, detail))


def _fmt_expect(q):
    e = q["expect"]
    if isinstance(e, dict):                      # {fn: params}
        ps = next(iter(e.values()))
        return ", ".join(ps) if ps else "()"
    if isinstance(e, list):                      # deleted names
        return ", ".join(e) if e else "NONE"
    return str(e)


def correct_reply(questions):
    return "\n".join(f"A{i}: {_fmt_expect(q)}"
                     for i, q in enumerate(questions, 1)
                     if q["grade"] is not None)


def wrong_reply(questions):
    return "\n".join(f"A{i}: wrongling 99999"
                     for i, q in enumerate(questions, 1)
                     if q["grade"] is not None)


def unit_tests():
    for domain in DOMAINS:
        tasks = json.loads(
            (config8.DATA_DIR / f"tasks3_{domain}.json").read_text())
        task = tasks[0]
        store = store6.make_store(domain, task)
        for turn in task["turns"][:6]:
            store.apply(turn)
        qs = quiz_questions(domain, task, store, 6)
        check(f"{domain}: three questions generated", len(qs) == 3)
        qs2 = quiz_questions(domain, task, store, 6)
        check(f"{domain}: questions deterministic",
              [q["text"] for q in qs] == [q["text"] for q in qs2])
        corr, n_wrong, fail = grade_quiz(qs, correct_reply(qs))
        check(f"{domain}: correct reply grades clean",
              n_wrong == 0 and not fail, f"{corr}")
        corr, n_wrong, fail = grade_quiz(qs, wrong_reply(qs))
        n_app = sum(1 for q in qs if q["grade"] is not None)
        check(f"{domain}: wrong reply grades all-wrong and fails",
              n_wrong == n_app and fail, f"{corr}")
        corr, n_wrong, fail = grade_quiz(qs, "")
        check(f"{domain}: empty reply fails", fail, f"{corr}")
        msg = build_quiz_message(qs, 6)
        check(f"{domain}: quiz message carries marker + turn",
              QUIZ_MARKER in msg and "after task turn 6" in msg)

    # carried-arm resume: rule + payload log appended to the baseline resume
    tasks = json.loads((config8.DATA_DIR / "tasks3_registers.json").read_text())
    task = tasks[0]
    store = store6.make_store("registers", task)
    for turn in task["turns"][:5]:
        store.apply(turn)
    base = harness8.reset_messages("registers", task, store, 6, "baseline")
    carried = harness8.reset_messages("registers", task, store, 6,
                                      "chain_checksum")
    check("baseline resume == store6 resume",
          base[1]["content"] == store6.resume_reground("registers", task,
                                                       store, 6))
    rule = instruction_text("chain_checksum", task)
    check("carried resume contains the standing rule",
          rule in carried[1]["content"])
    keys = [t["c3"]["key"] for t in task["turns"][:5]]
    check("carried resume logs every prior KEY line",
          all(f"KEY: {k}" in carried[1]["content"] for k in keys))
    check("carried resume logs no future payload",
          "Turn 6:" not in carried[1]["content"].split("Standing-rule")[1])

    check("quiz cadence: due at 3,6,9; not 4; not at horizon",
          harness8.quiz_due(3, 15) and harness8.quiz_due(6, 15)
          and harness8.quiz_due(9, 15) and not harness8.quiz_due(4, 15)
          and not harness8.quiz_due(15, 15))


class QuizAwareChat:
    """Routes quiz forks to a truth-based responder, everything else to the
    exp-5 MockLLM."""

    def __init__(self, mock, pool):
        self.mock = mock
        self.task_index = {(d, t["task_id"]): t
                           for d, ts in pool.items() for t in ts}
        self.quiz_calls = []

    async def chat(self, cfg, messages, max_tokens=None, temperature=0.2):
        last = messages[-1]["content"]
        if QUIZ_MARKER in last:
            key = CURRENT_TASK.get()
            t = int(re.search(r"after task turn (\d+)", last).group(1))
            self.quiz_calls.append((key, t))
            task = self.task_index[key]
            store = store6.make_store(key[0], task)
            for turn in task["turns"]:
                if turn["turn"] > t:
                    break
                store.apply(turn)
            qs = quiz_questions(key[0], task, store, t)
            reply = (wrong_reply(qs) if t in QUIZ_BAD_TURNS
                     else correct_reply(qs))
            return reply, {"prompt_tokens": 200, "completion_tokens": 30}
        return await self.mock.chat(cfg, messages, max_tokens=max_tokens,
                                    temperature=temperature)


def integration(tmp):
    pool = {d: select_tasks(d, 2) for d in DOMAINS}
    mock = MockLLM(pool)
    wrapper = QuizAwareChat(mock, pool)
    harness8.chat = wrapper.chat
    quiz8.chat = wrapper.chat
    run_all8.RUNS_DIR = tmp / "runs8"
    shadow8.RUNS_DIR = tmp / "runs8"
    shadow8.RUNS5_DIR = tmp / "runs5"
    metrics8.RUNS6_DIR = tmp / "runs6"
    metrics8.RESULTS_DIR = tmp / "results8"
    metrics8.select_tasks = lambda d, n=None: pool[d]

    orig_run_one = run_all8.run_one

    async def tagged_run_one(cfg, model, domain, task, arm_name):
        CURRENT_TASK.set((domain, task["task_id"]))
        return await orig_run_one(cfg, model, domain, task, arm_name)
    run_all8.run_one = tagged_run_one

    orig_shadow_one = shadow8.shadow_one

    async def tagged_shadow_one(cfg, model, domain, task):
        CURRENT_TASK.set((domain, task["task_id"]))
        return await orig_shadow_one(cfg, model, domain, task)
    shadow8.shadow_one = tagged_shadow_one

    # fabricate runs5 A_no_reset + the reused runs6 arms from one baseline
    # no-reset pass per task (mirrors selftest6's plumbing trick)
    async def gen_bases():
        for domain in DOMAINS:
            for task in pool[domain]:
                CURRENT_TASK.set((domain, task["task_id"]))
                out = await harness8.run_arm({}, domain, task, "A_no_reset",
                                             "baseline", lambda s: False)
                p5 = (tmp / "runs5" / "gpt-oss-20b" / domain / "A_no_reset"
                      / f"task_{task['task_id']:03d}.json")
                p5.parent.mkdir(parents=True, exist_ok=True)
                p5.write_text(json.dumps(out))
                for arm6 in config8.REUSED_ARMS:
                    p6 = (tmp / "runs6" / "gpt-oss-20b" / domain / arm6
                          / f"task_{task['task_id']:03d}.json")
                    p6.parent.mkdir(parents=True, exist_ok=True)
                    p6.write_text(json.dumps({**out, "arm": arm6}))
    asyncio.run(gen_bases())

    for arm in config8.ARM_ORDER:
        asyncio.run(run_all8.run_arm_all("gpt-oss-20b", arm, pool,
                                         progress_every=10**9))
    asyncio.run(shadow8.run_all("gpt-oss-20b", pool, progress_every=10**9))

    ids = {d: [t["task_id"] for t in pool[d]] for d in DOMAINS}
    recs = {arm: {(d, tid): r for d in DOMAINS
                  for tid, r in run_all8.load_arm(
                      "gpt-oss-20b", d, arm, ids[d]).items()}
            for arm in ARMS}
    n_expected = sum(len(v) for v in ids.values())
    for arm in ARMS:
        check(f"{arm}: all {n_expected} trajectories complete",
              len(recs[arm]) == n_expected, str(len(recs[arm])))

    tasks_by_key = {(d, t["task_id"]): t for d in DOMAINS for t in pool[d]}

    for key, r in recs["QUIZ"].items():
        horizon = r["horizon"]
        expect_cps = [t for t in range(3, horizon, 3)]
        got_cps = [q["turn"] for q in r["quizzes"]]
        # a reset consumes no checkpoint; every due turn must have one
        check(f"QUIZ {key}: checkpoints on the dense cadence",
              got_cps == expect_cps, f"{got_cps} vs {expect_cps}")
        check(f"QUIZ {key}: scripted bad checkpoint failed",
              all(q["fail"] for q in r["quizzes"]
                  if q["turn"] in QUIZ_BAD_TURNS))
        check(f"QUIZ {key}: clean checkpoints passed",
              all(not q["fail"] for q in r["quizzes"]
                  if q["turn"] not in QUIZ_BAD_TURNS),
              str([(q['turn'], q['correct']) for q in r['quizzes']]))
        check(f"QUIZ {key}: reset right after the failed checkpoint",
              r["reset_turns"] == [7], str(r["reset_turns"]))
        check(f"QUIZ {key}: quiz tokens accounted separately",
              r["quiz_prompt_tokens"] > 0 and r["prompt_tokens"] > 0)
        # the fork is discarded: no assistant record contains quiz answers
        check(f"QUIZ {key}: no quiz content in agent replies",
              all("A1:" not in rec["assistant"] for rec in r["records"]))
        task = tasks_by_key[key]
        ref = store6.make_store(key[0], task)
        for turn in task["turns"]:
            if turn["turn"] == 7:
                want = store6.resume_reground(key[0], task, ref, 7)
                check(f"QUIZ {key}: resume == store state",
                      r["resets"][0]["resume"] == want)
            ref.apply(turn)

    for key, r in recs["ACT_probe"].items():
        check(f"ACT_probe {key}: probe carried (scores recorded)",
              any(rec["probe_score"] is not None for rec in r["records"]))
        check(f"ACT_probe {key}: reset after first probe failure",
              r["reset_turns"][:1] == [6], str(r["reset_turns"]))
        resume = r["resets"][0]["resume"]
        check(f"ACT_probe {key}: resume restates the standing rule",
              "One more standing rule" in resume)
        check(f"ACT_probe {key}: resume carries the payload log",
              "Standing-rule payload lines" in resume)

    for key, r in recs["ACT_carry_clock"].items():
        want = [t for t in range(6, r["horizon"] + 1, 6)][:config8.MAX_RESETS]
        check(f"ACT_carry_clock {key}: clock cadence while carrying",
              r["reset_turns"] == want, str(r["reset_turns"]))
        check(f"ACT_carry_clock {key}: probe recorded",
              any(rec["probe_score"] is not None for rec in r["records"]))

    shadows = {(d, tid): s for d in DOMAINS
               for tid, s in shadow8.load_shadow(
                   "gpt-oss-20b", d, ids[d]).items()}
    check(f"shadow: all {n_expected} tasks", len(shadows) == n_expected,
          str(len(shadows)))
    for key, s in shadows.items():
        horizon = s["horizon"]
        expect_cps = [t for t in range(3, horizon, 3)]
        check(f"shadow {key}: checkpoints on cadence",
              [c["turn"] for c in s["checkpoints"]] == expect_cps)
        check(f"shadow {key}: scripted bad checkpoint failed, others clean",
              all(c["fail"] == (c["turn"] in QUIZ_BAD_TURNS)
                  for c in s["checkpoints"]),
              str([(c['turn'], c['n_wrong']) for c in s['checkpoints']]))

    out, pred = metrics8.compute("gpt-oss-20b")
    check("metrics: pooled over all tasks", out["n_tasks"] == n_expected,
          str(out["n_tasks"]))
    check("metrics: observer-cost contrast present",
          any(c["contrast"].startswith("ACT_carry_clock - C_clock")
              for c in out["contrasts"]))
    check("metrics: QUIZ vs oracle contrast present",
          any(c["contrast"].startswith("QUIZ - F_oracle")
              for c in out["contrasts"]))
    check("prediction: same-trajectory table includes the shadow quiz",
          "frozen-state quiz (shadow)"
          in pred["sets"]["A_no_reset"]["signals"])
    check("prediction: clean active read present",
          "carried probe (clock-truncated read)"
          in pred["sets"].get("ACT_carry_clock", {}).get("signals", {}))
    check("prediction: quiz threshold ablation present",
          len(pred["shadow"].get("signals", {})) == 3)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="exp8_selftest_"))
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
