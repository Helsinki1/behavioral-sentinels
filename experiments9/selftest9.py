"""Offline selftest for experiment 9. No network; the mock answers from
generator truth with scripted degradations:

  turn 2            premature ANSWER   (task hallucination + monitor fire)
  episode index 1   wrong final ANSWER (task hallucination on that final turn)
  turn 5            wrong ECHO         (probe failure when lag_span carried)
  checkpoint 6      wrong quiz answers (quiz failure)

  python -m experiments9.selftest9
"""
import asyncio
import contextvars
import json
import re
import shutil
import tempfile
from pathlib import Path

from . import build9, config9, domain9, harness9, metrics9, run_all9, shadow9
from .config9 import ARMS
from experiments8.quiz8 import QUIZ_MARKER

FAILURES = []
CURRENT_TASK = contextvars.ContextVar("exp9_selftest_task", default=None)

PREMATURE_TURN = 2
BAD_FINAL_EPISODE = 1
PROBE_BAD_TURNS = {5}
QUIZ_BAD_TURNS = {6}


def check(name, ok, detail=""):
    print(("  ok  " if ok else "  FAIL") + f" {name}" +
          (f" -- {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append((name, detail))


def _fmt_expect(q):
    e = q["expect"]
    return str(e)


class MockLLM:
    def __init__(self, pool):
        self.tasks = {t["task_id"]: t for t in pool}

    def _quiz_reply(self, task, t, bad):
        store = domain9.make_store(task)
        for turn in task["turns"]:
            if turn["turn"] > t:
                break
            store.apply(turn)
        qs = domain9.quiz_questions(task, store, t)
        if bad:
            return "\n".join(f"A{i}: wrongling 99999"
                             for i, q in enumerate(qs, 1)
                             if q["grade"] is not None)
        return "\n".join(f"A{i}: {_fmt_expect(q)}"
                         for i, q in enumerate(qs, 1)
                         if q["grade"] is not None)

    async def chat(self, cfg, messages, max_tokens=None, temperature=0.2):
        last = messages[-1]["content"]
        tid = CURRENT_TASK.get()
        task = self.tasks[tid]
        if QUIZ_MARKER in last:
            t = int(re.search(r"after task turn (\d+)", last).group(1))
            return (self._quiz_reply(task, t, t in QUIZ_BAD_TURNS),
                    {"prompt_tokens": 150, "completion_tokens": 25})
        t = int(re.search(r"Turn (\d+):", last).group(1))
        turn = task["turns"][t - 1]
        lines = []
        joined = "\n".join(m["content"] for m in messages)
        if "ECHO" in joined:                      # lag_span carried
            c3 = turn["c3"]
            vals = ([c3["lag1"], c3["lag3"], c3["lag6"]]
                    if t not in PROBE_BAD_TURNS else ["ENG-0000"] * 3)
            lines.append("ECHO: " + " ".join(vals))
        if turn["final"]:
            ans = ("999999" if turn["episode"] == BAD_FINAL_EPISODE
                   else turn["answer"])
            lines.append(f"ANSWER: {ans}")
        elif t == PREMATURE_TURN:
            lines.append("ANSWER: 123")
        else:
            lines.append("WAIT")
        return "\n".join(lines), {"prompt_tokens": 50 * len(messages),
                                  "completion_tokens": 12}


def unit_tests(pool):
    check("build: 34 sessions", len(pool) == config9.N_SESSIONS,
          str(len(pool)))
    hz = [t["horizon"] for t in pool]
    check("build: horizons in a long-session range",
          min(hz) >= 12 and max(hz) <= 36, str((min(hz), max(hz))))
    task = pool[0]
    finals = [t for t in task["turns"] if t["final"]]
    check("build: one final per episode",
          len(finals) == task["n_episodes"])
    check("build: answers only on finals",
          all((t["answer"] is not None) == t["final"]
              for t in task["turns"]))
    check("build: gold answers numeric",
          all(domain9._num(f["answer"]) is not None for f in finals))

    fin = finals[0]
    non = next(t for t in task["turns"] if not t["final"])
    check("checker: correct final accepted",
          domain9.check_hallucination(f"ANSWER: {fin['answer']}", fin) == [])
    check("checker: wrong final flagged",
          domain9.check_hallucination("ANSWER: 31337.5", fin) != [])
    check("checker: missing final flagged",
          domain9.check_hallucination("WAIT", fin) != [])
    check("checker: WAIT accepted on non-final",
          domain9.check_hallucination("WAIT", non) == [])
    check("checker: premature answer flagged",
          any(k == "premature_answer" for k, _ in
              domain9.check_hallucination("ANSWER: 5", non)))
    check("checker: babble without WAIT flagged",
          domain9.check_hallucination("thinking about it", non) != [])

    mon = domain9.make_monitor()
    body_non = domain9.build_turn_body(task, non, "baseline")
    body_fin = domain9.build_turn_body(task, fin, "baseline")
    check("monitor: quiet on clean WAIT", mon.check(body_non, "WAIT") == [])
    check("monitor: fires on premature answer",
          mon.check(body_non, "ANSWER: 5") != [])
    check("monitor: fires on missing final answer",
          mon.check(body_fin, "WAIT") != [])
    check("monitor: quiet on final answer (no key consulted)",
          mon.check(body_fin, "ANSWER: 31337") == [])

    store = domain9.make_store(task)
    for turn in task["turns"][:2]:
        store.apply(turn)
    resume = domain9.resume_reground(task, store, 3, "baseline")
    check("resume: carries current shards mid-problem",
          all(s in resume for s in store.shards))
    check("resume: reports completed count",
          f"Problems fully answered so far: {store.completed}" in resume)
    carried = domain9.resume_reground(task, store, 6, "lag_span")
    check("resume: carried arm restates rule + ticket log",
          "ECHO" in carried and "Ticket:" in carried)

    qs = domain9.quiz_questions(task, store, 5)
    check("quiz: three questions", len(qs) == 3)


def integration(tmp, pool):
    mock = MockLLM(pool)
    harness9.chat = mock.chat
    domain9.chat = mock.chat
    run_all9.RUNS_DIR = tmp / "runs9"
    shadow9.RUNS_DIR = tmp / "runs9"
    metrics9.RESULTS_DIR = tmp / "results9"

    small = pool[:2]

    orig_run_one = run_all9.run_one

    async def tagged_run_one(cfg, model, task, arm_name, extras):
        CURRENT_TASK.set(task["task_id"])
        return await orig_run_one(cfg, model, task, arm_name, extras)
    run_all9.run_one = tagged_run_one

    orig_shadow_one = shadow9.shadow_one

    async def tagged_shadow_one(cfg, model, task):
        CURRENT_TASK.set(task["task_id"])
        return await orig_shadow_one(cfg, model, task)
    shadow9.shadow_one = tagged_shadow_one

    model = "gpt-oss-120b"
    for arm in config9.ARM_ORDER:
        asyncio.run(run_all9.run_arm_all(model, arm, small,
                                         progress_every=10**9))
    asyncio.run(shadow9.run_all(model, small, progress_every=10**9))

    ids = [t["task_id"] for t in small]
    recs = {arm: run_all9.load_arm(model, arm, ids) for arm in ARMS}
    for arm in ARMS:
        check(f"{arm}: {len(small)} trajectories", len(recs[arm]) == len(small))

    for tid, r in recs["A_no_reset"].items():
        check(f"A {tid}: premature turn is first hallucination",
              r["first_hallucination"] == PREMATURE_TURN)
        bad_finals = [t["turn"] for t in small[tid]["turns"]
                      if t["final"] and t["episode"] == BAD_FINAL_EPISODE]
        errs = {rec["turn"] for rec in r["records"] if rec["hallucination"]}
        check(f"A {tid}: bad-episode final flagged",
              all(b in errs for b in bad_finals))
    for tid, r in recs["C_clock"].items():
        want = [t for t in range(6, r["horizon"] + 1, 6)][:config9.MAX_RESETS]
        check(f"C_clock {tid}: cadence", r["reset_turns"] == want,
              str(r["reset_turns"]))
    for tid, r in recs["Z_trace"].items():
        check(f"Z_trace {tid}: reset after premature answer",
              r["reset_turns"][:1] == [PREMATURE_TURN + 1],
              str(r["reset_turns"]))
    for tid, r in recs["QUIZ"].items():
        got = [q["turn"] for q in r["quizzes"]]
        want = [t for t in range(3, r["horizon"], 3)]
        check(f"QUIZ {tid}: checkpoints", got == want, f"{got} vs {want}")
        check(f"QUIZ {tid}: reset after failed checkpoint",
              7 in r["reset_turns"], str(r["reset_turns"]))
    for tid, r in recs["ACT_probe"].items():
        check(f"ACT_probe {tid}: probe scored",
              any(rec["probe_score"] is not None for rec in r["records"]))
        check(f"ACT_probe {tid}: reset after probe failure",
              6 in r["reset_turns"], str(r["reset_turns"]))
    for tid, r in recs["F_oracle"].items():
        check(f"F_oracle {tid}: reset at oracle turn",
              r["reset_turns"] == [PREMATURE_TURN], str(r["reset_turns"]))

    shadows = shadow9.load_shadow(model, ids)
    check("shadow: all present", len(shadows) == len(small))
    for tid, s in shadows.items():
        check(f"shadow {tid}: bad checkpoint fails, others clean",
              all(c["fail"] == (c["turn"] in QUIZ_BAD_TURNS)
                  for c in s["checkpoints"]),
              str([(c['turn'], c['n_wrong']) for c in s['checkpoints']]))

    out, pred = metrics9.compute()
    check("metrics: model present", model in out["models"])
    check("metrics: observer contrast present",
          any(c["contrast"].startswith("ACT_carry_clock - C_clock")
              for c in out["models"][model]["contrasts"]))
    check("prediction: shadow quiz row on A set",
          "frozen-state quiz (shadow)"
          in pred["models"][model]["sets"]["A_no_reset"]["signals"])


def main():
    pool = build9.build()
    tmp = Path(tempfile.mkdtemp(prefix="exp9_selftest_"))
    try:
        print("== unit tests ==")
        unit_tests(pool)
        print("== integration (mock LLM, all arms) ==")
        integration(tmp, pool)
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
