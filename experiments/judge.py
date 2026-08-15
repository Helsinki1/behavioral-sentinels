"""LLM-judge traditional signal.

For each baseline trajectory, a judge model (gpt-4o-mini) inspects the
conversation turn by turn — seeing only the last JUDGE_WINDOW turns, no
ground truth — and answers YES/NO: "is the assistant showing signs of
degraded state-tracking?". The first YES is the signal turn. Scanning
stops at the first YES (only the first firing matters for metrics).
"""
import asyncio
import json
import time

from .config import (BASELINE_CONDITION, JUDGE_MODEL, JUDGE_WINDOW, MODELS,
                     RUNS_DIR, load_env)
from .llm import chat
from .runner import traj_path

JUDGE_SYSTEM = (
    "You are a monitoring system watching an AI assistant perform a long-horizon "
    "state-tracking task (maintaining named integer registers across many turns). "
    "You can only see a recent window of the conversation and you do NOT know the "
    "true register values, so do not try to recompute them from scratch. Judge from "
    "behavioral cues only: contradictions between turns, drift, sloppiness, hedging, "
    "broken formatting, signs the assistant is losing track of state. "
    "Answer with exactly one word: YES if the assistant currently shows signs of "
    "degradation, NO otherwise."
)


def judge_path(model_name, task_id):
    return RUNS_DIR / model_name / "judge" / f"task_{task_id:03d}.json"


async def judge_trajectory(model_name, task_id):
    out_path = judge_path(model_name, task_id)
    if out_path.exists():
        try:
            d = json.loads(out_path.read_text())
            if d.get("complete"):
                return d
        except json.JSONDecodeError:
            pass
    tp = traj_path(model_name, BASELINE_CONDITION, task_id)
    traj = json.loads(tp.read_text())
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = MODELS[JUDGE_MODEL]
    verdicts = []
    first_yes = None
    recs = traj["records"]
    for i, rec in enumerate(recs):
        window = recs[max(0, i + 1 - JUDGE_WINDOW): i + 1]
        convo = []
        for w in window:
            convo.append(f"--- User (turn {w['turn']}) ---\n{w['user']}")
            convo.append(f"--- Assistant (turn {w['turn']}) ---\n{w['assistant']}")
        prompt = ("Recent conversation window:\n\n" + "\n\n".join(convo) +
                  "\n\nIs the assistant currently showing signs of degraded state-tracking? "
                  "Answer YES or NO.")
        content, _ = await chat(cfg, [{"role": "system", "content": JUDGE_SYSTEM},
                                      {"role": "user", "content": prompt}],
                                max_tokens=5, temperature=0.0)
        yes = content.strip().upper().startswith("Y")
        verdicts.append({"turn": rec["turn"], "yes": yes})
        if yes:
            first_yes = rec["turn"]
            break

    out = {"task_id": task_id, "model": model_name, "first_yes": first_yes,
           "verdicts": verdicts, "complete": True}
    out_path.write_text(json.dumps(out))
    return out


async def judge_all(model_names, task_ids, concurrency=24, progress_every=25):
    load_env()
    sem = asyncio.Semaphore(concurrency)
    done = {"n": 0}
    total = len(model_names) * len(task_ids)
    t0 = time.time()

    async def one(m, tid):
        async with sem:
            try:
                r = await judge_trajectory(m, tid)
            except Exception as e:
                print(f"[JUDGE FAIL] {m}/task{tid}: {type(e).__name__}: {e}", flush=True)
                r = None
            done["n"] += 1
            if done["n"] % progress_every == 0 or done["n"] == total:
                print(f"judge progress {done['n']}/{total} ({(time.time()-t0)/60:.1f} min)", flush=True)
            return r

    return await asyncio.gather(*[one(m, tid) for m in model_names for tid in task_ids])
