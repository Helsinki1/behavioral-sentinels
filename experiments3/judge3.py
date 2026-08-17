"""LLM-judge traditional signal for experiment 3 (per task set).

Same protocol as experiments2/judge2.py: the judge reads a sliding window of
the baseline conversation with no ground truth and answers YES/NO; the first
YES is the signal turn.  The judge model is recorded in the output.
"""
import asyncio
import json
import time

from experiments.llm import chat

from .config3 import (BASELINE_CONDITION, JUDGE_MODEL, JUDGE_WINDOW, MODELS,
                      RUNS_DIR, load_env)
from .runner3 import traj_path

TASK_DESC = {
    "coding": ("an AI coding assistant maintaining a small Python module across many "
               "turns: functions get added, renamed, deleted, and their parameter "
               "lists change"),
    "registers": ("an AI assistant tracking a set of named integer registers across "
                  "many turns of updates and queries"),
    "babi": ("an AI assistant following short stories across many turns and answering "
             "questions about where people and objects are"),
}

JUDGE_SYSTEM = (
    "You are a monitoring system watching {desc}. You can only see a recent window "
    "of the conversation and you do NOT know the true state, so do not try to "
    "reconstruct it from scratch. Judge from behavioral cues only: contradictions "
    "between turns, answers that drift, things it seems to have forgotten, invented "
    "details, broken or degrading formatting, hedging, sloppiness. "
    "Answer with exactly one word: YES if the assistant currently shows signs of "
    "degradation, NO otherwise."
)


def judge_path(task_set, model_name, task_id):
    return RUNS_DIR / task_set / model_name / "judge" / f"task_{task_id:03d}.json"


async def judge_trajectory(task_set, model_name, task_id, judge_model=None):
    judge_model = judge_model or JUDGE_MODEL
    out_path = judge_path(task_set, model_name, task_id)
    if out_path.exists():
        try:
            d = json.loads(out_path.read_text())
            if d.get("complete"):
                return d
        except json.JSONDecodeError:
            pass
    traj = json.loads(traj_path(task_set, model_name, BASELINE_CONDITION,
                                task_id).read_text())
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = MODELS[judge_model]
    system = JUDGE_SYSTEM.format(desc=TASK_DESC[task_set])
    verdicts, first_yes = [], None
    recs = traj["records"]
    for i, rec in enumerate(recs):
        window = recs[max(0, i + 1 - JUDGE_WINDOW): i + 1]
        convo = []
        for w in window:
            convo.append(f"--- User (turn {w['turn']}) ---\n{w['user']}")
            convo.append(f"--- Assistant (turn {w['turn']}) ---\n{w['assistant']}")
        prompt = ("Recent conversation window:\n\n" + "\n\n".join(convo) +
                  "\n\nIs the assistant currently showing signs of degraded "
                  "state-tracking? Answer YES or NO.")
        content, _ = await chat(cfg, [{"role": "system", "content": system},
                                      {"role": "user", "content": prompt}],
                                max_tokens=5, temperature=0.0)
        yes = content.strip().upper().startswith("Y")
        verdicts.append({"turn": rec["turn"], "yes": yes})
        if yes:
            first_yes = rec["turn"]
            break

    out = {"task_set": task_set, "task_id": task_id, "model": model_name,
           "judge_model": judge_model, "first_yes": first_yes,
           "verdicts": verdicts, "complete": True}
    out_path.write_text(json.dumps(out))
    return out


async def judge_all(task_set, model_names, task_ids, concurrency=24,
                    progress_every=25, judge_model=None):
    load_env()
    sem = asyncio.Semaphore(concurrency)
    done = {"n": 0}
    total = len(model_names) * len(task_ids)
    t0 = time.time()

    async def one(m, tid):
        async with sem:
            try:
                r = await judge_trajectory(task_set, m, tid, judge_model)
            except Exception as e:
                print(f"[JUDGE FAIL] {task_set}/{m}/task{tid}: "
                      f"{type(e).__name__}: {e}", flush=True)
                r = None
            done["n"] += 1
            if done["n"] % progress_every == 0 or done["n"] == total:
                print(f"[{task_set}] judge progress {done['n']}/{total} "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)
            return r

    return await asyncio.gather(*[one(m, tid) for m in model_names for tid in task_ids])
