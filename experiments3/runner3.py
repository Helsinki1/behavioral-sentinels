"""Runs experiment-3 trajectories: task_set x model x condition x task.

Same control flow as experiments2/runner2.py (early stop at the first
hallucination, resumable, per-turn records) with two additions: canaries are
scored in [0,1] rather than pass/fail (a failure is score < 1 on an applicable
turn), and the ensemble condition records per-member sub-scores.
"""
import asyncio
import json
import time

from experiments.llm import chat

from .canaries3 import (SYSTEM_PROMPTS, build_first_user_message,
                        build_turn_body, check_hallucination, score_canary)
from .config3 import (BASELINE_CONDITION, MODELS, RUNS_DIR, TEMPERATURE,
                      conditions_for, load_env)


def traj_path(task_set, model_name, condition, task_id):
    return RUNS_DIR / task_set / model_name / condition / f"task_{task_id:03d}.json"


async def run_trajectory(cfg, task_set, model_name, condition, task):
    path = traj_path(task_set, model_name, condition, task["task_id"])
    if path.exists():
        try:
            d = json.loads(path.read_text())
            if d.get("complete"):
                return d
        except json.JSONDecodeError:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)

    messages = [{"role": "system", "content": SYSTEM_PROMPTS[task_set]}]
    records = []
    first_hallu = None
    first_canary_fail = None

    for turn in task["turns"]:
        t = turn["turn"]
        user_msg = (build_first_user_message(task_set, task, condition) if t == 1
                    else build_turn_body(task_set, task, turn, condition))
        messages.append({"role": "user", "content": user_msg})
        content, usage = await chat(cfg, messages, temperature=TEMPERATURE)
        messages.append({"role": "assistant", "content": content})

        errors = check_hallucination(task_set, content, turn)
        hallu = len(errors) > 0
        canary = None
        if condition != BASELINE_CONDITION:
            canary = score_canary(condition, content, task, turn)

        rec = {
            "turn": t,
            "user": user_msg,
            "assistant": content,
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "hallucination": hallu,
            "errors": [list(e) for e in errors],
            "canary_score": canary["score"] if canary else None,
        }
        if canary:
            for k in ("subs", "fabricated", "rehearsed"):
                if k in canary:
                    rec[f"canary_{k}"] = canary[k]
        records.append(rec)

        if hallu and first_hallu is None:
            first_hallu = t
        if canary and canary["score"] is not None and canary["score"] < 1.0 \
                and first_canary_fail is None:
            first_canary_fail = t
        if first_hallu is not None:
            break  # early stop: first hallucination observed

    out = {
        "task_set": task_set,
        "task_id": task["task_id"],
        "model": model_name,
        "condition": condition,
        "horizon": task["horizon"],
        "difficulty": task.get("difficulty"),
        "turns_run": len(records),
        "first_hallucination": first_hallu,
        "first_canary_fail": first_canary_fail,
        "records": records,
        "complete": True,
    }
    path.write_text(json.dumps(out))
    return out


async def run_all(task_set, tasks, model_names=None, conditions=None,
                  task_limit=None, progress_every=25):
    load_env()
    model_names = model_names or list(MODELS)
    conditions = conditions or conditions_for(task_set)
    if task_limit:
        tasks = tasks[:task_limit]

    sems = {m: asyncio.Semaphore(MODELS[m]["concurrency"]) for m in model_names}
    done = {"n": 0}
    total = len(model_names) * len(conditions) * len(tasks)
    t0 = time.time()

    async def one(m, c, task):
        async with sems[m]:
            try:
                r = await run_trajectory(MODELS[m], task_set, m, c, task)
            except Exception as e:
                print(f"[FAIL] {task_set}/{m}/{c}/task{task['task_id']}: "
                      f"{type(e).__name__}: {e}", flush=True)
                r = None
            done["n"] += 1
            if done["n"] % progress_every == 0 or done["n"] == total:
                print(f"[{task_set}] progress {done['n']}/{total} trajectories "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)
            return r

    jobs = [one(m, c, task) for m in model_names for c in conditions for task in tasks]
    return await asyncio.gather(*jobs)
