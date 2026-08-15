"""Runs trajectories: model x condition x task, saving per-turn records.

Early stop: a trajectory stops after the first hallucination turn (H).
Everything the metrics need — first canary failure S (only meaningful when
S <= H) and first hallucination H — is determined by then; if no
hallucination occurs the trajectory runs its full horizon.
"""
import asyncio
import json
import time

from .canaries import (SYSTEM_PROMPT, build_first_user_message, build_turn_body,
                       check_canary, check_hallucination)
from .config import (ALL_CONDITIONS, BASELINE_CONDITION, MODELS, RUNS_DIR,
                     TEMPERATURE, load_env)
from .llm import chat


def traj_path(model_name, condition, task_id):
    return RUNS_DIR / model_name / condition / f"task_{task_id:03d}.json"


async def run_trajectory(cfg, model_name, condition, task):
    path = traj_path(model_name, condition, task["task_id"])
    if path.exists():
        try:
            d = json.loads(path.read_text())
            if d.get("complete"):
                return d
        except json.JSONDecodeError:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    memo = {}
    records = []
    first_hallu = None
    first_canary_fail = None

    for turn in task["turns"]:
        t = turn["turn"]
        if t == 1:
            user_msg = build_first_user_message(task, condition)
        else:
            user_msg = build_turn_body(task, turn, condition)
        messages.append({"role": "user", "content": user_msg})
        content, usage = await chat(cfg, messages, temperature=TEMPERATURE)
        messages.append({"role": "assistant", "content": content})

        errors = check_hallucination(content, turn)
        hallu = len(errors) > 0
        canary_pass = None
        if condition != BASELINE_CONDITION:
            canary_pass = check_canary(condition, content, task, turn, memo)

        rec = {
            "turn": t,
            "user": user_msg,
            "assistant": content,
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "hallucination": hallu,
            "errors": [[r, e, g] for r, e, g in errors],
            "canary_pass": canary_pass,
        }
        records.append(rec)
        if hallu and first_hallu is None:
            first_hallu = t
        if canary_pass is False and first_canary_fail is None:
            first_canary_fail = t
        if first_hallu is not None:
            break  # early stop: first hallucination observed

    out = {
        "task_id": task["task_id"],
        "model": model_name,
        "condition": condition,
        "horizon": task["horizon"],
        "difficulty": task["difficulty"],
        "num_keys": task["num_keys"],
        "turns_run": len(records),
        "first_hallucination": first_hallu,
        "first_canary_fail": first_canary_fail,
        "records": records,
        "complete": True,
    }
    path.write_text(json.dumps(out))
    return out


async def run_all(tasks, model_names=None, conditions=None, task_limit=None, progress_every=25):
    load_env()
    model_names = model_names or list(MODELS)
    conditions = conditions or ALL_CONDITIONS
    if task_limit:
        tasks = tasks[:task_limit]

    sems = {m: asyncio.Semaphore(MODELS[m]["concurrency"]) for m in model_names}
    done = {"n": 0}
    total = len(model_names) * len(conditions) * len(tasks)
    t0 = time.time()

    async def one(m, c, task):
        async with sems[m]:
            try:
                r = await run_trajectory(MODELS[m], m, c, task)
            except Exception as e:
                print(f"[FAIL] {m}/{c}/task{task['task_id']}: {type(e).__name__}: {e}", flush=True)
                r = None
            done["n"] += 1
            if done["n"] % progress_every == 0 or done["n"] == total:
                el = time.time() - t0
                print(f"progress {done['n']}/{total} trajectories ({el/60:.1f} min)", flush=True)
            return r

    jobs = [one(m, c, task) for m in model_names for c in conditions for task in tasks]
    return await asyncio.gather(*jobs)
