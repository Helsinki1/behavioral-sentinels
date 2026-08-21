"""Orchestrator for experiment 9: seven arms x four models on the sharded-
math session pool. Resumable per (model, arm, session).

  python -m experiments9.run_all9 --model gpt-oss-120b --gate
  python -m experiments9.run_all9 --model gpt-oss-120b
  python -m experiments9.run_all9              # every model in MODEL_ORDER
"""
import argparse
import asyncio
import json
import time

from experiments8.policies8 import POLICIES

from .config9 import (ARMS, ARM_ORDER, DATA_DIR, DOMAIN, GATE_ARMS,
                      MODEL_ORDER, MODELS, RUNS_DIR, load_env)
from .harness9 import run_arm


def load_pool():
    return json.loads((DATA_DIR / "tasks9_shardmath.json").read_text())


def traj_path(model, arm, task_id):
    return RUNS_DIR / model / DOMAIN / arm / f"task_{task_id:03d}.json"


def load_arm(model, arm, task_ids):
    out = {}
    for tid in task_ids:
        p = traj_path(model, arm, tid)
        if p.exists():
            try:
                d = json.loads(p.read_text())
            except json.JSONDecodeError:
                continue
            if d.get("complete"):
                out[tid] = d
    return out


async def run_one(cfg, model, task, arm_name, extras):
    path = traj_path(model, arm_name, task["task_id"])
    if path.exists():
        try:
            d = json.loads(path.read_text())
            if d.get("complete"):
                return d
        except json.JSONDecodeError:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    arm = ARMS[arm_name]
    decide = POLICIES[arm["policy"]](task, **extras)
    out = await run_arm(cfg, task, arm_name, arm["probe"], decide,
                        quiz=arm["policy"] == "quiz")
    path.write_text(json.dumps(out))
    return out


async def run_arm_all(model, arm_name, pool, progress_every=10):
    cfg = MODELS[model]
    sem = asyncio.Semaphore(cfg["concurrency"])
    ids = [t["task_id"] for t in pool]
    extras = {tid: {} for tid in ids}
    if ARMS[arm_name]["policy"] == "oracle":
        base = load_arm(model, "A_no_reset", ids)
        if len(base) < len(ids):
            print(f"[skip] {arm_name}/{model}: needs A_no_reset first "
                  f"({len(base)}/{len(ids)})", flush=True)
            return
        extras = {tid: {"oracle_turn": base[tid]["first_hallucination"]}
                  for tid in ids}
    done = {"n": 0}
    t0 = time.time()

    async def one(task):
        async with sem:
            try:
                r = await run_one(cfg, model, task, arm_name,
                                  extras[task["task_id"]])
            except Exception as e:
                print(f"[FAIL] {model}/{arm_name}/task{task['task_id']}: "
                      f"{type(e).__name__}: {e}", flush=True)
                r = None
            done["n"] += 1
            if done["n"] % progress_every == 0 or done["n"] == len(pool):
                print(f"  {model}/{arm_name}: {done['n']}/{len(pool)} "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)
            return r
    return await asyncio.gather(*[one(t) for t in pool])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--arms", type=str, default=None)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    load_env()
    pool = load_pool()
    if args.limit:
        pool = pool[:args.limit]
    arms = (args.arms.split(",") if args.arms
            else GATE_ARMS if args.gate else ARM_ORDER)
    models = [args.model] if args.model else MODEL_ORDER

    t0 = time.time()
    for model in models:
        for arm_name in arms:
            print(f"=== {model} / {arm_name} "
                  f"(policy={ARMS[arm_name]['policy']}, "
                  f"probe={ARMS[arm_name]['probe']}) ===", flush=True)
            asyncio.run(run_arm_all(model, arm_name, pool))
    print(f"=== done in {(time.time()-t0)/60:.1f} min ===", flush=True)


if __name__ == "__main__":
    main()
