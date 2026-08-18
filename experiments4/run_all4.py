"""Orchestrator for experiment 4. Resumable; arms are run in dependency order.

  python -m experiments4.run_all4 --gate      # A, C, F only  (go/no-go)
  python -m experiments4.run_all4             # all six arms
"""
import argparse
import asyncio
import json
import time

from .config4 import (ARMS, DATA_DIR, GATE_ARMS, MODELS, N_TASKS, RUNS_DIR,
                      SEED, load_env)
from .harness4 import run_arm
from .policies4 import POLICIES


def traj_path(model, arm, task_id):
    return RUNS_DIR / model / arm / f"task_{task_id:03d}.json"


def load_arm(model, arm, task_ids):
    out = {}
    for tid in task_ids:
        p = traj_path(model, arm, tid)
        if p.exists():
            d = json.loads(p.read_text())
            if d.get("complete"):
                out[tid] = d
    return out


async def run_one(cfg, model, task, arm_name, extra):
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
    decide = POLICIES[arm["policy"]](task, **extra)
    out = await run_arm(cfg, task, arm_name, arm, decide)
    path.write_text(json.dumps(out))
    return out


async def run_arm_all(model, arm_name, tasks, extras, progress_every=10):
    cfg = MODELS[model]
    sem = asyncio.Semaphore(cfg["concurrency"])
    done = {"n": 0}
    t0 = time.time()

    async def one(task):
        async with sem:
            try:
                r = await run_one(cfg, model, task, arm_name,
                                  extras.get(task["task_id"], {}))
            except Exception as e:
                print(f"[FAIL] {arm_name}/task{task['task_id']}: "
                      f"{type(e).__name__}: {e}", flush=True)
                r = None
            done["n"] += 1
            if done["n"] % progress_every == 0 or done["n"] == len(tasks):
                print(f"  {arm_name}: {done['n']}/{len(tasks)} "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)
            return r
    return await asyncio.gather(*[one(t) for t in tasks])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true", help="run only A, C, F")
    ap.add_argument("--arms", type=str, default=None)
    ap.add_argument("--model", type=str, default="gpt-oss-20b")
    ap.add_argument("--limit", type=int, default=N_TASKS)
    args = ap.parse_args()

    load_env()
    tasks = json.loads((DATA_DIR / "tasks2.json").read_text())[:args.limit]
    ids = [t["task_id"] for t in tasks]
    model = args.model
    arms = (args.arms.split(",") if args.arms
            else GATE_ARMS if args.gate else list(ARMS))

    t0 = time.time()
    for arm_name in arms:
        extras = {}
        if ARMS[arm_name]["policy"] == "oracle":
            base = load_arm(model, "A_no_reset", ids)
            if len(base) < len(ids):
                print(f"[skip] {arm_name}: needs A_no_reset first "
                      f"({len(base)}/{len(ids)} present)", flush=True)
                continue
            extras = {tid: {"oracle_turn": base[tid]["first_hallucination"]}
                      for tid in ids}
        if ARMS[arm_name]["policy"] == "random":
            sent = load_arm(model, "D_sentinel", ids)
            if len(sent) < len(ids):
                print(f"[skip] {arm_name}: needs D_sentinel first for the "
                      f"reset budget ({len(sent)}/{len(ids)})", flush=True)
                continue
            extras = {tid: {"n_resets": sent[tid]["n_resets"], "seed": SEED,
                            "horizon": sent[tid]["horizon"]} for tid in ids}
        print(f"=== arm {arm_name} ({ARMS[arm_name]['policy']}, "
              f"sentinel={ARMS[arm_name]['carries_sentinel']}) ===", flush=True)
        asyncio.run(run_arm_all(model, arm_name, tasks, extras))
    print(f"=== done in {(time.time()-t0)/60:.1f} min ===", flush=True)


if __name__ == "__main__":
    main()
