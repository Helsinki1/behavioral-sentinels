"""Orchestrator for experiment 8. Resumable; only the three NEW arms run --
every baseline, bound, and passive-observational arm is read from runs6 (and
runs5 for A_no_reset) by the metrics layer, never re-run.

  python -m experiments8.run_all8 --gate       # QUIZ only
  python -m experiments8.run_all8              # QUIZ, ACT_carry_clock, ACT_probe
  python -m experiments8.run_all8 --arms X,Y   # specific arms
"""
import argparse
import asyncio
import json
import time

from experiments5.routing5 import GENRE_TO_PROBE
from experiments5.run_all5 import select_tasks

from .config8 import (ARM_ORDER, ARMS, DEFAULT_MODEL, DOMAINS, GATE_ARMS,
                      INTENDED_GENRE, MODELS, N_PER_DOMAIN, RUNS_DIR,
                      load_env)
from .harness8 import run_arm
from .policies8 import POLICIES


def traj_path(model, domain, arm, task_id):
    return RUNS_DIR / model / domain / arm / f"task_{task_id:03d}.json"


def load_arm(model, domain, arm, task_ids):
    out = {}
    for tid in task_ids:
        p = traj_path(model, domain, arm, tid)
        if p.exists():
            try:
                d = json.loads(p.read_text())
            except json.JSONDecodeError:
                continue
            if d.get("complete"):
                out[tid] = d
    return out


def condition_for(arm_name, domain):
    if ARMS[arm_name]["probe"] == "labeled":
        return GENRE_TO_PROBE[INTENDED_GENRE[domain]]
    return "baseline"


async def run_one(cfg, model, domain, task, arm_name):
    path = traj_path(model, domain, arm_name, task["task_id"])
    if path.exists():
        try:
            d = json.loads(path.read_text())
            if d.get("complete"):
                return d
        except json.JSONDecodeError:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    arm = ARMS[arm_name]
    decide = POLICIES[arm["policy"]](task)
    out = await run_arm(cfg, domain, task, arm_name,
                        condition_for(arm_name, domain), decide,
                        quiz=arm["policy"] == "quiz")
    path.write_text(json.dumps(out))
    return out


async def run_arm_all(model, arm_name, pool, progress_every=15):
    cfg = MODELS[model]
    sem = asyncio.Semaphore(cfg["concurrency"])
    jobs = [(d, t) for d in DOMAINS for t in pool[d]]
    done = {"n": 0}
    t0 = time.time()

    async def one(domain, task):
        async with sem:
            try:
                r = await run_one(cfg, model, domain, task, arm_name)
            except Exception as e:
                print(f"[FAIL] {arm_name}/{domain}/task{task['task_id']}: "
                      f"{type(e).__name__}: {e}", flush=True)
                r = None
            done["n"] += 1
            if done["n"] % progress_every == 0 or done["n"] == len(jobs):
                print(f"  {arm_name}: {done['n']}/{len(jobs)} "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)
            return r
    return await asyncio.gather(*[one(*j) for j in jobs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--arms", type=str, default=None)
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=N_PER_DOMAIN)
    args = ap.parse_args()

    load_env()
    pool = {d: select_tasks(d, args.limit) for d in DOMAINS}
    arms = (args.arms.split(",") if args.arms
            else GATE_ARMS if args.gate else ARM_ORDER)

    t0 = time.time()
    for arm_name in arms:
        print(f"=== arm {arm_name} (policy={ARMS[arm_name]['policy']}, "
              f"probe={ARMS[arm_name]['probe']}) ===", flush=True)
        asyncio.run(run_arm_all(args.model, arm_name, pool))
    print(f"=== done in {(time.time()-t0)/60:.1f} min ===", flush=True)


if __name__ == "__main__":
    main()
