"""Orchestrator for experiment 6. Resumable; arms run in dependency order.

  python -m experiments6.run_all6 --gate      # A (import), C_clock, F_oracle
  python -m experiments6.run_all6             # all arms
  python -m experiments6.run_all6 --arms X,Y  # specific arms

A_no_reset is IMPORTED from runs5 (identical protocol, identical tasks): it
never resets, so its trajectories are operator-independent, and reusing them
makes the exp-5-vs-exp-6 operator contrasts exactly paired.
"""
import argparse
import asyncio
import json
import shutil
import time

from experiments5.policies5 import POLICIES
from experiments5.run_all5 import select_tasks

from .config6 import (ARM_ORDER, ARMS, DEFAULT_MODEL, DOMAINS, GATE_ARMS,
                      JUDGE_MODEL, MODELS, N_PER_DOMAIN, RUNS5_DIR, RUNS_DIR,
                      SEED, load_env)
from .harness6 import run_arm


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


def import_A(model, pool):
    """Copy runs5 A_no_reset trajectories into runs6 (skip ones we have)."""
    copied = missing = present = 0
    for domain in DOMAINS:
        for task in pool[domain]:
            dst = traj_path(model, domain, "A_no_reset", task["task_id"])
            if dst.exists():
                present += 1
                continue
            src = (RUNS5_DIR / model / domain / "A_no_reset"
                   / f"task_{task['task_id']:03d}.json")
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, dst)
                copied += 1
            else:
                missing += 1
    print(f"A_no_reset import: {copied} copied, {present} already present, "
          f"{missing} missing from runs5", flush=True)
    return missing


async def run_one(cfg, model, domain, task, arm_name, extras):
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
    decide = POLICIES[arm["policy"]](task, **{**arm.get("extras", {}), **extras})
    judge_cfg = MODELS[JUDGE_MODEL] if arm["policy"] == "judge" else None
    out = await run_arm(cfg, domain, task, arm_name, arm["operator"], decide,
                        judge_cfg=judge_cfg)
    path.write_text(json.dumps(out))
    return out


async def run_arm_all(model, arm_name, pool, progress_every=15):
    if arm_name == "A_no_reset":
        return  # imported, never run
    cfg = MODELS[model]
    sem = asyncio.Semaphore(cfg["concurrency"])
    jobs = []
    for domain in DOMAINS:
        ids = [t["task_id"] for t in pool[domain]]
        extras = {tid: {} for tid in ids}
        if ARMS[arm_name]["policy"] == "oracle":
            base = load_arm(model, domain, "A_no_reset", ids)
            if len(base) < len(ids):
                print(f"[skip] {arm_name}/{domain}: needs A_no_reset first "
                      f"({len(base)}/{len(ids)})", flush=True)
                continue
            extras = {tid: {"oracle_turn": base[tid]["first_hallucination"]}
                      for tid in ids}
        if ARMS[arm_name]["policy"] == "random":
            ref = load_arm(model, domain, "Z_reground", ids)
            if len(ref) < len(ids):
                print(f"[skip] {arm_name}/{domain}: needs Z_reground first for "
                      f"the reset budget ({len(ref)}/{len(ids)})", flush=True)
                continue
            extras = {tid: {"n_resets": ref[tid]["n_resets"], "seed": SEED,
                            "horizon": ref[tid]["horizon"]} for tid in ids}
        for task in pool[domain]:
            jobs.append((domain, task, extras[task["task_id"]]))

    done = {"n": 0}
    t0 = time.time()

    async def one(domain, task, extra):
        async with sem:
            try:
                r = await run_one(cfg, model, domain, task, arm_name, extra)
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
    import_A(args.model, pool)
    for arm_name in arms:
        print(f"=== arm {arm_name} (policy={ARMS[arm_name]['policy']}, "
              f"operator={ARMS[arm_name]['operator']}) ===", flush=True)
        asyncio.run(run_arm_all(args.model, arm_name, pool))
    print(f"=== done in {(time.time()-t0)/60:.1f} min ===", flush=True)


if __name__ == "__main__":
    main()
