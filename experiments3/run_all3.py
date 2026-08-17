"""Orchestrates experiment 3: tasks -> trajectories -> judge -> metrics,
independently per task set.  Resumable throughout.

Usage:
  python -m experiments3.run_all3 [--sets coding,registers,babi]
                                  [--models gpt-oss-20b,...]
                                  [--conditions x,y] [--limit N]
                                  [--skip-judge] [--skip-metrics]
                                  [--pilot]

--pilot runs the first 20 tasks under the canary conditions only and prints a
fire-rate report per (set, model, condition): the target band is roughly
30-70 percent.  Tune the knobs in config3.py (LAGS, N_COUNTERS, EVENT_P,
CHECKSUM_MOD, SHADOW_RENAME_P, TAG_*, STAIR_PERIOD) per model before the full
run -- experiment 2 showed canary difficulty does not transfer across models.
"""
import argparse
import asyncio
import json
import time

from . import tasks3
from .config3 import (BASELINE_CONDITION, DEFAULT_MODELS, MODELS, TASK_SETS,
                      conditions_for, load_env, task_file)
from .judge3 import judge_all
from .metrics3 import compute_all
from .runner3 import run_all as run_trajectories, traj_path

PILOT_N = 20


def fire_report(task_set, model_names, conditions, tasks):
    print(f"\n=== pilot fire-rate report: {task_set} "
          f"(n={len(tasks)}, target band 0.30-0.70) ===")
    for m in model_names:
        for c in conditions:
            if c == BASELINE_CONDITION:
                continue
            fired = total = hallu = 0
            for t in tasks:
                p = traj_path(task_set, m, c, t["task_id"])
                if not p.exists():
                    continue
                d = json.loads(p.read_text())
                if not d.get("complete"):
                    continue
                total += 1
                fired += d["first_canary_fail"] is not None
                hallu += d["first_hallucination"] is not None
            if total:
                print(f"  {m:14s} {c:20s} fire {fired/total:.2f}  "
                      f"hallucinate {hallu/total:.2f}  (n={total})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", type=str, default=None)
    ap.add_argument("--models", type=str, default=None)
    ap.add_argument("--conditions", type=str, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-judge", action="store_true")
    ap.add_argument("--skip-metrics", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()

    load_env()
    sets = args.sets.split(",") if args.sets else TASK_SETS
    model_names = args.models.split(",") if args.models else list(DEFAULT_MODELS)

    t0 = time.time()
    for ts in sets:
        if not task_file(ts).exists():
            tasks3.main([ts])
        tasks = json.loads(task_file(ts).read_text())

        conditions = (args.conditions.split(",") if args.conditions
                      else conditions_for(ts))
        limit = args.limit
        if args.pilot:
            limit = min(PILOT_N, limit or PILOT_N)
            conditions = [c for c in conditions if c != BASELINE_CONDITION]
        if limit:
            tasks = tasks[:limit]

        print(f"=== [{ts}] trajectories: {len(model_names)} models x "
              f"{len(conditions)} conditions x {len(tasks)} tasks ===", flush=True)
        asyncio.run(run_trajectories(ts, tasks, model_names, conditions))

        if args.pilot:
            fire_report(ts, model_names, conditions, tasks)
            continue

        if BASELINE_CONDITION in conditions and not args.skip_judge:
            print(f"=== [{ts}] LLM judge on baseline trajectories ===", flush=True)
            asyncio.run(judge_all(ts, model_names, [t["task_id"] for t in tasks]))

    if not args.pilot and not args.skip_metrics:
        print("=== metrics ===", flush=True)
        compute_all(sets, model_names)

    # usage roll-up
    for ts in sets:
        if not task_file(ts).exists():
            continue
        tasks = json.loads(task_file(ts).read_text())
        for m in model_names:
            pi = co = 0
            for c in conditions_for(ts):
                for t in tasks:
                    p = traj_path(ts, m, c, t["task_id"])
                    if p.exists():
                        try:
                            for r in json.loads(p.read_text())["records"]:
                                pi += r.get("prompt_tokens") or 0
                                co += r.get("completion_tokens") or 0
                        except (json.JSONDecodeError, KeyError):
                            pass
            if pi or co:
                print(f"usage {ts}/{m}: {pi/1e6:.1f}M prompt, {co/1e6:.2f}M completion")
    print(f"=== done in {(time.time()-t0)/60:.1f} min ===", flush=True)


if __name__ == "__main__":
    main()
