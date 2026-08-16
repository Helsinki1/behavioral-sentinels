"""Orchestrates experiment 2: coding tasks -> trajectories -> judge -> metrics.

Resumable: completed trajectories and judge files are skipped on re-run.
Usage: python -m experiments2.run_all2 [--limit N] [--models a,b] [--conditions x,y]
"""
import argparse
import asyncio
import json
import time

from . import tasks2 as tasks_mod
from .config2 import ALL_CONDITIONS, DATA_DIR, MODELS, load_env
from .judge2 import judge_all
from .metrics2 import compute_all
from .runner2 import run_all as run_trajectories, traj_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only first N tasks")
    ap.add_argument("--models", type=str, default=None)
    ap.add_argument("--conditions", type=str, default=None)
    ap.add_argument("--skip-metrics", action="store_true")
    ap.add_argument("--skip-judge", action="store_true")
    args = ap.parse_args()

    load_env()
    task_file = DATA_DIR / "tasks2.json"
    if not task_file.exists():
        tasks_mod.main()
    tasks = json.loads(task_file.read_text())
    if args.limit:
        tasks = tasks[:args.limit]

    model_names = args.models.split(",") if args.models else list(MODELS)
    conditions = args.conditions.split(",") if args.conditions else ALL_CONDITIONS

    t0 = time.time()
    print(f"=== phase 1: trajectories ({len(model_names)} models x {len(conditions)} "
          f"conditions x {len(tasks)} coding tasks) ===", flush=True)
    asyncio.run(run_trajectories(tasks, model_names, conditions))

    if "baseline" in conditions and not args.skip_judge:
        print("=== phase 2: LLM judge on baseline trajectories ===", flush=True)
        asyncio.run(judge_all(model_names, [t["task_id"] for t in tasks]))

    if not args.skip_metrics:
        print("=== phase 3: metrics ===", flush=True)
        compute_all([t["task_id"] for t in tasks], model_names)

    tot = {m: [0, 0] for m in model_names}
    for m in model_names:
        for c in conditions:
            for t in tasks:
                p = traj_path(m, c, t["task_id"])
                if p.exists():
                    for r in json.loads(p.read_text())["records"]:
                        tot[m][0] += r.get("prompt_tokens") or 0
                        tot[m][1] += r.get("completion_tokens") or 0
    for m, (pi, co) in tot.items():
        print(f"usage {m}: {pi/1e6:.1f}M prompt tokens, {co/1e6:.2f}M completion tokens",
              flush=True)
    print(f"=== done in {(time.time()-t0)/60:.1f} min ===", flush=True)


if __name__ == "__main__":
    main()
