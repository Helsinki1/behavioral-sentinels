"""Orchestrates the full experiment: tasks -> trajectories -> judge -> metrics.

Resumable: completed trajectories/judge files are skipped on re-run.
Usage: python3 -m experiments.run_all [--limit N] [--models a,b] [--conditions x,y]
"""
import argparse
import asyncio
import json
import time

from . import tasks as tasks_mod
from .config import ALL_CONDITIONS, DATA_DIR, MODELS, load_env
from .judge import judge_all
from .metrics import compute_all
from .runner import run_all as run_trajectories


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only first N tasks (smoke test)")
    ap.add_argument("--models", type=str, default=None)
    ap.add_argument("--conditions", type=str, default=None)
    ap.add_argument("--skip-metrics", action="store_true")
    args = ap.parse_args()

    load_env()
    task_file = DATA_DIR / "tasks.json"
    if not task_file.exists():
        tasks_mod.main()
    tasks = json.loads(task_file.read_text())
    if args.limit:
        tasks = tasks[:args.limit]

    model_names = args.models.split(",") if args.models else list(MODELS)
    conditions = args.conditions.split(",") if args.conditions else ALL_CONDITIONS

    t0 = time.time()
    print(f"=== phase 1: trajectories ({len(model_names)} models x {len(conditions)} "
          f"conditions x {len(tasks)} tasks) ===", flush=True)
    asyncio.run(run_trajectories(tasks, model_names, conditions))

    if "baseline" in conditions:
        print("=== phase 2: LLM judge on baseline trajectories ===", flush=True)
        asyncio.run(judge_all(model_names, [t["task_id"] for t in tasks]))

    if not args.skip_metrics:
        print("=== phase 3: metrics ===", flush=True)
        compute_all([t["task_id"] for t in tasks], model_names)

    # rough usage/cost report from stored trajectories
    from .runner import traj_path
    tot = {m: [0, 0] for m in model_names}
    for m in model_names:
        for c in conditions:
            for t in tasks:
                p = traj_path(m, c, t["task_id"])
                if p.exists():
                    d = json.loads(p.read_text())
                    for r in d["records"]:
                        tot[m][0] += r.get("prompt_tokens") or 0
                        tot[m][1] += r.get("completion_tokens") or 0
    for m, (pi, co) in tot.items():
        print(f"usage {m}: {pi/1e6:.1f}M prompt tokens, {co/1e6:.2f}M completion tokens", flush=True)
    print(f"=== done in {(time.time()-t0)/60:.1f} min ===", flush=True)


if __name__ == "__main__":
    main()
