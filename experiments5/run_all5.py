"""Orchestrator for experiment 5. Resumable; arms run in dependency order.

  python -m experiments5.run_all5 --gate      # A, C_clock, D_routed (go/no-go)
  python -m experiments5.run_all5             # all eleven arms
  python -m experiments5.run_all5 --arms X,Y  # specific arms
"""
import argparse
import asyncio
import json
import time

from .config5 import (ARM_ORDER, ARMS, DATA_DIR, DEFAULT_MODEL, DOMAINS,
                      GATE_ARMS, JUDGE_MODEL, MODELS, N_PER_DOMAIN, RUNS_DIR,
                      SEED, load_env)
from .harness5 import run_arm
from .policies5 import POLICIES
from .routing5 import probe_for, route_task


def select_tasks(domain, n=N_PER_DOMAIN):
    """Difficulty-stratified deterministic sample of n tasks."""
    tasks = json.loads((DATA_DIR / f"tasks3_{domain}.json").read_text())
    ranked = sorted(tasks, key=lambda t: (float(t["difficulty"]), t["task_id"]))
    if n >= len(ranked):
        return ranked
    picked, seen = [], set()
    for i in range(n):
        j = round(i * (len(ranked) - 1) / (n - 1))
        while j in seen:
            j += 1
        seen.add(j)
        picked.append(ranked[j])
    return sorted(picked, key=lambda t: t["task_id"])


def traj_path(model, domain, arm, task_id):
    return RUNS_DIR / model / domain / arm / f"task_{task_id:03d}.json"


def router_path(model, domain, task_id):
    return RUNS_DIR / model / domain / "router" / f"task_{task_id:03d}.json"


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


def load_router(model, domain, task_ids):
    out = {}
    for tid in task_ids:
        p = router_path(model, domain, tid)
        if p.exists():
            out[tid] = json.loads(p.read_text())
    return out


async def route_all(model, pool):
    """Ensure every task has a cached router verdict; one call per task."""
    cfg = MODELS[model]
    sem = asyncio.Semaphore(cfg["concurrency"])

    async def one(domain, task):
        p = router_path(model, domain, task["task_id"])
        if p.exists():
            return
        async with sem:
            verdict = await route_task(cfg, domain, task)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(
            {"task_id": task["task_id"], "domain": domain, **verdict}))

    await asyncio.gather(*[one(d, t) for d in DOMAINS for t in pool[d]])
    counts = {}
    for d in DOMAINS:
        router = load_router(model, d, [t["task_id"] for t in pool[d]])
        probes = sorted(v["probe"] for v in router.values())
        counts[d] = {p: probes.count(p) for p in dict.fromkeys(probes)}
    print(f"router verdicts: {counts}", flush=True)


async def run_one(cfg, model, domain, task, arm_name, extras, routed_probe):
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
    condition = probe_for(arm["probe"], domain, routed_probe)
    decide = POLICIES[arm["policy"]](task, **extras)
    judge_cfg = MODELS[JUDGE_MODEL] if arm["policy"] == "judge" else None
    out = await run_arm(cfg, domain, task, arm_name, condition, decide,
                        judge_cfg=judge_cfg)
    path.write_text(json.dumps(out))
    return out


async def run_arm_all(model, arm_name, pool, progress_every=15):
    cfg = MODELS[model]
    sem = asyncio.Semaphore(cfg["concurrency"])
    jobs = []
    for domain in DOMAINS:
        ids = [t["task_id"] for t in pool[domain]]
        router = load_router(model, domain, ids)
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
            ref = load_arm(model, domain, "D_routed", ids)
            if len(ref) < len(ids):
                print(f"[skip] {arm_name}/{domain}: needs D_routed first for "
                      f"the reset budget ({len(ref)}/{len(ids)})", flush=True)
                continue
            extras = {tid: {"n_resets": ref[tid]["n_resets"], "seed": SEED,
                            "horizon": ref[tid]["horizon"]} for tid in ids}
        for task in pool[domain]:
            routed = router.get(task["task_id"], {}).get("probe")
            jobs.append((domain, task, extras[task["task_id"]], routed))

    done = {"n": 0}
    t0 = time.time()

    async def one(domain, task, extra, routed):
        async with sem:
            try:
                r = await run_one(cfg, model, domain, task, arm_name, extra, routed)
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
    asyncio.run(route_all(args.model, pool))
    for arm_name in arms:
        print(f"=== arm {arm_name} (policy={ARMS[arm_name]['policy']}, "
              f"probe={ARMS[arm_name]['probe']}) ===", flush=True)
        asyncio.run(run_arm_all(args.model, arm_name, pool))
    print(f"=== done in {(time.time()-t0)/60:.1f} min ===", flush=True)


if __name__ == "__main__":
    main()
