"""The shadow quiz pass: the quiz's signal quality WITHOUT new agent runs.

runs5/A_no_reset stores every assistant reply, and the baseline user messages
are pure functions of the task data, so the exact frozen conversation prefix
at any turn is reconstructible. At every checkpoint turn this pass rebuilds
that prefix, asks the frozen-state quiz on it, grades against the store, and
records the result -- yielding quiz precision/recall/lead-time on the SAME
full-horizon trajectories where every passive signal is already scored
(prediction5's same-trajectory protocol), with no censoring by resets.

  python -m experiments8.shadow8            # full pool
  python -m experiments8.shadow8 --limit 5  # first slice
"""
import argparse
import asyncio
import json
import time

from experiments3.canaries3 import (SYSTEM_PROMPTS, build_first_user_message,
                                    build_turn_body)
from experiments6.store6 import make_store
from experiments8.quiz8 import ask_quiz

from .config8 import (DEFAULT_MODEL, DOMAINS, MODELS, N_PER_DOMAIN, RUNS5_DIR,
                      RUNS_DIR, load_env)
from .harness8 import quiz_due
from experiments5.run_all5 import select_tasks


def shadow_path(model, domain, task_id):
    return RUNS_DIR / model / domain / "shadow_quiz" / f"task_{task_id:03d}.json"


def load_shadow(model, domain, task_ids):
    out = {}
    for tid in task_ids:
        p = shadow_path(model, domain, tid)
        if p.exists():
            try:
                d = json.loads(p.read_text())
            except json.JSONDecodeError:
                continue
            if d.get("complete"):
                out[tid] = d
    return out


def load_base(model, domain, task_id):
    p = (RUNS5_DIR / model / domain / "A_no_reset"
         / f"task_{task_id:03d}.json")
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d if d.get("complete") else None


async def shadow_one(cfg, model, domain, task):
    path = shadow_path(model, domain, task["task_id"])
    if path.exists():
        try:
            d = json.loads(path.read_text())
            if d.get("complete"):
                return d
        except json.JSONDecodeError:
            pass
    base = load_base(model, domain, task["task_id"])
    if base is None:
        print(f"[skip] shadow {domain}/task{task['task_id']}: no A_no_reset "
              "in runs5", flush=True)
        return None
    replies = {r["turn"]: r["assistant"] for r in base["records"]}

    messages = [{"role": "system", "content": SYSTEM_PROMPTS[domain]}]
    store = make_store(domain, task)
    checkpoints = []
    prompt_tok = completion_tok = 0
    for i, turn in enumerate(task["turns"]):
        t = turn["turn"]
        if t not in replies:
            break                     # trajectory shorter than the task
        user_msg = (build_first_user_message(domain, task, "baseline") if i == 0
                    else build_turn_body(domain, task, turn, "baseline"))
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": replies[t]})
        store.apply(turn)
        if quiz_due(t, task["horizon"]):
            qrec, usage = await ask_quiz(cfg, messages, domain, task, store, t)
            prompt_tok += usage["prompt_tokens"] or 0
            completion_tok += usage["completion_tokens"] or 0
            checkpoints.append(qrec)

    out = {
        "task_id": task["task_id"], "domain": domain,
        "horizon": task["horizon"],
        "first_hallucination": base["first_hallucination"],
        "checkpoints": checkpoints,
        "quiz_prompt_tokens": prompt_tok,
        "quiz_completion_tokens": completion_tok,
        "complete": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out


async def run_all(model, pool, progress_every=15):
    cfg = MODELS[model]
    sem = asyncio.Semaphore(cfg["concurrency"])
    jobs = [(d, t) for d in DOMAINS for t in pool[d]]
    done = {"n": 0}
    t0 = time.time()

    async def one(domain, task):
        async with sem:
            try:
                r = await shadow_one(cfg, model, domain, task)
            except Exception as e:
                print(f"[FAIL] shadow/{domain}/task{task['task_id']}: "
                      f"{type(e).__name__}: {e}", flush=True)
                r = None
            done["n"] += 1
            if done["n"] % progress_every == 0 or done["n"] == len(jobs):
                print(f"  shadow: {done['n']}/{len(jobs)} "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)
            return r
    return await asyncio.gather(*[one(*j) for j in jobs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=N_PER_DOMAIN)
    args = ap.parse_args()
    load_env()
    pool = {d: select_tasks(d, args.limit) for d in DOMAINS}
    asyncio.run(run_all(args.model, pool))


if __name__ == "__main__":
    main()
