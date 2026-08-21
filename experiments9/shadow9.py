"""Shadow quiz pass for experiment 9: ask the frozen-state quiz on
reconstructed prefixes of each model's OWN A_no_reset trajectories, so quiz
precision/recall is scored per model on full-horizon trajectories.

  python -m experiments9.shadow9 --model gpt-oss-120b
  python -m experiments9.shadow9                # every model
"""
import argparse
import asyncio
import json
import time

from .config9 import DOMAIN, MODEL_ORDER, MODELS, RUNS_DIR, load_env
from .domain9 import (SYSTEM_PROMPT, ask_quiz, build_first_user_message,
                      build_turn_body, make_store)
from .harness9 import quiz_due
from .run_all9 import load_pool, traj_path


def shadow_path(model, task_id):
    return RUNS_DIR / model / DOMAIN / "shadow_quiz" / f"task_{task_id:03d}.json"


def load_shadow(model, task_ids):
    out = {}
    for tid in task_ids:
        p = shadow_path(model, tid)
        if p.exists():
            try:
                d = json.loads(p.read_text())
            except json.JSONDecodeError:
                continue
            if d.get("complete"):
                out[tid] = d
    return out


async def shadow_one(cfg, model, task):
    path = shadow_path(model, task["task_id"])
    if path.exists():
        try:
            d = json.loads(path.read_text())
            if d.get("complete"):
                return d
        except json.JSONDecodeError:
            pass
    bp = traj_path(model, "A_no_reset", task["task_id"])
    if not bp.exists():
        print(f"[skip] shadow {model}/task{task['task_id']}: no A_no_reset",
              flush=True)
        return None
    base = json.loads(bp.read_text())
    if not base.get("complete"):
        return None
    replies = {r["turn"]: r["assistant"] for r in base["records"]}

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    store = make_store(task)
    checkpoints = []
    prompt_tok = completion_tok = 0
    for i, turn in enumerate(task["turns"]):
        t = turn["turn"]
        if t not in replies:
            break
        user_msg = (build_first_user_message(task, "baseline") if i == 0
                    else build_turn_body(task, turn, "baseline"))
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": replies[t]})
        store.apply(turn)
        if quiz_due(t, task["horizon"]):
            qrec, usage = await ask_quiz(cfg, messages, task, store, t)
            prompt_tok += usage["prompt_tokens"] or 0
            completion_tok += usage["completion_tokens"] or 0
            checkpoints.append(qrec)

    out = {"task_id": task["task_id"], "domain": DOMAIN,
           "horizon": task["horizon"],
           "first_hallucination": base["first_hallucination"],
           "checkpoints": checkpoints,
           "quiz_prompt_tokens": prompt_tok,
           "quiz_completion_tokens": completion_tok,
           "complete": True}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out


async def run_all(model, pool, progress_every=10):
    cfg = MODELS[model]
    sem = asyncio.Semaphore(cfg["concurrency"])
    done = {"n": 0}
    t0 = time.time()

    async def one(task):
        async with sem:
            try:
                r = await shadow_one(cfg, model, task)
            except Exception as e:
                print(f"[FAIL] shadow/{model}/task{task['task_id']}: "
                      f"{type(e).__name__}: {e}", flush=True)
                r = None
            done["n"] += 1
            if done["n"] % progress_every == 0 or done["n"] == len(pool):
                print(f"  shadow/{model}: {done['n']}/{len(pool)} "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)
            return r
    return await asyncio.gather(*[one(t) for t in pool])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    load_env()
    pool = load_pool()
    if args.limit:
        pool = pool[:args.limit]
    for model in ([args.model] if args.model else MODEL_ORDER):
        asyncio.run(run_all(model, pool))


if __name__ == "__main__":
    main()
