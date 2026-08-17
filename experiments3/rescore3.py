"""Recompute canary fields for every stored experiment-3 trajectory from the
saved assistant text, using the CURRENT checkers in canaries3.py.

Canary scoring never influences the trajectory control flow (early stop is on
hallucination only), so post-hoc rescoring is exact and costs no API calls.
Run it after any checker change, then re-run metrics:

  python -m experiments3.rescore3
  python -m experiments3.run_all3 --skip-judge --limit 0  # or metrics3 directly
"""
import glob
import json

from .canaries3 import score_canary
from .config3 import RUNS_DIR, TASK_SETS, task_file


def rescore_all():
    for task_set in TASK_SETS:
        tf = task_file(task_set)
        if not tf.exists():
            continue
        tasks = {t["task_id"]: t for t in json.loads(tf.read_text())}
        changed = total = 0
        for f in glob.glob(str(RUNS_DIR / task_set / "*" / "*" / "task_*.json")):
            if "/judge/" in f:
                continue
            d = json.loads(open(f).read())
            if not d.get("complete") or d["condition"] == "baseline":
                continue
            task = tasks.get(d["task_id"])
            if task is None:
                continue
            turns = {t["turn"]: t for t in task["turns"]}
            total += 1
            first_fail = None
            dirty = False
            for rec in d["records"]:
                canary = score_canary(d["condition"], rec["assistant"],
                                      task, turns[rec["turn"]])
                new = {"canary_score": canary["score"]}
                for k in ("subs", "fabricated", "rehearsed"):
                    if k in canary:
                        new[f"canary_{k}"] = canary[k]
                for k, v in new.items():
                    if rec.get(k) != v:
                        rec[k] = v
                        dirty = True
                for k in ("canary_subs", "canary_fabricated", "canary_rehearsed"):
                    if k in rec and k not in new:
                        del rec[k]
                        dirty = True
                if canary["score"] is not None and canary["score"] < 1.0 \
                        and first_fail is None:
                    first_fail = rec["turn"]
            if d.get("first_canary_fail") != first_fail:
                d["first_canary_fail"] = first_fail
                dirty = True
            if dirty:
                open(f, "w").write(json.dumps(d))
                changed += 1
        print(f"[{task_set}] rescored {total} trajectories, {changed} changed")


if __name__ == "__main__":
    rescore_all()
