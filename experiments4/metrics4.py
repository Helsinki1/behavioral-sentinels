"""Experiment-4 analysis: paired comparison of reset policies on task accuracy.

Primary outcome  : mean per-turn accuracy (share of turns with zero errors)
Primary contrast : D_sentinel vs C_prime_carried   <- both carry the probe, so
                   this isolates the VALUE OF THE TIMING INFORMATION
Cost contrast    : C_prime_carried vs C_scheduled  <- both reset on the same
                   schedule, so this isolates the OBSERVER-EFFECT COST
All tests are paired over task ids, with a bootstrap CI on the paired delta.
"""
import json
import random
import statistics

from .config4 import ARMS, RESULTS_DIR, SUCCESS_THRESHOLD
from .run_all4 import load_arm


def paired(model, arms, task_ids):
    """Return {arm: [record...]} restricted to tasks present in EVERY arm."""
    loaded = {a: load_arm(model, a, task_ids) for a in arms}
    common = sorted(set.intersection(*[set(v) for v in loaded.values()])) if loaded else []
    return {a: [loaded[a][t] for t in common] for a in loaded}, common


def boot_ci(deltas, n=10000, seed=7):
    if not deltas:
        return (None, None)
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        s = [deltas[rng.randrange(len(deltas))] for _ in range(len(deltas))]
        means.append(sum(s) / len(s))
    means.sort()
    return (means[int(0.025 * n)], means[int(0.975 * n)])


def arm_stats(recs):
    acc = [r["accuracy"] for r in recs]
    return {
        "n": len(recs),
        "accuracy": round(statistics.mean(acc), 4),
        "success_rate": round(sum(1 for a in acc if a >= SUCCESS_THRESHOLD) / len(acc), 4),
        "resets_per_task": round(statistics.mean(r["n_resets"] for r in recs), 3),
        "prompt_tokens": round(statistics.mean(r["prompt_tokens"] for r in recs)),
        "completion_tokens": round(statistics.mean(r["completion_tokens"] for r in recs)),
        "clean_turn_rate": round(sum(r["clean_turns"] for r in recs) /
                                 sum(r["turns_run"] for r in recs), 4),
    }


def contrast(a_recs, b_recs, label):
    d = [x["accuracy"] - y["accuracy"] for x, y in zip(a_recs, b_recs)]
    lo, hi = boot_ci(d)
    wins = sum(1 for v in d if v > 1e-9)
    losses = sum(1 for v in d if v < -1e-9)
    return {
        "contrast": label, "n": len(d),
        "mean_delta": round(statistics.mean(d), 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "significant": bool(lo > 0 or hi < 0),
        "tasks_better": wins, "tasks_worse": losses, "tasks_tied": len(d) - wins - losses,
    }


def compute(model="gpt-oss-20b", task_ids=None):
    task_ids = task_ids if task_ids is not None else list(range(200))
    present = [a for a in ARMS if load_arm(model, a, task_ids)]
    recs, common = paired(model, present, task_ids)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    stats = {a: arm_stats(v) for a, v in recs.items()}
    contrasts = []
    pairs = [("D_sentinel", "C_prime_carried", "timing value (both carry probe)"),
             ("C_prime_carried", "C_scheduled", "observer-effect cost"),
             ("D_sentinel", "C_scheduled", "sentinel vs plain clock"),
             ("D_sentinel", "B_random", "sentinel vs random, budget-matched"),
             ("C_scheduled", "A_no_reset", "does resetting help at all"),
             ("F_oracle", "C_scheduled", "PERFECT predictor vs clock (headroom)"),
             ("F_oracle", "A_no_reset", "perfect predictor vs no reset")]
    for a, b, lab in pairs:
        if a in recs and b in recs:
            contrasts.append(contrast(recs[a], recs[b], f"{a} - {b}: {lab}"))

    out = {"model": model, "n_tasks": len(common), "arms": stats, "contrasts": contrasts}
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(out, indent=2))

    L = ["# Experiment 4 — Sentinel-triggered resets vs a clock", "",
         f"Model {model} - {len(common)} tasks, paired across every arm - full horizon, no early stop.",
         "Primary outcome: share of turns with zero errors. Compaction = the agent's own",
         "state self-summary (no ground truth injected), identical in every resetting arm.", "",
         "| arm | policy | carries probe | accuracy | success@0.9 | resets/task | prompt tok |",
         "|---|---|---|---|---|---|---|"]
    order = ["A_no_reset", "B_random", "C_scheduled", "C_prime_carried", "D_sentinel", "F_oracle"]
    for a in [x for x in order if x in stats]:
        s = stats[a]
        L.append(f"| {a} | {ARMS[a]['policy']} | {'yes' if ARMS[a]['carries_sentinel'] else 'no'} "
                 f"| {s['accuracy']:.3f} | {s['success_rate']:.3f} | {s['resets_per_task']:.2f} "
                 f"| {s['prompt_tokens']:,} |")
    L += ["", "## Paired contrasts (bootstrap 95% CI on the per-task delta)", "",
          "| contrast | delta accuracy | 95% CI | significant | better/worse/tied |", "|---|---|---|---|---|"]
    for c in contrasts:
        L.append(f"| {c['contrast']} | {c['mean_delta']:+.4f} | "
                 f"[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] | "
                 f"{'**yes**' if c['significant'] else 'no'} | "
                 f"{c['tasks_better']}/{c['tasks_worse']}/{c['tasks_tied']} |")
    (RESULTS_DIR / "SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L).encode("ascii", "replace").decode("ascii"))
    return out


if __name__ == "__main__":
    compute(task_ids=list(range(40)))
