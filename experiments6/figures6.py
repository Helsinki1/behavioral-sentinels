"""Experiment-6 figures.

fig1  accuracy vs resets/task: every arm on the cost/benefit plane, pooled
fig2  per-domain accuracy by arm (grouped bars)
fig3  the operator effect: same trigger, re-grounding minus exp-5 compaction
fig4  accuracy vs prompt tokens: the deployment Pareto plane
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config6 import DOMAINS, RESULTS_DIR

ARM_STYLE = {
    "A_no_reset":      ("no reset", "#888888", "o"),
    "B_random":        ("random (reground)", "#b0b0b0", "v"),
    "C_clock":         ("clock (reground)", "#1f77b4", "s"),
    "C_ctx":           ("context growth (reground)", "#17becf", "P"),
    "C_judge":         ("LLM judge (reground)", "#9467bd", "X"),
    "Z_reground":      ("ZERO-CARRY reground", "#2ca02c", "*"),
    "F_oracle":        ("oracle (reground)", "#000000", "^"),
    "G_dense":         ("dense schedule (reground)", "#d62728", "D"),
    "Z_replay":        ("zero-carry REPLAY", "#98df8a", "*"),
    "C_clock_replay":  ("clock replay", "#aec7e8", "s"),
    "F_oracle_replay": ("oracle replay", "#666666", "^"),
}


def fig1(m):
    fig, ax = plt.subplots(figsize=(8.8, 5))
    for arm, s in m["arms"].items():
        label, color, marker = ARM_STYLE[arm]
        big = arm in ("Z_reground", "G_dense")
        ax.scatter(s["resets_per_task"], s["accuracy"], s=260 if big else 110,
                   c=color, marker=marker, zorder=3, edgecolors="black",
                   linewidths=0.6, label=label)
    ax.set_xlabel("resets per task")
    ax.set_ylabel("per-turn accuracy")
    ax.set_title(f"Accuracy vs intervention budget, re-grounding operator "
                 f"({m['model']}, n={m['n_tasks']})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figures" / "fig1_accuracy_vs_budget.png", dpi=160)


def fig2(m):
    arms = [a for a in ARM_STYLE if a in m["arms_by_domain"]]
    x = range(len(DOMAINS))
    width = 0.8 / len(arms)
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for i, arm in enumerate(arms):
        label, color, _ = ARM_STYLE[arm]
        vals = [m["arms_by_domain"][arm].get(d, {}).get("accuracy") for d in DOMAINS]
        ax.bar([xi + i * width for xi in x], [v or 0 for v in vals],
               width=width, color=color, edgecolor="black", linewidth=0.4,
               label=label)
    ax.set_xticks([xi + 0.4 for xi in x])
    ax.set_xticklabels(DOMAINS)
    ax.set_ylabel("per-turn accuracy")
    ax.set_ylim(bottom=0.4)
    ax.set_title("Accuracy by domain and reset policy (re-grounding)")
    ax.legend(fontsize=7, ncols=3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figures" / "fig2_domain_bars.png", dpi=160)


def fig3(m):
    rows = [(c["contrast"].split(":")[0], c["mean_delta"], c["ci95"])
            for c in m["cross_experiment"] if "domain" not in c]
    if not rows:
        print("fig3 skipped: no cross-experiment contrasts")
        return
    fig, ax = plt.subplots(figsize=(7.6, 0.7 * len(rows) + 1.6))
    ys = range(len(rows))
    for y, (label, d, (lo, hi)) in zip(ys, rows):
        color = "#2ca02c" if lo > 0 else "#d62728" if hi < 0 else "#888888"
        ax.errorbar(d, y, xerr=[[d - lo], [hi - d]], fmt="o", color=color,
                    capsize=4, markersize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlabel("Δ accuracy: re-grounding − compaction, same trigger (95% CI)")
    ax.set_title("The operator effect, paired on task across experiments 5→6")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figures" / "fig3_operator_effect.png", dpi=160)


def fig4(m):
    fig, ax = plt.subplots(figsize=(8.8, 5))
    for arm, s in m["arms"].items():
        label, color, marker = ARM_STYLE[arm]
        big = arm in ("Z_reground", "G_dense")
        ax.scatter(s["prompt_tokens"] / 1000, s["accuracy"],
                   s=260 if big else 110, c=color, marker=marker, zorder=3,
                   edgecolors="black", linewidths=0.6, label=label)
    ax.set_xlabel("prompt tokens per task (thousands)")
    ax.set_ylabel("per-turn accuracy")
    ax.set_title("The deployment Pareto plane: accuracy vs token cost")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figures" / "fig4_pareto.png", dpi=160)


def main():
    m = json.loads((RESULTS_DIR / "metrics.json").read_text())
    (RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
    fig1(m)
    fig2(m)
    fig3(m)
    fig4(m)
    print("figures written to", RESULTS_DIR / "figures")


if __name__ == "__main__":
    main()
