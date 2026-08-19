"""Experiment-5 figures.

fig1  accuracy vs resets/task: every arm on the cost/benefit plane, pooled
fig2  per-domain accuracy by arm (grouped bars): where routing pays
fig3  the routing decomposition: carrying cost vs timing value vs zero-carry
fig4  precision/recall plane per domain, signals scored on same trajectories
fig5  lead-time distributions where signals fire before the failure
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config5 import DOMAINS, RESULTS_DIR

ARM_STYLE = {
    "A_no_reset":     ("no reset", "#888888", "o"),
    "B_random":       ("random", "#b0b0b0", "v"),
    "C_clock":        ("clock (turns)", "#1f77b4", "s"),
    "C_ctx":          ("context growth", "#17becf", "P"),
    "C_judge":        ("LLM judge", "#9467bd", "X"),
    "C_prime_routed": ("clock + carried probe", "#aec7e8", "s"),
    "D_routed":       ("ROUTED sentinel", "#d62728", "*"),
    "D_labeled":      ("labeled routing", "#ff7f0e", "h"),
    "D_blanket":      ("blanket probe", "#ff9896", "D"),
    "D_rotated":      ("anti-routed probe", "#e377c2", "d"),
    "Z_routed":       ("ZERO-CARRY routed", "#2ca02c", "*"),
    "F_oracle":       ("oracle", "#000000", "^"),
}


def fig1(m):
    fig, ax = plt.subplots(figsize=(7.2, 5))
    for arm, s in m["arms"].items():
        label, color, marker = ARM_STYLE[arm]
        big = arm in ("D_routed", "Z_routed")
        ax.scatter(s["resets_per_task"], s["accuracy"], s=260 if big else 110,
                   c=color, marker=marker, zorder=3, edgecolors="black",
                   linewidths=0.6, label=label)
    ax.set_xlabel("resets per task")
    ax.set_ylabel("per-turn accuracy")
    ax.set_title(f"Accuracy vs intervention budget ({m['model']}, "
                 f"n={m['n_tasks']} mixed tasks)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right", ncols=2)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figures" / "fig1_accuracy_vs_budget.png", dpi=160)


def fig2(m):
    arms = [a for a in ARM_STYLE if a in m["arms_by_domain"]]
    x = range(len(DOMAINS))
    width = 0.8 / len(arms)
    fig, ax = plt.subplots(figsize=(9, 4.6))
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
    ax.set_title("Accuracy by domain and reset policy")
    ax.legend(fontsize=7, ncols=3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figures" / "fig2_domain_bars.png", dpi=160)


def fig3(m):
    keep = [
        ("C_prime_routed - C_clock", "carrying cost\n(routed probe)"),
        ("D_routed - C_prime_routed", "timing value\n(routed probe)"),
        ("D_routed - C_clock", "routed sentinel\nnet vs clock"),
        ("D_labeled - D_routed", "router noise cost\n(labeled - LLM router)"),
        ("Z_routed - C_clock", "zero-carry routed\nnet vs clock"),
        ("F_oracle - C_clock", "perfect timing\n(headroom)"),
    ]
    rows = []
    for c in m["contrasts"]:
        head = c["contrast"].split(":")[0]
        for key, label in keep:
            if head == key:
                rows.append((label, c["mean_delta"], c["ci95"]))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ys = range(len(rows))
    for y, (label, d, (lo, hi)) in zip(ys, rows):
        color = "#2ca02c" if lo > 0 else "#d62728" if hi < 0 else "#888888"
        ax.errorbar(d, y, xerr=[[d - lo], [hi - d]], fmt="o", color=color,
                    capsize=4, markersize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlabel("Δ per-turn accuracy vs comparison arm (95% CI)")
    ax.set_title("Decomposing the sentinel's ledger")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figures" / "fig3_decomposition.png", dpi=160)


# --------------------------------------------------- prediction-layer figures
# (set, signal display name) -> plot style; routed signals drawn as big stars
PRED_POINTS = [
    ("A_no_reset", "zero-carry monitor", "zero-carry monitor (A_no_reset)",
     "#2ca02c", "*", True),
    ("C_prime_routed", "routed probe", "routed probe (C′, clock-segmented)",
     "#d62728", "*", True),
    ("D_routed", "routed probe", "routed probe (D_routed, self-censored)",
     "#d62728", "o", False),
    ("D_labeled", "labeled probe", "labeled probe (D_labeled, self-censored)",
     "#ff7f0e", "h", False),
    ("C_judge", "LLM judge", "LLM judge (C_judge, self-censored)",
     "#9467bd", "X", False),
    ("A_no_reset", "turn_number", "turn number (best F1)", "#1f77b4", "s", False),
    ("A_no_reset", "context_length", "context length (best F1)",
     "#17becf", "P", False),
    ("A_no_reset", "random (expected)", "random (expected)", "#b0b0b0", "v", False),
]


def _pred_metric(p, arm, signal, domain):
    e = p["sets"].get(arm)
    if not e or signal not in e["signals"]:
        return None
    if domain is None:
        return e["signals"][signal]["pooled"]
    return e["signals"][signal]["by_domain"].get(domain)


def fig4(p):
    fig, axes = plt.subplots(1, len(DOMAINS), figsize=(12.5, 4.4),
                             sharex=True, sharey=True)
    for ax, domain in zip(axes, DOMAINS):
        for arm, sig, label, color, marker, routed in PRED_POINTS:
            m = _pred_metric(p, arm, sig, domain)
            if not m or m["precision"] is None or m["recall"] is None:
                continue
            ax.scatter(m["recall"], m["precision"], s=300 if routed else 110,
                       c=color, marker=marker, edgecolors="black",
                       linewidths=0.6, zorder=3,
                       label=label if domain == DOMAINS[0] else None)
        ax.set_title(domain)
        ax.set_xlabel("recall")
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(-0.04, 1.04)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("precision")
    fig.suptitle(f"Predicting the first failure within K={p['K']} turns — "
                 "each signal scored on its own arm's pre-reset segments, "
                 "baselines re-scored per set", fontsize=10)
    fig.legend(fontsize=8, loc="lower center", ncols=4,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(RESULTS_DIR / "figures" / "fig4_precision_recall.png", dpi=160,
                bbox_inches="tight")


def fig5(p):
    rows = []
    for arm, sig, label, color, marker, routed in PRED_POINTS:
        m = _pred_metric(p, arm, sig, None)
        if m and m.get("leads"):
            rows.append((label, color, m["leads"]))
    if not rows:
        print("fig5 skipped: no lead-time data")
        return
    fig, ax = plt.subplots(figsize=(8.2, 0.62 * len(rows) + 1.8))
    bp = ax.boxplot([r[2] for r in rows], vert=False, patch_artist=True,
                    widths=0.55, showfliers=True,
                    medianprops={"color": "black"})
    for patch, (_, color, _) in zip(bp["boxes"], rows):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for y, (_, _, leads) in enumerate(rows, start=1):
        ax.text(max(leads) + 0.35, y, f"n={len(leads)}", va="center",
                fontsize=8, color="#444444")
    ax.set_yticks(range(1, len(rows) + 1))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlabel("lead time (turns between first fire and first failure, "
                  "fires at/before the failure only)")
    ax.set_title("Where signals fire early: lead-time distributions, pooled",
                 fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figures" / "fig5_lead_time.png", dpi=160)


def main():
    (RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
    m = json.loads((RESULTS_DIR / "metrics.json").read_text())
    fig1(m)
    fig2(m)
    fig3(m)
    pred = RESULTS_DIR / "prediction.json"
    if pred.exists():
        p = json.loads(pred.read_text())
        fig4(p)
        fig5(p)
    else:
        print("fig4/fig5 skipped: run experiments5.prediction5 first")
    print("figures written to", RESULTS_DIR / "figures")


if __name__ == "__main__":
    main()
