"""Experiment-8 figures -- one per claim of the active-vs-passive writeup.

fig2  signal quality: precision & recall per method, grouped by observation
      category (clean reads only; censored reads excluded)
fig3  downstream gain: accuracy per signal-routed arm on the SAME x-axis,
      bounds and baselines as reference lines
fig4  cost of observation: observer-effect delta + monitoring tokens
fig5  the decision plane: accuracy vs total token spend, Pareto front
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config8 import RESULTS_DIR

CAT_COLORS = {"active": "#d62728", "passive-behavioural": "#2ca02c",
              "passive-observational": "#1f77b4", "baseline": "#b0b0b0",
              "bound": "#555555"}

# (arm, display) in the shared x-axis order: active -> passive-behavioural ->
# passive-observational
METHOD_AXIS = [("ACT_probe", "carried probe\n(active)"),
               ("QUIZ", "frozen-state quiz\n(passive-behav.)"),
               ("Z_reground", "trace monitor\n(passive-obs.)"),
               ("C_judge", "LLM judge\n(passive-obs.)")]

# clean prediction reads per method: (set, signal name in prediction.json)
SIGNAL_SOURCE = {
    "ACT_probe": ("ACT_carry_clock", "carried probe (clock-truncated read)"),
    "QUIZ": ("A_no_reset", "frozen-state quiz (shadow)"),
    "Z_reground": ("A_no_reset", "zero-carry trace monitor"),
    "C_judge": ("C_judge", "LLM judge"),
}
BASELINE_SIGNALS = [("turn_number", "turn-count clock"),
                    ("context_length", "context growth"),
                    ("random (expected)", "random")]


def fig2(pred):
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    xs = range(len(METHOD_AXIS))
    prec, rec, colors = [], [], []
    for arm, _ in METHOD_AXIS:
        s, name = SIGNAL_SOURCE[arm]
        m = pred["sets"][s]["signals"][name]["pooled"]
        prec.append(m["precision"] or 0)
        rec.append(m["recall"] or 0)
        colors.append(CAT_COLORS["active" if arm.startswith("ACT") else
                                 "passive-behavioural" if arm == "QUIZ" else
                                 "passive-observational"])
    w = 0.38
    ax.bar([x - w / 2 for x in xs], prec, w, color=colors, edgecolor="black",
           linewidth=0.6, label="precision")
    ax.bar([x + w / 2 for x in xs], rec, w, color=colors, edgecolor="black",
           linewidth=0.6, alpha=0.45, label="recall")
    base = pred["sets"]["A_no_reset"]["signals"]
    for key, lab in BASELINE_SIGNALS:
        m = base.get(key, {}).get("pooled")
        if m and m["precision"] is not None:
            ax.axhline(m["precision"], color="#888888", linestyle="--",
                       linewidth=0.9)
            ax.annotate(f"{lab} precision", (len(METHOD_AXIS) - 0.5,
                        m["precision"]), fontsize=7, color="#555555",
                        va="bottom", ha="right")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([d for _, d in METHOD_AXIS], fontsize=9)
    ax.set_ylabel(f"predicting first hallucination (K={pred['K']})")
    ax.set_title("Fig 2 — signal quality by observation method "
                 "(solid = precision, faded = recall)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figures" / "fig2_signal_quality.png", dpi=160)


def fig3(m):
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    arms = [(a, d) for a, d in METHOD_AXIS if a in m["arms"]]
    xs = range(len(arms))
    for x, (arm, _) in zip(xs, arms):
        ax.bar(x, m["arms"][arm]["accuracy"],
               color=CAT_COLORS[m["categories"][arm]], edgecolor="black",
               linewidth=0.6, width=0.55)
    for ref, style in [("A_no_reset", ":"), ("C_clock", "--"),
                       ("G_dense", "-."), ("F_oracle", "-")]:
        if ref in m["arms"]:
            ax.axhline(m["arms"][ref]["accuracy"], color="#444444",
                       linestyle=style, linewidth=1.0)
            ax.annotate(ref, (len(arms) - 0.45, m["arms"][ref]["accuracy"]),
                        fontsize=7, va="bottom", ha="right", color="#333333")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([d for _, d in arms], fontsize=9)
    ax.set_ylabel("per-turn accuracy (signal-routed reground resets)")
    lo = min(m["arms"][a]["accuracy"] for a, _ in arms) - 0.06
    ax.set_ylim(bottom=max(0, lo))
    ax.set_title("Fig 3 — does the signal convert to downstream gain?")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figures" / "fig3_downstream_gain.png", dpi=160)


def fig4(m):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.2))
    # observer-effect delta: the paired ACT_carry_clock - C_clock contrast
    oc = next((c for c in m["contrasts"]
               if c["contrast"].startswith("ACT_carry_clock - C_clock")), None)
    labels = ["carried probe\n(active)", "frozen-state quiz", "trace monitor",
              "LLM judge"]
    deltas = [oc["mean_delta"] if oc else 0, 0, 0, 0]
    colors = [CAT_COLORS["active"], CAT_COLORS["passive-behavioural"],
              CAT_COLORS["passive-observational"],
              CAT_COLORS["passive-observational"]]
    ax1.bar(range(4), deltas, color=colors, edgecolor="black", linewidth=0.6)
    if oc:
        lo, hi = oc["ci95"]
        ax1.errorbar(0, oc["mean_delta"],
                     yerr=[[oc["mean_delta"] - lo], [hi - oc["mean_delta"]]],
                     fmt="none", ecolor="black", capsize=4)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("Δ accuracy caused by being observed")
    ax1.set_title("observer effect (matched schedule)\npassive methods: zero "
                  "by construction", fontsize=9)

    toks = [0,
            m["arms"].get("QUIZ", {}).get("quiz_tokens", 0) / 1000,
            0, 0]
    # active carrying cost in tokens: extra prompt tokens vs the clock arm
    if "ACT_carry_clock" in m["arms"] and "C_clock" in m["arms"]:
        toks[0] = max(0, (m["arms"]["ACT_carry_clock"]["prompt_tokens"]
                          - m["arms"]["C_clock"]["prompt_tokens"]) / 1000)
    if "C_judge" in m["arms"] and "C_clock" in m["arms"]:
        toks[3] = max(0, (m["arms"]["C_judge"]["prompt_tokens"]
                          - m["arms"]["C_clock"]["prompt_tokens"]) / 1000)
    ax2.bar(range(4), toks, color=colors, edgecolor="black", linewidth=0.6)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("monitoring tokens per task (thousands)")
    ax2.set_title("what observing cost\n(quiz = fork tokens; probe/judge = "
                  "extra prompt tokens vs clock)", fontsize=9)
    fig.suptitle("Fig 4 — the total cost of observation")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figures" / "fig4_cost_of_observation.png", dpi=160)


def fig5(m):
    fig, ax = plt.subplots(figsize=(8.8, 5))
    pts = []
    for arm, s in m["arms"].items():
        total = (s["prompt_tokens"] + s["completion_tokens"]
                 + s.get("quiz_tokens", 0)) / 1000
        pts.append((arm, total, s["accuracy"]))
        big = arm in ("QUIZ", "ACT_probe", "Z_reground")
        ax.scatter(total, s["accuracy"], s=240 if big else 100,
                   c=CAT_COLORS[m["categories"][arm]], edgecolors="black",
                   linewidths=0.6, zorder=3)
        ax.annotate(arm, (total, s["accuracy"]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    front = []
    for arm, x, y in sorted(pts, key=lambda p: p[1]):
        if not front or y > front[-1][2]:
            front.append((arm, x, y))
    ax.plot([p[1] for p in front], [p[2] for p in front], "--",
            color="#666666", linewidth=1.0, zorder=2, label="Pareto front")
    ax.set_xlabel("total tokens per task, agent + resets + monitoring "
                  "(thousands)")
    ax.set_ylabel("per-turn accuracy")
    ax.set_title(f"Fig 5 — the decision plane ({m['model']}, "
                 f"n={m['n_tasks']})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figures" / "fig5_decision_plane.png", dpi=160)


def main():
    m = json.loads((RESULTS_DIR / "metrics.json").read_text())
    pred = json.loads((RESULTS_DIR / "prediction.json").read_text())
    (RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
    fig2(pred)
    fig3(m)
    fig4(m)
    fig5(m)
    print("figures written to", RESULTS_DIR / "figures")


if __name__ == "__main__":
    main()
