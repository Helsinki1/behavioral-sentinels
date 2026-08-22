"""Experiment 10 figures — the decision map.

Given YOUR cost of a restart and YOUR cost of a token, which reset policy
should you run? The map is the upper envelope of the policy planes.
Palette is the validated light-mode reference instance, fixed order, never
cycled; only policies that actually win a region are drawn.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from experiments.figures import BASELINE, INK, INK2, MUTED, SURFACE, style, title
from .config10 import REGIMES, RESULTS_DIR

FIG = RESULTS_DIR / "figures"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
NICE = {"clock": "reset on a clock", "llm_judge": "LLM judge",
        "sentinel_zerocarry": "zero-carry sentinel", "ctx_growth": "context-growth",
        "never_reset": "never reset", "oracle": "oracle", "dense_clock": "dense clock",
        "random": "random", "sentinel_carried": "carried sentinel"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.edgecolor": BASELINE, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.labelcolor": INK2, "font.size": 10})


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    d = json.loads((RESULTS_DIR / "breakeven.json").read_text())
    regimes = list(d)
    winners = sorted({w for r in regimes for row in d[r]["grid"] for w in row},
                     key=lambda k: -sum(row.count(k) for r in regimes for row in d[r]["grid"]))
    color = {w: PALETTE[i % len(PALETTE)] for i, w in enumerate(winners)}

    fig, axes = plt.subplots(1, len(regimes), figsize=(10.2, 4.9), sharey=True)
    fig.subplots_adjust(top=0.70, bottom=0.30, left=0.085, right=0.985, wspace=0.09)
    for ax, r in zip(axes, regimes):
        R, T, grid = d[r]["R_grid"], d[r]["T_grid"], d[r]["grid"]
        idx = {w: i for i, w in enumerate(winners)}
        Z = [[idx[c] for c in row] for row in grid]
        cmap = matplotlib.colors.ListedColormap([color[w] for w in winners])
        ax.pcolormesh(R, T, Z, cmap=cmap, vmin=-0.5, vmax=len(winners) - 0.5,
                      shading="nearest", rasterized=True)
        style(ax, ygrid=False)
        ax.set_xlabel("cost of one restart (accuracy-equivalents)")
        ax.set_title(r.upper() + " operator", fontsize=10, color=INK2, loc="left")
    axes[0].set_ylabel("cost of 1k prompt tokens\n(accuracy-equivalents)")
    for ax in axes:
        ax.tick_params(labelsize=9)
    handles = [Patch(facecolor=color[w], label=NICE.get(w, w)) for w in winners]
    fig.legend(handles=handles, loc="lower center", ncol=min(3, len(winners)),
               frameon=False, fontsize=8.5, labelcolor=INK2, bbox_to_anchor=(0.5, 0.005))
    title(fig, "When is a sentinel worth it? It depends on what a restart costs you",
          "Which policy maximises utility at each price point - 90 tasks, gpt-oss-20b - "
          "re-analysis of experiments 5 and 6, no new runs")
    fig.savefig(FIG / "fig1_decision_map.png", dpi=220)
    plt.close(fig)
    print("winning policies:", winners)
    print("wrote", *sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
