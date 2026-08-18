"""Experiment-4 figures: the success/intervention-cost tradeoff, and the
paired contrasts with their uncertainty shown honestly.

Palette is the validated light-mode reference instance shared with
experiments/figures.py (all checks PASS at surface #fcfcfb).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.figures import BASELINE, GRID, INK, INK2, MUTED, SURFACE, style, title
from .config4 import RESULTS_DIR

FIG = RESULTS_DIR / "figures"
ORDER = ["A_no_reset", "B_random", "C_scheduled", "C_prime_carried", "D_sentinel", "F_oracle"]
LABEL = {"A_no_reset": "no reset", "B_random": "random", "C_scheduled": "scheduled (clock)",
         "C_prime_carried": "scheduled + probe", "D_sentinel": "sentinel-triggered",
         "F_oracle": "oracle (perfect)"}
COLOR = {"A_no_reset": "#898781", "B_random": "#e87ba4", "C_scheduled": "#2a78d6",
         "C_prime_carried": "#6ea8dc", "D_sentinel": "#eb6834", "F_oracle": "#1baf7a"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.edgecolor": BASELINE, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.labelcolor": INK2, "font.size": 10})


def fig_tradeoff(m):
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    fig.subplots_adjust(top=0.80, bottom=0.12, left=0.10, right=0.97)
    dy = {"C_prime_carried": -17, "B_random": 9, "A_no_reset": 8, "D_sentinel": -17}
    dx = {"D_sentinel": -128, "B_random": -60, "C_prime_carried": -150}
    for a in ORDER:
        if a not in m["arms"]:
            continue
        s = m["arms"][a]
        ax.scatter(s["resets_per_task"], s["accuracy"], s=170, color=COLOR[a],
                   edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(LABEL[a], (s["resets_per_task"], s["accuracy"]),
                    xytext=(dx.get(a, 9), dy.get(a, 8)), textcoords="offset points",
                    fontsize=9, color=COLOR[a], fontweight="bold")
    style(ax)
    ax.set_xlabel("average resets per task  (intervention budget)")
    ax.set_ylabel("task accuracy  (share of turns with zero errors)")
    ax.set_xlim(-0.5, 5.2)
    ax.annotate("better", xy=(0.055, 0.93), xycoords="axes fraction", fontsize=8.5,
                color=MUTED, style="italic")
    ax.annotate("", xy=(0.05, 0.97), xytext=(0.05, 0.86), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    title(fig, "A perfect signal buys real accuracy - with four times fewer resets",
          "40 paired coding tasks - gpt-oss-20b - full horizon - the sentinel sits below "
          "the clock it was meant to beat")
    fig.savefig(FIG / "fig1_tradeoff.png", dpi=220)
    plt.close(fig)


def fig_contrasts(m):
    cs = [c for c in m["contrasts"]]
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    fig.subplots_adjust(top=0.78, bottom=0.13, left=0.42, right=0.97)
    ys = range(len(cs))[::-1]
    for y, c in zip(ys, cs):
        lo, hi = c["ci95"]
        sig = c["significant"]
        col = "#1baf7a" if (sig and c["mean_delta"] > 0) else ("#eb6834" if sig else MUTED)
        ax.plot([lo, hi], [y, y], color=col, lw=2.2, solid_capstyle="round", zorder=2)
        ax.scatter([c["mean_delta"]], [y], s=70, color=col, edgecolor=SURFACE,
                   linewidth=1.6, zorder=3)
    ax.axvline(0, color=BASELINE, lw=1.2, zorder=1)
    style(ax, ygrid=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_yticks(list(ys), [c["contrast"].split(":")[1].strip() for c in cs])
    ax.tick_params(axis="y", labelsize=8.5, labelcolor=INK2)
    ax.set_xlabel("paired difference in task accuracy (95% bootstrap CI)")
    ax.set_ylim(-0.6, len(cs) - 0.4)
    title(fig, "Only the oracle contrast clears zero",
          "Paired per-task deltas with bootstrap CIs - 40 tasks - green = significant gain, "
          "grey = indistinguishable from no effect")
    fig.savefig(FIG / "fig2_contrasts.png", dpi=220)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    m = json.loads((RESULTS_DIR / "metrics.json").read_text())
    fig_tradeoff(m); fig_contrasts(m)
    print("wrote", *sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
