"""Publication figures for the behavioral-sentinels results.

Four charts, each comparing two dimensions on x/y with model or signal as
the third (color/line) dimension. Palette and chart chrome follow the
validated reference palette (light mode).
"""
import glob
import json
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .config import RESULTS_DIR, RUNS_DIR

FIG_DIR = RESULTS_DIR / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

MODEL_COLORS = {"gpt-4o-mini": "#2a78d6", "gpt-oss-20b": "#eb6834"}
MODEL_LABELS = {"gpt-4o-mini": "gpt-4o-mini (proprietary)",
                "gpt-oss-20b": "gpt-oss-20b (open)"}
SIGNAL_COLORS = {
    "variable_check": "#2a78d6",
    "multi_resolution": "#eb6834",
    "turn_number": "#1baf7a",
    "LLM_judge": "#eda100",
    "context_length": "#e87ba4",
    "random_compaction": "#008300",
}
MODELS = ["gpt-4o-mini", "gpt-oss-20b"]
CANARIES = ["say_my_name", "remember_fact", "format_response",
            "variable_check", "early_decision", "multi_resolution"]
KS = ["2", "5", "10", "inf"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.labelcolor": INK2,
    "font.size": 10,
})


def style(ax, ygrid=True):
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(BASELINE)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def title(fig, text, sub):
    fig.text(0.02, 0.965, text, fontsize=13.5, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.915, sub, fontsize=9.5, color=INK2, va="top")


def load_metrics(sig, model, K):
    if sig in CANARIES:
        d = json.loads((RESULTS_DIR / sig / f"metrics_{model}.json").read_text())
        return d["metrics_by_K"][K]
    d = json.loads((RESULTS_DIR / "Traditional" / sig / f"metrics_{model}.json").read_text())
    if "metrics_by_K" in d:
        return d["metrics_by_K"][K]
    th = str(d["best_threshold_at_primary_K"])
    return d["sweep"][th][K]


def load_runs(model, condition):
    out = []
    for f in sorted(glob.glob(str(RUNS_DIR / model / condition / "task_*.json"))):
        out.append(json.loads(open(f).read()))
    return out


# ------------------------------------------------------------------ fig 1
def fig1():
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    fig.subplots_adjust(top=0.83, bottom=0.16, left=0.09, right=0.97)
    sigs = CANARIES + ["turn_number", "LLM_judge", "context_length", "random_compaction"]
    skipped = {m: [] for m in MODELS}
    label_offsets = {  # (dx, dy, ha)
        ("gpt-4o-mini", "variable_check"): (-4, -18, "right"),
        ("gpt-4o-mini", "turn_number"): (0, 10, "center"),
        ("gpt-4o-mini", "LLM_judge"): (10, -4, "left"),
        ("gpt-4o-mini", "random_compaction"): (10, 4, "left"),
        ("gpt-4o-mini", "context_length"): (10, 4, "left"),
        ("gpt-4o-mini", "multi_resolution"): (10, -10, "left"),
        ("gpt-oss-20b", "variable_check"): (10, 2, "left"),
        ("gpt-oss-20b", "turn_number"): (2, -16, "left"),
        ("gpt-oss-20b", "LLM_judge"): (10, -4, "left"),
        ("gpt-oss-20b", "random_compaction"): (10, 2, "left"),
        ("gpt-oss-20b", "context_length"): (10, 4, "left"),
        ("gpt-oss-20b", "multi_resolution"): (8, -14, "left"),
    }
    for m in MODELS:
        for sig in sigs:
            met = load_metrics(sig, m, "5")
            p, r = met["precision"], met["recall"]
            if p is None:
                skipped[m].append(sig)
                continue
            marker = "o" if sig in CANARIES else "s"
            ax.scatter(r, p, s=90, color=MODEL_COLORS[m], marker=marker,
                       edgecolors=SURFACE, linewidths=1.5, zorder=3)
            off = label_offsets.get((m, sig))
            if off:
                ax.annotate(sig, (r, p), xytext=off[:2], textcoords="offset points",
                            fontsize=8.5, color=INK2, ha=off[2])
    style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_xlim(-0.03, 0.72)
    ax.set_ylim(0.28, 1.06)
    ax.set_xlabel("recall (share of hallucinating tasks predicted)")
    ax.set_ylabel("precision (share of firings that were right)")
    handles = [
        Line2D([], [], marker="o", ls="", ms=9, mfc=MODEL_COLORS[m], mec=SURFACE,
               label=MODEL_LABELS[m]) for m in MODELS
    ] + [
        Line2D([], [], marker="o", ls="", ms=8, mfc="#b7b5ae", mec=SURFACE, label="canary signal"),
        Line2D([], [], marker="s", ls="", ms=8, mfc="#b7b5ae", mec=SURFACE, label="traditional signal"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.5,
              labelcolor=INK2)
    title(fig, "Only the demanding canary competes with tuned traditional signals",
          "Precision vs recall predicting the first hallucination within K=5 turns · 200 synthetic "
          "state book-keeping tasks per point")
    note = ("Not shown (never fired, precision undefined): gpt-4o-mini " +
            ", ".join(skipped["gpt-4o-mini"]) + ".\n"
            "The unlabeled gpt-oss-20b points at precision 1.0 are its four simple canaries "
            "(recall ≤ 0.04).")
    fig.text(0.02, 0.015, note, fontsize=8, color=MUTED)
    fig.savefig(FIG_DIR / "fig1_precision_vs_recall.png", dpi=220)
    plt.close(fig)


# ------------------------------------------------------------------ fig 2
def fig2():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.0), sharey=True)
    fig.subplots_adjust(top=0.80, bottom=0.20, left=0.07, right=0.98, wspace=0.08)
    xs = range(len(KS))
    direct = {"variable_check", "turn_number", "LLM_judge", "random_compaction"}
    dy = {("gpt-oss-20b", "random_compaction"): 9,
          ("gpt-oss-20b", "turn_number"): -1,
          ("gpt-oss-20b", "variable_check"): -11}
    for ax, m in zip(axes, MODELS):
        for sig, color in SIGNAL_COLORS.items():
            ys = [load_metrics(sig, m, K)["recall"] or 0.0 for K in KS]
            ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5.5,
                    markeredgecolor=SURFACE, markeredgewidth=1)
            if sig in direct:
                ax.annotate(sig, (xs[-1], ys[-1]), xytext=(6, dy.get((m, sig), 0)),
                            textcoords="offset points", fontsize=8, color=color,
                            va="center")
        style(ax)
        ax.set_xticks(list(xs), ["2", "5", "10", "∞"])
        ax.set_xlim(-0.3, 4.1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("prediction window K (turns)")
        ax.set_title(MODEL_LABELS[m], fontsize=10, color=INK2, loc="left")
    axes[0].set_ylabel("recall")
    handles = [Line2D([], [], color=c, lw=2, label=s) for s, c in SIGNAL_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               fontsize=8, labelcolor=INK2, bbox_to_anchor=(0.5, 0.0))
    title(fig, "Widen the window and the sentinel's recall keeps climbing",
          "Recall vs prediction window K · lines = signal · four never-firing simple canaries omitted "
          "(recall ≤ 0.04 at every K)")
    fig.savefig(FIG_DIR / "fig2_recall_vs_window.png", dpi=220)
    plt.close(fig)


# ------------------------------------------------------------------ fig 3
def fig3():
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    fig.subplots_adjust(top=0.82, bottom=0.13, left=0.09, right=0.97)
    max_t = 35
    for m in MODELS:
        runs = load_runs(m, "variable_check")
        n = len(runs)
        for key, ls, lab in [("first_canary_fail", (0, (4, 2)), "canary fires"),
                             ("first_hallucination", "-", "hallucination")]:
            events = [r[key] for r in runs if r[key] is not None]
            ys = [sum(1 for e in events if e <= t) / n for t in range(1, max_t + 1)]
            ax.plot(range(1, max_t + 1), ys, ls=ls, color=MODEL_COLORS[m], linewidth=2)
    style(ax)
    ax.set_xlim(1, max_t)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("turn")
    ax.set_ylabel("cumulative share of the 200 tasks")
    handles = [Line2D([], [], color=MODEL_COLORS[m], lw=2, label=MODEL_LABELS[m]) for m in MODELS] + [
        Line2D([], [], color=INK2, lw=2, ls=(0, (4, 2)), label="first canary failure (signal)"),
        Line2D([], [], color=INK2, lw=2, label="first hallucination (outcome)"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK2)
    ax.annotate("dashed leads solid:\nthe sentinel fires first", xy=(0.33, 0.72),
                xycoords="axes fraction", fontsize=9, color=INK2, ha="center")
    title(fig, "The variable-check sentinel fires ahead of the hallucination it predicts",
          "Cumulative onset curves within the variable_check runs · dashed = signal, solid = outcome · "
          "color = model")
    fig.savefig(FIG_DIR / "fig3_onset_curves.png", dpi=220)
    plt.close(fig)


# ------------------------------------------------------------------ fig 4
def fig4():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), sharey=True)
    fig.subplots_adjust(top=0.78, bottom=0.15, left=0.07, right=0.98, wspace=0.08)
    for ax, m in zip(axes, MODELS):
        runs = load_runs(m, "variable_check")
        leads = [r["first_hallucination"] - r["first_canary_fail"] for r in runs
                 if r["first_hallucination"] is not None and r["first_canary_fail"] is not None
                 and r["first_canary_fail"] <= r["first_hallucination"]]
        bins = range(0, max(leads) + 2)
        ax.hist(leads, bins=bins, color=MODEL_COLORS[m], edgecolor=SURFACE, linewidth=1.5)
        med = statistics.median(leads)
        ax.axvline(med + 0.5, color=INK2, linewidth=1.2, ls=(0, (3, 2)))
        ax.annotate(f"median {med:g} turns", xy=(med + 0.9, 0.92), xycoords=("data", "axes fraction"),
                    fontsize=9, color=INK2)
        style(ax)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_xlabel("warning lead time (turns from canary failure to hallucination)")
        ax.set_title(f"{MODEL_LABELS[m]}  ·  n={len(leads)} early warnings", fontsize=10,
                     color=INK2, loc="left")
    axes[0].set_ylabel("tasks")
    title(fig, "When the sentinel fires early, it buys 4–7 turns of warning",
          "Lead time between the first variable_check canary failure and the first hallucination "
          "(tasks where the canary fired at or before it)")
    fig.savefig(FIG_DIR / "fig4_lead_time.png", dpi=220)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig1(); fig2(); fig3(); fig4()
    print("wrote", *sorted(p.name for p in FIG_DIR.glob("*.png")))


if __name__ == "__main__":
    main()
