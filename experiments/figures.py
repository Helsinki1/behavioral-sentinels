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
BAR_ORDER = ["say_my_name", "remember_fact", "format_response", "early_decision",
             "multi_resolution", "variable_check",
             "turn_number", "context_length", "LLM_judge", "random_compaction"]


def _bar_panel(ax, m, metric):
    # canary rows sit 0.9 higher, opening a gap for the group divider + header
    ys = [len(BAR_ORDER) - 1 - i + (0.9 if sig in CANARIES else 0)
          for i, sig in enumerate(BAR_ORDER)]
    for y, sig in zip(ys, BAR_ORDER):
        v = load_metrics(sig, m, "5")[metric]
        if v is None:  # precision undefined: the signal never fired
            ax.annotate("never fired", (0.015, y), fontsize=8.5, color=MUTED,
                        va="center", style="italic")
            continue
        ax.barh(y, v, height=0.62, color=MODEL_COLORS[m],
                edgecolor=SURFACE, linewidth=1)
        ax.annotate(f"{v:.2f}", (v + 0.015, y), fontsize=8.5, color=INK2, va="center")
    style(ax, ygrid=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_yticks(ys, BAR_ORDER)
    ax.tick_params(axis="y", labelsize=9, labelcolor=INK2)
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.5", "0.75", "1"])
    ax.set_ylim(-0.6, len(BAR_ORDER) + 1.25)
    # separator between the six canaries (top block) and four traditional signals
    ax.axhline(3.95, color=BASELINE, linewidth=0.8)
    for label, y, va in [("canaries", 10.35, "bottom"), ("traditional", 3.75, "top")]:
        ax.annotate(label, xy=(-0.02, y), xycoords=("axes fraction", "data"),
                    fontsize=8, color=MUTED, ha="right", va=va, style="italic")
    ax.set_title(MODEL_LABELS[m], fontsize=10, color=INK2, loc="left")


def _bar_figure(metric, fname, head, sub):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.0))
    fig.subplots_adjust(top=0.80, bottom=0.10, left=0.17, right=0.985, wspace=0.62)
    for ax, m in zip(axes, MODELS):
        _bar_panel(ax, m, metric)
    title(fig, head, sub)
    fig.savefig(FIG_DIR / fname, dpi=220)
    plt.close(fig)


def fig1():
    _bar_figure(
        "precision", "fig1a_precision.png",
        "When a canary fires, it is almost always right",
        "Precision at K=5: of the tasks where the signal fired, how many really hallucinated "
        "within 5 turns · 200 tasks per model")
    _bar_figure(
        "recall", "fig1b_recall.png",
        "But only the demanding signals catch enough hallucinations",
        "Recall at K=5: of the tasks that hallucinated, how many the signal predicted in time "
        "· 200 tasks per model")


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
