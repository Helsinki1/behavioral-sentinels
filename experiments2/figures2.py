"""Publication figures for experiment 2 (dynamic canaries, coding task).

Deliberately the same four charts as experiments/figures.py so the two
experiments can be read side by side, plus one chart that only makes sense
here: what the coding hallucinations actually were.

Chart chrome and the categorical palette are imported from experiment 1's
figure module -- the palette is the validated light-mode reference instance
(6 slots, all checks PASS at surface #fcfcfb; the sub-3:1 contrast warning is
relieved by direct labels on every mark and by the markdown tables in
results2/).
"""
import glob
import json
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from experiments.figures import (BASELINE, GRID, INK, INK2, MUTED, SURFACE,
                                 style, title)

from .config2 import (CANARY_CONDITIONS, K_VALUES, PRIMARY_K, RESULTS_DIR,
                      RUNS_DIR, TRADITIONAL_SIGNALS)

FIG_DIR = RESULTS_DIR / "figures"

def _available_models():
    """Only chart models that actually produced a scored metrics file."""
    return [m for m in ["gpt-4o-mini", "gpt-oss-20b"]
            if (RESULTS_DIR / CANARY_CONDITIONS[0] / f"metrics_{m}.json").exists()]


MODELS = ["gpt-4o-mini", "gpt-oss-20b"]
MODEL_COLORS = {"gpt-4o-mini": "#2a78d6", "gpt-oss-20b": "#eb6834"}
MODEL_LABELS = {"gpt-4o-mini": "gpt-4o-mini (proprietary)",
                "gpt-oss-20b": "gpt-oss-20b (open)"}

# fixed categorical order -- validated 6-slot palette, never cycled
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
KS = [str(k) for k in K_VALUES]

# canaries first (dynamic ones, then the static control), then traditional
BAR_ORDER = CANARY_CONDITIONS + TRADITIONAL_SIGNALS
N_CANARIES = len(CANARY_CONDITIONS)

ERROR_LABELS = {
    "other": "other (rarer kinds)",
    "wrong_sig": "wrong signature reported",
    "missing_sig": "signature not reported",
    "wrong_def_sig": "emitted def has wrong params",
    "missing_def": "requested edit not emitted",
    "fabricated_sig": "signature invented for a deleted fn",
    "fabricated_symbol": "call to a symbol that does not exist",
    "missing_wired_call": "required call omitted",
    "syntax_error": "code does not parse",
    "no_code_block": "no code emitted at all",
}

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


def load_metrics(sig, model, K):
    if sig in CANARY_CONDITIONS:
        d = json.loads((RESULTS_DIR / sig / f"metrics_{model}.json").read_text())
        return d["metrics_by_K"][K]
    d = json.loads((RESULTS_DIR / "Traditional" / sig / f"metrics_{model}.json").read_text())
    if "metrics_by_K" in d:
        return d["metrics_by_K"][K]
    return d["sweep"][str(d["best_threshold_at_primary_K"])][K]


def load_runs(model, condition):
    return [json.loads(open(f).read())
            for f in sorted(glob.glob(str(RUNS_DIR / model / condition / "task_*.json")))]


def headline_canary():
    """The dynamic canary with the best mean F1 at the primary K -- fig3/fig4
    are drawn for it. Chosen from the data, not hard-coded."""
    scored = {c: statistics.mean(load_metrics(c, m, str(PRIMARY_K))["f1"] for m in MODELS)
              for c in CANARY_CONDITIONS}
    return max(scored, key=scored.get), scored


# ------------------------------------------------------------------ fig 1
def _bar_panel(ax, m, metric):
    ys = [len(BAR_ORDER) - 1 - i + (0.9 if sig in CANARY_CONDITIONS else 0)
          for i, sig in enumerate(BAR_ORDER)]
    for y, sig in zip(ys, BAR_ORDER):
        v = load_metrics(sig, m, str(PRIMARY_K))[metric]
        if v is None:
            ax.annotate("never fired", (0.015, y), fontsize=8.5, color=MUTED,
                        va="center", style="italic")
            continue
        ax.barh(y, v, height=0.62, color=MODEL_COLORS[m], edgecolor=SURFACE, linewidth=1)
        ax.annotate(f"{v:.2f}", (v + 0.015, y), fontsize=8.5, color=INK2, va="center")
    style(ax, ygrid=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_yticks(ys, BAR_ORDER)
    ax.tick_params(axis="y", labelsize=9, labelcolor=INK2)
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.5", "0.75", "1"])
    ax.set_ylim(-0.6, len(BAR_ORDER) + 1.25)
    divider = len(TRADITIONAL_SIGNALS) - 0.05
    ax.axhline(divider, color=BASELINE, linewidth=0.8)
    ax.annotate("canaries", xy=(-0.02, len(BAR_ORDER) + 0.35),
                xycoords=("axes fraction", "data"), fontsize=8, color=MUTED,
                ha="right", va="bottom", style="italic")
    ax.annotate("traditional", xy=(-0.02, divider - 0.2),
                xycoords=("axes fraction", "data"), fontsize=8, color=MUTED,
                ha="right", va="top", style="italic")
    ax.set_title(MODEL_LABELS[m], fontsize=10, color=INK2, loc="left")


def _bar_figure(metric, fname, head, sub):
    fig, axes = plt.subplots(1, len(MODELS), figsize=(9.6, 5.2), squeeze=False)
    axes = axes[0]
    one = len(MODELS) == 1
    fig.subplots_adjust(top=0.80, bottom=0.10, left=0.22 if one else 0.19,
                        right=0.72 if one else 0.985, wspace=0.66)
    for ax, m in zip(axes, MODELS):
        _bar_panel(ax, m, metric)
    title(fig, head, sub)
    fig.savefig(FIG_DIR / fname, dpi=220)
    plt.close(fig)


def fig1(n_tasks):
    _bar_figure(
        "precision", "fig1a_precision.png",
        "Exp 2: a canary only warns if its answer can't be copied",
        f"Precision at K={PRIMARY_K} on the coding task: of the tasks where the signal fired, how many "
        f"really hallucinated within {PRIMARY_K} turns · {n_tasks} tasks per model")
    _bar_figure(
        "recall", "fig1b_recall.png",
        "Exp 2: recomputed canaries catch hallucinations; copyable ones don't",
        f"Recall at K={PRIMARY_K} on the coding task: of the tasks that hallucinated, how many the signal "
        f"predicted in time · {n_tasks} tasks per model")


# ------------------------------------------------------------------ fig 2
def fig2(n_tasks):
    _, scored = headline_canary()
    top2 = sorted(CANARY_CONDITIONS, key=lambda c: -scored[c])[:2]
    shown = top2 + TRADITIONAL_SIGNALS          # 6 slots, fixed order
    colors = dict(zip(shown, PALETTE))
    omitted = [c for c in CANARY_CONDITIONS if c not in top2]

    fig, axes = plt.subplots(1, len(MODELS), figsize=(9.6, 5.2),
                             sharey=True, squeeze=False)
    axes = axes[0]
    one = len(MODELS) == 1
    fig.subplots_adjust(top=0.79, bottom=0.22, left=0.10,
                        right=0.62 if one else 0.98, wspace=0.08)
    xs = range(len(KS))
    for ax, m in zip(axes, MODELS):
        for sig in shown:
            ys = [load_metrics(sig, m, K)["recall"] or 0.0 for K in KS]
            ax.plot(xs, ys, color=colors[sig], linewidth=2, marker="o", markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.annotate(sig, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                        fontsize=7.5, color=colors[sig], va="center")
        style(ax)
        ax.set_xticks(list(xs), ["2", "5", "10", "∞"])
        ax.set_xlim(-0.3, 4.6)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("prediction window K (turns)")
        ax.set_title(MODEL_LABELS[m], fontsize=10, color=INK2, loc="left")
    axes[0].set_ylabel("recall")
    handles = [Line2D([], [], color=colors[s], lw=2, label=s) for s in shown]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               fontsize=8, labelcolor=INK2, bbox_to_anchor=(0.5, 0.0))
    title(fig, "Widening the window keeps lifting the canaries' recall",
          f"Recall vs prediction window K · {n_tasks} coding tasks · the four lowest-F1 "
          f"canaries are omitted for legibility (all rows in results2/SUMMARY.md)")
    fig.savefig(FIG_DIR / "fig2_recall_vs_window.png", dpi=220)
    plt.close(fig)


# ------------------------------------------------------------------ fig 3
def fig3(cond, n_tasks):
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    fig.subplots_adjust(top=0.82, bottom=0.13, left=0.09, right=0.97)
    max_t = 30
    for m in MODELS:
        runs = load_runs(m, cond)
        n = len(runs) or 1
        for key, ls in [("first_canary_fail", (0, (4, 2))), ("first_hallucination", "-")]:
            events = [r[key] for r in runs if r[key] is not None]
            ys = [sum(1 for e in events if e <= t) / n for t in range(1, max_t + 1)]
            ax.plot(range(1, max_t + 1), ys, ls=ls, color=MODEL_COLORS[m], linewidth=2)
    style(ax)
    ax.set_xlim(1, max_t)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("turn")
    ax.set_ylabel(f"cumulative share of the {n_tasks} tasks")
    handles = [Line2D([], [], color=MODEL_COLORS[m], lw=2, label=MODEL_LABELS[m]) for m in MODELS] + [
        Line2D([], [], color=INK2, lw=2, ls=(0, (4, 2)), label="first canary failure (signal)"),
        Line2D([], [], color=INK2, lw=2, label="first hallucination (outcome)")]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK2)
    # An honest reading of these curves: the dashed line sits BELOW the solid
    # because the canary fires on fewer trajectories than hallucinate -- it is
    # not a claim that it fires later. The lead is a per-trajectory quantity,
    # so state it as one.
    notes = []
    for m in MODELS:
        runs = load_runs(m, cond)
        both = [r for r in runs if r["first_hallucination"] is not None
                and r["first_canary_fail"] is not None]
        early = [r for r in both if r["first_canary_fail"] <= r["first_hallucination"]]
        if both:
            med = statistics.median(r["first_hallucination"] - r["first_canary_fail"]
                                    for r in early) if early else 0
            notes.append(f"{m}: canary fired on {sum(1 for r in runs if r['first_canary_fail']) / (len(runs) or 1):.0%} "
                         f"of tasks; where both occur it leads {len(early)}/{len(both)} "
                         f"of the time (median {med:g} turns)")
    ax.annotate("\n".join(notes), xy=(0.03, 0.93), xycoords="axes fraction",
                fontsize=8.5, color=INK2, ha="left", va="top")
    title(fig, f"Where the {cond} sentinel fires, it fires early",
          f"Cumulative onset curves within the {cond} runs · dashed = signal, solid = outcome · "
          "the dashed curve is lower because the canary fires on fewer tasks, not later on them")
    fig.savefig(FIG_DIR / "fig3_onset_curves.png", dpi=220)
    plt.close(fig)


# ------------------------------------------------------------------ fig 4
def fig4(cond):
    fig, axes = plt.subplots(1, len(MODELS), figsize=(9.6, 4.6),
                             sharey=True, squeeze=False)
    axes = axes[0]
    one = len(MODELS) == 1
    fig.subplots_adjust(top=0.78, bottom=0.15, left=0.10,
                        right=0.60 if one else 0.98, wspace=0.08)
    meds = []
    for ax, m in zip(axes, MODELS):
        runs = load_runs(m, cond)
        leads = [r["first_hallucination"] - r["first_canary_fail"] for r in runs
                 if r["first_hallucination"] is not None and r["first_canary_fail"] is not None
                 and r["first_canary_fail"] <= r["first_hallucination"]]
        if not leads:
            ax.annotate("no early warnings", (0.5, 0.5), xycoords="axes fraction",
                        ha="center", color=MUTED, style="italic")
            style(ax)
            continue
        ax.hist(leads, bins=range(0, max(leads) + 2), color=MODEL_COLORS[m],
                edgecolor=SURFACE, linewidth=1.5)
        med = statistics.median(leads)
        meds.append(med)
        ax.axvline(med + 0.5, color=INK2, linewidth=1.2, ls=(0, (3, 2)))
        ax.annotate(f"median {med:g} turns", xy=(med + 0.9, 0.92),
                    xycoords=("data", "axes fraction"), fontsize=9, color=INK2)
        style(ax)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_xlabel("warning lead time (turns from canary failure to hallucination)")
        ax.set_title(f"{MODEL_LABELS[m]}  ·  n={len(leads)} early warnings",
                     fontsize=10, color=INK2, loc="left")
    axes[0].set_ylabel("tasks")
    span = f"{min(meds):g}–{max(meds):g}" if len(set(meds)) > 1 else f"{meds[0]:g}" if meds else "?"
    title(fig, f"When the sentinel fires early it buys ~{span} turns of warning",
          f"Lead time between the first {cond} canary failure and the first coding hallucination "
          "(tasks where the canary fired at or before it)")
    fig.savefig(FIG_DIR / "fig4_lead_time.png", dpi=220)
    plt.close(fig)


# ------------------------------------------------------------------ fig 5
def fig5(n_tasks):
    """Experiment-2 only: what the coding hallucinations actually were."""
    mixes = {}
    for m in MODELS:
        runs = load_runs(m, "baseline")
        c = {}
        for r in runs:
            if r["first_hallucination"] is None:
                continue
            for kind, _ in r["records"][-1]["errors"]:
                c[kind] = c.get(kind, 0) + 1
        mixes[m] = c
    ranked = sorted({k for c in mixes.values() for k in c},
                    key=lambda k: -sum(c.get(k, 0) for c in mixes.values()))
    # never cycle a categorical palette: keep the top 5 kinds and fold the
    # long tail into a single neutral "other" slot
    head, tail = ranked[:5], ranked[5:]
    if tail:
        for m in mixes:
            mixes[m] = {**{k: v for k, v in mixes[m].items() if k in head},
                        "other": sum(v for k, v in mixes[m].items() if k in tail)}
    kinds = head + (["other"] if tail else [])
    colors = dict(zip(head, PALETTE))
    if tail:
        colors["other"] = BASELINE

    fig, ax = plt.subplots(figsize=(9.6, 3.6))
    fig.subplots_adjust(top=0.70, bottom=0.16, left=0.20, right=0.66)
    for row, m in enumerate(MODELS):
        tot = sum(mixes[m].values()) or 1
        left = 0.0
        for k in kinds:
            v = mixes[m].get(k, 0) / tot
            if v <= 0:
                continue
            ax.barh(row, v, left=left, height=0.5, color=colors[k],
                    edgecolor=SURFACE, linewidth=2)   # 2px surface gap between segments
            if v > 0.07:
                ax.annotate(f"{v:.0%}", (left + v / 2, row), ha="center", va="center",
                            fontsize=8.5, color=SURFACE, fontweight="bold")
            left += v
    style(ax, ygrid=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_yticks(range(len(MODELS)), [MODEL_LABELS[m] for m in MODELS])
    ax.tick_params(axis="y", labelsize=9, labelcolor=INK2)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25%", "50%", "75%", "100%"])
    ax.set_ylim(-0.55, len(MODELS) - 0.45)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k],
                             label=ERROR_LABELS.get(k, k)) for k in kinds]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.675, 0.44),
               frameon=False, fontsize=8, labelcolor=INK2)
    title(fig, "What the first coding hallucination actually was",
          f"Composition of the error kinds present on the first hallucinating turn · "
          f"baseline (no-canary) runs · {n_tasks} tasks per model")
    fig.savefig(FIG_DIR / "fig5_error_mix.png", dpi=220)
    plt.close(fig)


def main():
    global MODELS
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODELS = _available_models()
    print("charting models:", MODELS)
    n_tasks = len(load_runs(MODELS[0], "baseline")) or len(load_runs(MODELS[0], CANARY_CONDITIONS[0]))
    cond, scored = headline_canary()
    print("mean F1@K=%s by canary: %s" % (PRIMARY_K,
          {k: round(v, 3) for k, v in sorted(scored.items(), key=lambda x: -x[1])}))
    print("headline canary ->", cond)
    fig1(n_tasks); fig2(n_tasks); fig3(cond, n_tasks); fig4(cond); fig5(n_tasks)
    print("wrote", *sorted(p.name for p in FIG_DIR.glob("*.png")))


if __name__ == "__main__":
    main()
