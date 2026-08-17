"""Publication figures for experiment 3 -- the experiment-1/2 chart set,
drawn once PER TASK SET so the three domains can be read side by side:

  results3/figures/<set>/fig1a_precision.png   signal bars (canary vs traditional)
  results3/figures/<set>/fig1b_recall.png
  results3/figures/<set>/fig2_recall_vs_window.png
  results3/figures/<set>/fig3_onset_curves.png  (headline canary)
  results3/figures/<set>/fig4_lead_time.png     (headline canary)

Chart chrome and the categorical palette are imported from experiment 1's
figure module -- the validated light-mode reference instance (6 fixed slots,
never cycled; direct labels on every mark relieve the contrast warning, and
every number is also in the results3/ markdown tables).
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

from .config3 import (CANARY_CONDITIONS, K_VALUES, PRIMARY_K, RESULTS_DIR,
                      RUNS_DIR, TASK_SETS, TRADITIONAL_SIGNALS, conditions_for)

# fixed categorical slots (same 6-slot validated palette as experiments 1-2)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MODEL_COLORS = {"gpt-4o-mini": "#2a78d6", "gpt-oss-20b": "#eb6834",
                "llama-v3p1-8b": "#1baf7a"}
MODEL_LABELS = {"gpt-4o-mini": "gpt-4o-mini (proprietary)",
                "gpt-oss-20b": "gpt-oss-20b (open)",
                "llama-v3p1-8b": "llama-3.1-8b (open, small)"}
KS = [str(k) for k in K_VALUES]

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


def canary_conds(task_set):
    return [c for c in conditions_for(task_set) if c != "baseline"]


def bar_order(task_set):
    return canary_conds(task_set) + TRADITIONAL_SIGNALS


def set_dir(task_set):
    return RESULTS_DIR / task_set


def fig_dir(task_set):
    d = RESULTS_DIR / "figures" / task_set
    d.mkdir(parents=True, exist_ok=True)
    return d


def available_models(task_set):
    probe = canary_conds(task_set)[0]
    return [m for m in MODEL_LABELS
            if (set_dir(task_set) / probe / f"metrics_{m}.json").exists()]


def load_metrics(task_set, sig, model, K):
    if sig in canary_conds(task_set):
        d = json.loads((set_dir(task_set) / sig / f"metrics_{model}.json").read_text())
        return d["metrics_by_K"][K]
    d = json.loads((set_dir(task_set) / "Traditional" / sig /
                    f"metrics_{model}.json").read_text())
    if d.get("signal") == "LLM_judge" and not d.get("n_judged"):
        raise FileNotFoundError(f"LLM_judge not run for {task_set}/{model}")
    if "metrics_by_K" in d:
        return d["metrics_by_K"][K]
    return d["sweep"][str(d["best_threshold_at_primary_K"])][K]


def load_runs(task_set, model, condition):
    return [json.loads(open(f).read()) for f in
            sorted(glob.glob(str(RUNS_DIR / task_set / model / condition / "task_*.json")))]


def headline_canary(task_set, models):
    scored = {}
    for c in canary_conds(task_set):
        if c in ("static_trailer", "ensemble"):
            continue  # rank single-axis canaries; ensemble charted separately
        try:
            scored[c] = statistics.mean(
                load_metrics(task_set, c, m, str(PRIMARY_K))["f1"] for m in models)
        except FileNotFoundError:
            pass
    return max(scored, key=scored.get), scored


# ------------------------------------------------------------------ fig 1

def _bar_panel(task_set, ax, m, metric):
    order = bar_order(task_set)
    n_can = len(canary_conds(task_set))
    ys = [len(order) - 1 - i + (0.9 if i < n_can else 0) for i in range(len(order))]
    for y, sig in zip(ys, order):
        try:
            v = load_metrics(task_set, sig, m, str(PRIMARY_K))[metric]
        except FileNotFoundError:
            ax.annotate("not run", (0.015, y), fontsize=8.5, color=MUTED,
                        va="center", style="italic")
            continue
        if v is None:
            ax.annotate("never fired", (0.015, y), fontsize=8.5, color=MUTED,
                        va="center", style="italic")
            continue
        ax.barh(y, v, height=0.62, color=MODEL_COLORS[m], edgecolor=SURFACE, linewidth=1)
        ax.annotate(f"{v:.2f}", (v + 0.015, y), fontsize=8.5, color=INK2, va="center")
    style(ax, ygrid=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_yticks(ys, order)
    ax.tick_params(axis="y", labelsize=9, labelcolor=INK2)
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.5", "0.75", "1"])
    ax.set_ylim(-0.6, len(order) + 1.25)
    divider = len(TRADITIONAL_SIGNALS) - 0.05
    ax.axhline(divider, color=BASELINE, linewidth=0.8)
    ax.annotate("canaries", xy=(-0.02, len(order) + 0.35),
                xycoords=("axes fraction", "data"), fontsize=8, color=MUTED,
                ha="right", va="bottom", style="italic")
    ax.annotate("traditional", xy=(-0.02, divider - 0.2),
                xycoords=("axes fraction", "data"), fontsize=8, color=MUTED,
                ha="right", va="top", style="italic")
    ax.set_title(MODEL_LABELS[m], fontsize=10, color=INK2, loc="left")


def fig1(task_set, models, n_tasks):
    for metric, fname, head in [
        ("precision", "fig1a_precision.png",
         f"[{task_set}] which axis of strain warns most precisely"),
        ("recall", "fig1b_recall.png",
         f"[{task_set}] which axis of strain catches the most failures"),
    ]:
        fig, axes = plt.subplots(1, len(models), figsize=(9.6, 5.4), squeeze=False)
        axes = axes[0]
        one = len(models) == 1
        fig.subplots_adjust(top=0.80, bottom=0.10, left=0.24 if one else 0.20,
                            right=0.72 if one else 0.985, wspace=0.66)
        for ax, m in zip(axes, models):
            _bar_panel(task_set, ax, m, metric)
        title(fig, head,
              f"{metric.capitalize()} at K={PRIMARY_K} on the {task_set} task set · "
              f"{n_tasks} tasks per model · each canary isolates one cognitive axis")
        fig.savefig(fig_dir(task_set) / fname, dpi=220)
        plt.close(fig)


# ------------------------------------------------------------------ fig 2

def fig2(task_set, models, n_tasks):
    _, scored = headline_canary(task_set, models)
    top2 = sorted(scored, key=lambda c: -scored[c])[:2]
    shown = top2 + TRADITIONAL_SIGNALS          # 6 fixed slots
    colors = dict(zip(shown, PALETTE))
    fig, axes = plt.subplots(1, len(models), figsize=(9.6, 5.2),
                             sharey=True, squeeze=False)
    axes = axes[0]
    one = len(models) == 1
    fig.subplots_adjust(top=0.79, bottom=0.22, left=0.10,
                        right=0.62 if one else 0.98, wspace=0.08)
    xs = range(len(KS))
    for ax, m in zip(axes, models):
        for sig in shown:
            try:
                ys = [load_metrics(task_set, sig, m, K)["recall"] or 0.0 for K in KS]
            except FileNotFoundError:
                continue
            ax.plot(xs, ys, color=colors[sig], linewidth=2, marker="o", markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.annotate(sig, (xs[-1], ys[-1]), xytext=(6, 0),
                        textcoords="offset points", fontsize=7.5,
                        color=colors[sig], va="center")
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
    title(fig, f"[{task_set}] widening the window lifts canary recall",
          f"Recall vs prediction window K · {n_tasks} tasks · top-2 canaries by F1 shown "
          f"with the four traditional signals (all rows in results3/{task_set}/SUMMARY.md)")
    fig.savefig(fig_dir(task_set) / "fig2_recall_vs_window.png", dpi=220)
    plt.close(fig)


# ------------------------------------------------------------------ fig 3

def fig3(task_set, models, cond, n_tasks):
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    fig.subplots_adjust(top=0.82, bottom=0.13, left=0.09, right=0.97)
    max_t = 30
    for m in models:
        runs = load_runs(task_set, m, cond)
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
    handles = [Line2D([], [], color=MODEL_COLORS[m], lw=2, label=MODEL_LABELS[m])
               for m in models] + [
        Line2D([], [], color=INK2, lw=2, ls=(0, (4, 2)), label="first canary failure (signal)"),
        Line2D([], [], color=INK2, lw=2, label="first hallucination (outcome)")]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.5,
              labelcolor=INK2)
    title(fig, f"[{task_set}] where the {cond} sentinel fires, it fires early",
          f"Cumulative onset within the {cond} runs · dashed = signal, solid = outcome ·\n"
          "a lower dashed curve means the canary fires on fewer tasks, not later on them")
    fig.savefig(fig_dir(task_set) / "fig3_onset_curves.png", dpi=220)
    plt.close(fig)


# ------------------------------------------------------------------ fig 4

def fig4(task_set, models, cond):
    fig, axes = plt.subplots(1, len(models), figsize=(9.6, 4.6),
                             sharey=True, squeeze=False)
    axes = axes[0]
    one = len(models) == 1
    fig.subplots_adjust(top=0.78, bottom=0.15, left=0.10,
                        right=0.60 if one else 0.98, wspace=0.08)
    meds = []
    for ax, m in zip(axes, models):
        runs = load_runs(task_set, m, cond)
        leads = [r["first_hallucination"] - r["first_canary_fail"] for r in runs
                 if r["first_hallucination"] is not None
                 and r["first_canary_fail"] is not None
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
        ax.set_xlabel("warning lead time (turns)")
        ax.set_title(f"{MODEL_LABELS[m]}  ·  n={len(leads)} early warnings",
                     fontsize=10, color=INK2, loc="left")
    axes[0].set_ylabel("tasks")
    span = (f"{min(meds):g}–{max(meds):g}" if len(set(meds)) > 1
            else f"{meds[0]:g}" if meds else "?")
    title(fig, f"[{task_set}] the {cond} sentinel buys ~{span} turns of warning",
          f"Lead time from the first {cond} canary failure to the first hallucination "
          "(tasks where the canary fired at or before it)")
    fig.savefig(fig_dir(task_set) / "fig4_lead_time.png", dpi=220)
    plt.close(fig)


def main():
    for task_set in TASK_SETS:
        if not set_dir(task_set).exists():
            print(f"[{task_set}] no results yet -- skipped")
            continue
        models = available_models(task_set)
        if not models:
            print(f"[{task_set}] no scored models -- skipped")
            continue
        n_tasks = len(load_runs(task_set, models[0], "baseline")) or \
            len(load_runs(task_set, models[0], canary_conds(task_set)[0]))
        cond, scored = headline_canary(task_set, models)
        print(f"[{task_set}] models={models} headline={cond} "
              f"F1: { {k: round(v, 3) for k, v in sorted(scored.items(), key=lambda x: -x[1])} }")
        fig1(task_set, models, n_tasks)
        fig2(task_set, models, n_tasks)
        fig3(task_set, models, cond, n_tasks)
        fig4(task_set, models, cond)
        print(f"[{task_set}] wrote",
              *sorted(p.name for p in fig_dir(task_set).glob("*.png")))


if __name__ == "__main__":
    main()
