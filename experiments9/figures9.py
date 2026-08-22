"""Experiment-9 figures -- one idea per figure, deliberately few dimensions each.

fig1  observer effect by model: ACT_carry_clock - C_clock with 95% CI
      (plus exp 8's gpt-oss-20b for the cross-experiment sign flip)
fig2  is the model "lost"?  A_no_reset accuracy and the clock / oracle lift
fig3  accuracy per arm, one panel per model, shared axis
fig4  success@0.9 per arm, one panel per model, shared axis
fig5  forest plot of every paired contrast, one panel per model
fig6  signal quality: precision vs recall of every behavioural signal
fig7  resets per task, per arm and model
fig8  token cost per arm and model (agent prompt + quiz fork)
fig9  accuracy vs total token spend, one panel per model

Run:  python -m experiments9.figures9
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .config9 import RESULTS_DIR, ROOT

FIG_DIR = RESULTS_DIR / "figures"
EXP8_METRICS = ROOT / "results8" / "metrics.json"

# validated categorical palette (dataviz reference instance, light surface)
BLUE, ORANGE, AQUA, YELLOW, VIOLET, RED = ("#2a78d6", "#eb6834", "#1baf7a",
                                           "#eda100", "#4a3aa7", "#e34948")
GRAY, DARK, MUTED, GRID = "#9a9892", "#3d3c39", "#6b6a66", "#e6e5e1"

CAT_COLORS = {"active": ORANGE, "passive-behavioural": AQUA,
              "passive-observational": BLUE, "baseline": GRAY, "bound": DARK}

MODELS = ["gpt-oss-120b", "qwen3p7-plus", "deepseek-v4-flash", "gpt-4o-mini"]
MODEL_LABEL = {"gpt-oss-120b": "gpt-oss-120b", "qwen3p7-plus": "qwen3.7-plus",
               "deepseek-v4-flash": "deepseek-v4-flash",
               "gpt-4o-mini": "gpt-4o-mini"}

# arm display order: bounds & baseline first, then the three observation kinds
ARMS = [("A_no_reset", "never reset", "bound"),
        ("C_clock", "clock", "baseline"),
        ("F_oracle", "oracle", "bound"),
        ("Z_trace", "trace monitor", "passive-observational"),
        ("QUIZ", "frozen-state quiz", "passive-behavioural"),
        ("ACT_carry_clock", "carried probe, clock schedule", "active"),
        ("ACT_probe", "carried probe, probe-triggered", "active")]
ARM_LABEL = {a: d for a, d, _ in ARMS}
ARM_CAT = {a: c for a, _, c in ARMS}

CONTRAST_ORDER = [  # short label, prefix used to match metrics.json entries
    ("clock − never reset", "C_clock - A_no_reset"),
    ("oracle − never reset", "F_oracle - A_no_reset"),
    ("trace − never reset", "Z_trace - A_no_reset"),
    ("trace − clock", "Z_trace - C_clock"),
    ("quiz − never reset", "QUIZ - A_no_reset"),
    ("quiz − clock", "QUIZ - C_clock"),
    ("quiz − trace", "QUIZ - Z_trace"),
    ("quiz − oracle", "QUIZ - F_oracle"),
    ("carry-clock − clock  (observer effect)", "ACT_carry_clock - C_clock"),
    ("probe − carry-clock  (timing value)", "ACT_probe - ACT_carry_clock"),
    ("probe − clock", "ACT_probe - C_clock"),
    ("probe − quiz", "ACT_probe - QUIZ"),
    ("probe − never reset", "ACT_probe - A_no_reset"),
]


def style(ax, ygrid=True, xgrid=False):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRAY)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    if xgrid:
        ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def find_contrast(m, model, prefix):
    for c in m["models"][model]["contrasts"]:
        if c["contrast"].startswith(prefix):
            return c
    raise KeyError(prefix)


def cat_legend(ax, cats, **kw):
    handles = [Patch(facecolor=CAT_COLORS[c], label=c) for c in cats]
    ax.legend(handles=handles, frameon=False, fontsize=8, **kw)


def fig_legend(fig, handles, ncol=None):
    fig.legend(handles=handles, frameon=False, fontsize=8.5,
               ncol=ncol or len(handles), loc="lower center",
               bbox_to_anchor=(0.5, -0.01))


CAT_HANDLES = [Patch(facecolor=CAT_COLORS[c], label=c) for c in
               ("active", "passive-behavioural", "passive-observational",
                "baseline", "bound")]


def save(fig, name):
    fig.savefig(FIG_DIR / name, dpi=170, facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------
def fig1(m):
    """Observer effect per model: carrying the probe at the clock's schedule."""
    rows = []
    if EXP8_METRICS.exists():
        m8 = json.loads(EXP8_METRICS.read_text())
        c8 = next(c for c in m8["contrasts"] if "OBSERVER" in c["contrast"])
        rows.append(("gpt-oss-20b\n(exp 8, synthetic pool)", c8))
    for model in MODELS:
        rows.append((MODEL_LABEL[model] + "\n(sharded math)",
                     find_contrast(m, model, "ACT_carry_clock - C_clock")))

    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    ys = list(range(len(rows)))[::-1]
    for y, (label, c) in zip(ys, rows):
        lo, hi = c["ci95"]
        sig = c["significant"]
        color = (RED if c["mean_delta"] < 0 else BLUE) if sig else GRAY
        ax.plot([lo, hi], [y, y], color=color, linewidth=2, solid_capstyle="round")
        ax.plot(c["mean_delta"], y, "o", color=color, markersize=9,
                markeredgecolor="white", markeredgewidth=1.2)
        ax.annotate(f"{c['mean_delta']:+.3f}" + ("" if sig else "  n.s."),
                    (hi, y), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=DARK)
    ax.axvline(0, color=DARK, linewidth=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlim(-0.075, 0.055)
    ax.set_xlabel("accuracy change from carrying the lag_span probe\n"
                  "(ACT_carry_clock − C_clock, identical reset schedule; 95% CI)",
                  fontsize=9)
    ax.text(-0.072, ys[0] + 0.7, "◀ probe is a tax", color=RED, fontsize=8.5,
            va="bottom")
    ax.text(0.052, ys[0] + 0.7, "probe is a treatment ▶", color=BLUE,
            fontsize=8.5, va="bottom", ha="right")
    ax.set_ylim(-0.7, len(rows) - 0.3 + 0.9)
    ax.set_title("Fig 1 — the observer effect changes sign with the model",
                 fontsize=11, loc="left")
    style(ax, ygrid=False, xgrid=True)
    ax.legend(handles=[Line2D([], [], color=GRAY, marker="o", linewidth=2,
                              label="CI crosses zero")],
              frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, "fig1_observer_effect.png")


# --------------------------------------------------------------------------
def fig2(m):
    """Is the model in a degradation regime at all?"""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6),
                             gridspec_kw={"width_ratios": [1, 1.35]})
    ys = list(range(len(MODELS)))[::-1]

    ax = axes[0]
    for y, model in zip(ys, MODELS):
        acc = m["models"][model]["arms"]["A_no_reset"]["accuracy"]
        ax.plot(acc, y, "o", color=DARK, markersize=10, zorder=3)
        ax.annotate(f"{acc:.3f}", (acc, y), xytext=(0, 9),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=8.5, color=MUTED)
    ax.set_yticks(ys)
    ax.set_yticklabels([MODEL_LABEL[x] for x in MODELS], fontsize=9)
    ax.set_xlim(0.80, 1.0)
    ax.set_ylim(-0.6, len(MODELS) - 0.3)
    ax.set_xlabel("per-turn accuracy, never resetting", fontsize=9)
    ax.set_title("a) baseline accuracy", fontsize=10, loc="left")
    style(ax, ygrid=False, xgrid=True)

    ax = axes[1]
    off = {"C_clock - A_no_reset": (+0.16, GRAY, "clock restarts"),
           "F_oracle - A_no_reset": (-0.16, DARK, "oracle restarts")}
    for y, model in zip(ys, MODELS):
        for prefix, (dy, color, _) in off.items():
            c = find_contrast(m, model, prefix)
            lo, hi = c["ci95"]
            mk = "o" if c["significant"] else "o"
            face = color if c["significant"] else "white"
            ax.plot([lo, hi], [y + dy] * 2, color=color, linewidth=1.8)
            ax.plot(c["mean_delta"], y + dy, mk, color=color, markersize=7,
                    markerfacecolor=face, markeredgewidth=1.6)
    ax.axvline(0, color=DARK, linewidth=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([])
    ax.set_xlabel("accuracy gain from restarting vs never resetting (95% CI)",
                  fontsize=9)
    ax.set_title("b) does restarting help?  (filled = significant)",
                 fontsize=10, loc="left")
    style(ax, ygrid=False, xgrid=True)
    fig_legend(fig, [Line2D([], [], color=c, marker="o", linewidth=1.8,
                            label=l) for _, (_, c, l) in off.items()])
    fig.suptitle("Fig 2 — only gpt-oss-120b is actually \"lost\" under verbatim "
                 "shards", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig2_degradation_regime.png")


# --------------------------------------------------------------------------
def _per_arm_panels(m, key, xlabel, title, fname, xlim=None, fmt="{:.3f}",
                    ref_arm="C_clock"):
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.8), sharex=True, sharey=True)
    ys = list(range(len(ARMS)))[::-1]
    for ax, model in zip(axes, MODELS):
        arms = m["models"][model]["arms"]
        ref = arms[ref_arm][key]
        ax.axvline(ref, color=GRAY, linewidth=1, linestyle="--", zorder=1)
        for y, (arm, _, cat) in zip(ys, ARMS):
            v = arms[arm][key]
            ax.plot(v, y, "o", color=CAT_COLORS[cat], markersize=9,
                    markeredgecolor="white", markeredgewidth=1.2, zorder=3)
            ax.annotate(fmt.format(v), (v, y), xytext=(0, 7),
                        textcoords="offset points", ha="center",
                        fontsize=7.5, color=MUTED)
        ax.set_title(f"{MODEL_LABEL[model]}  (n={m['models'][model]['n_tasks']})",
                     fontsize=9.5, loc="left")
        ax.set_xlabel(xlabel, fontsize=8.5)
        style(ax, ygrid=False, xgrid=True)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([ARM_LABEL[a] for a, _, _ in ARMS], fontsize=8.5)
    if xlim:
        axes[0].set_xlim(*xlim)
    axes[0].set_ylim(-0.6, len(ARMS) - 0.2)
    fig.suptitle(title + "   (dashed line = clock baseline)", fontsize=11,
                 x=0.01, ha="left")
    fig_legend(fig, CAT_HANDLES)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, fname)


def fig3(m):
    _per_arm_panels(m, "accuracy", "per-turn accuracy",
                    "Fig 3 — accuracy of every arm, per model",
                    "fig3_accuracy_by_arm.png", xlim=(0.80, 1.0))


def fig4(m):
    _per_arm_panels(m, "success_rate", "fraction of sessions ≥ 0.9 accuracy",
                    "Fig 4 — success@0.9 of every arm, per model",
                    "fig4_success_by_arm.png", xlim=(0.0, 1.05), fmt="{:.2f}")


# --------------------------------------------------------------------------
def fig5(m):
    """Forest plot of every paired contrast."""
    fig, axes = plt.subplots(1, 4, figsize=(13, 5.2), sharex=True, sharey=True)
    ys = list(range(len(CONTRAST_ORDER)))[::-1]
    for ax, model in zip(axes, MODELS):
        for y, (label, prefix) in zip(ys, CONTRAST_ORDER):
            c = find_contrast(m, model, prefix)
            lo, hi = c["ci95"]
            sig = c["significant"]
            color = (BLUE if c["mean_delta"] > 0 else RED) if sig else GRAY
            ax.plot([lo, hi], [y, y], color=color, linewidth=1.8)
            ax.plot(c["mean_delta"], y, "o", color=color, markersize=6.5,
                    markerfacecolor=color if sig else "white",
                    markeredgewidth=1.5)
        ax.axvline(0, color=DARK, linewidth=1)
        # separators between contrast families
        for yy in (ys[3] - 0.5, ys[7] - 0.5):
            ax.axhline(yy, color=GRID, linewidth=1)
        ax.set_title(MODEL_LABEL[model], fontsize=9.5, loc="left")
        ax.set_xlabel("Δ accuracy (95% CI)", fontsize=8.5)
        style(ax, ygrid=False, xgrid=True)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([l for l, _ in CONTRAST_ORDER], fontsize=8.5)
    axes[0].set_xlim(-0.085, 0.125)
    fig.suptitle("Fig 5 — every paired contrast, per model   "
                 "(filled = significant; blue = first arm better, red = worse)",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "fig5_contrasts_forest.png")


# --------------------------------------------------------------------------
def fig6(pred):
    """Precision vs recall of behavioural signals, clean reads only."""
    SIGNALS = [  # (display, set, signal key, category, marker)
        ("trace monitor", "A_no_reset", "zero-carry trace monitor",
         "passive-observational", "o"),
        ("quiz, fail ≥ 1", None, "fail>=1", "passive-behavioural", "s"),
        ("quiz, fail ≥ 2", None, "fail>=2", "passive-behavioural", "D"),
        ("carried probe (clean read)", "ACT_carry_clock",
         "carried probe (clock-truncated read)", "active", "^"),
        ("turn-count clock (tuned)", "A_no_reset", "turn_number", "baseline",
         "x"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.7), sharex=True, sharey=True)
    for ax, model in zip(axes, MODELS):
        pm = pred["models"][model]
        for disp, s, key, cat, mk in SIGNALS:
            if s is None:
                sig = pm.get("shadow", {}).get(key)
            else:
                sig = pm["sets"][s]["signals"].get(key)
            if not sig:
                continue
            p, r = sig.get("precision"), sig.get("recall")
            if p is None or r is None:
                continue  # never fired / nothing to catch -> undefined
            ax.plot(r, p, mk, color=CAT_COLORS[cat],
                    markersize={"o": 12, "s": 9, "D": 6.5}.get(mk, 9),
                    markeredgecolor="white" if mk != "x" else CAT_COLORS[cat],
                    markeredgewidth=1.2, zorder=3)
        ax.set_title(MODEL_LABEL[model], fontsize=9.5, loc="left")
        ax.set_xlabel("recall", fontsize=8.5)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.08)
        style(ax, ygrid=True, xgrid=True)
    axes[0].set_ylabel("precision", fontsize=8.5)
    handles = [Line2D([], [], marker=mk, color=CAT_COLORS[cat], linestyle="",
                      markersize=8, label=disp)
               for disp, _, _, cat, mk in SIGNALS]
    fig_legend(fig, handles)
    fig.suptitle("Fig 6 — signal quality: predicting the first hallucination "
                 "(K=5).  Upper-left = precise but blind; "
                 "signals that never fired are omitted.",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig6_signal_precision_recall.png")


# --------------------------------------------------------------------------
def fig7(m):
    _per_arm_panels(m, "resets_per_task", "resets per session",
                    "Fig 7 — how often each arm restarts",
                    "fig7_resets_by_arm.png", xlim=(-0.3, 6.6), fmt="{:.2f}")


# --------------------------------------------------------------------------
def fig8(m):
    """Token cost per arm: stacked agent-prompt + quiz-fork tokens."""
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.8), sharex=True, sharey=True)
    ys = list(range(len(ARMS)))[::-1]
    for ax, model in zip(axes, MODELS):
        arms = m["models"][model]["arms"]
        for y, (arm, _, cat) in zip(ys, ARMS):
            a = arms[arm]
            agent = (a["prompt_tokens"] + a["completion_tokens"]) / 1000
            quiz = a["quiz_tokens"] / 1000
            ax.barh(y, agent, color=CAT_COLORS[cat], height=0.6)
            if quiz:
                ax.barh(y, quiz, left=agent + 0.08, color=CAT_COLORS[cat],
                        height=0.6, alpha=0.45, hatch="///",
                        edgecolor="white", linewidth=0)
            ax.annotate(f"{agent + quiz:.1f}k", (agent + quiz, y),
                        xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=7.5, color=MUTED)
        ax.set_title(MODEL_LABEL[model], fontsize=9.5, loc="left")
        ax.set_xlabel("tokens per session (thousands)", fontsize=8.5)
        style(ax, ygrid=False, xgrid=True)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([ARM_LABEL[a] for a, _, _ in ARMS], fontsize=8.5)
    axes[0].set_xlim(0, 17)
    handles = [Patch(facecolor=DARK, label="agent prompt + completion"),
               Patch(facecolor=DARK, alpha=0.45, hatch="///",
                     edgecolor="white", label="quiz fork (monitoring)")]
    fig_legend(fig, handles)
    fig.suptitle("Fig 8 — token cost of every arm, per model", fontsize=11,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig8_tokens_by_arm.png")


# --------------------------------------------------------------------------
def fig9(m):
    """Accuracy vs total spend; the decision plane, one panel per model."""
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.8), sharey=True,
                             sharex=True)
    for i, (ax, model) in enumerate(zip(axes, MODELS)):
        arms = m["models"][model]["arms"]
        pts = []
        for arm, disp, cat in ARMS:
            a = arms[arm]
            tok = (a["prompt_tokens"] + a["completion_tokens"]
                   + a["quiz_tokens"]) / 1000
            pts.append((tok, a["accuracy"], arm, cat))
            ax.plot(tok, a["accuracy"], "o", color=CAT_COLORS[cat],
                    markersize=9, markeredgecolor="white", markeredgewidth=1.2,
                    zorder=3)
            if i == 0:  # direct labels only where the points are spread out
                ax.annotate(disp.split(",")[0] if "carried" not in disp else
                            ("carry-clock" if "clock" in disp else "probe"),
                            (tok, a["accuracy"]),
                            xytext=(6, -9 if arm == "F_oracle" else 3),
                            textcoords="offset points", fontsize=7.5,
                            color=MUTED)
        # Pareto front: cheapest arm first, keep points that raise accuracy
        front, best = [], -1
        for tok, acc, _, _ in sorted(pts):
            if acc > best:
                front.append((tok, acc))
                best = acc
        ax.plot(*zip(*front), color=GRAY, linewidth=1, zorder=1,
                drawstyle="steps-post")
        ax.set_title(MODEL_LABEL[model], fontsize=9.5, loc="left")
        ax.set_xlabel("total tokens per session (thousands)", fontsize=8.5)
        style(ax, ygrid=True, xgrid=True)
    axes[0].set_ylabel("per-turn accuracy", fontsize=8.5)
    axes[0].set_ylim(0.82, 1.0)
    fig_legend(fig, CAT_HANDLES)
    fig.suptitle("Fig 9 — the decision plane: accuracy vs total spend "
                 "(step line = Pareto front)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig9_decision_plane.png")


def main():
    m = json.loads((RESULTS_DIR / "metrics.json").read_text())
    pred = json.loads((RESULTS_DIR / "prediction.json").read_text())
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for f in (fig1, fig2, fig3, fig4, fig5, fig7, fig8, fig9):
        f(m)
    fig6(pred)
    print("figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
