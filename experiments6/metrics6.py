"""Experiment-6 analysis: paired comparison of reset policies when the reset
operator is loss-free re-grounding (R1) or log replay (R2), plus the
cross-experiment operator contrasts against exp 5's compaction arms at a
fixed trigger.

Gate           : F_oracle - A_no_reset  (does restart pay at all when cheap?)
Goal line      : Z_reground vs clock / ctx / judge / random
Timing question: G_dense vs F_oracle vs Z_reground (is placement even needed
                 when resets are cheap, given the same 6-reset cap?)
Operator effect: exp6 arm - exp5 arm at the same trigger, paired on task
Bracket        : replay (R2) versions of Z / clock / oracle
"""
import json
import statistics

from experiments5.metrics5 import arm_stats, boot_ci, contrast
from experiments5.run_all5 import load_arm as load_arm5

from . import config6
from .config6 import ARMS, DOMAINS, SUCCESS_THRESHOLD
from .run_all6 import load_arm, select_tasks

ARM_DISPLAY = ["A_no_reset", "B_random", "C_clock", "C_ctx", "C_judge",
               "Z_reground", "F_oracle", "G_dense",
               "Z_replay", "C_clock_replay", "F_oracle_replay"]

CONTRAST_PAIRS = [
    ("F_oracle", "A_no_reset", "GATE: perfect timing vs never resetting"),
    ("Z_reground", "C_clock", "zero-carry reground vs turn-count clock"),
    ("Z_reground", "C_ctx", "zero-carry reground vs context-growth trigger"),
    ("Z_reground", "C_judge", "zero-carry reground vs LLM judge"),
    ("Z_reground", "B_random", "zero-carry reground vs random, budget-matched"),
    ("Z_reground", "A_no_reset", "zero-carry reground vs never resetting"),
    ("Z_reground", "F_oracle", "zero-carry reground vs perfect timing"),
    ("G_dense", "A_no_reset", "densest schedule vs never resetting"),
    ("G_dense", "F_oracle", "densest schedule vs perfect timing"),
    ("Z_reground", "G_dense", "sentinel placement vs densest schedule"),
    ("C_clock", "A_no_reset", "clock reground vs never resetting"),
    ("F_oracle", "C_clock", "perfect timing vs clock"),
    ("Z_replay", "Z_reground", "replay (R2) vs reground (R1), zero-carry"),
    ("C_clock_replay", "C_clock", "replay vs reground, clock"),
    ("F_oracle_replay", "F_oracle", "replay vs reground, oracle"),
    ("Z_replay", "A_no_reset", "zero-carry REPLAY vs never resetting"),
    ("Z_replay", "C_clock_replay", "zero-carry vs clock, both replay"),
]

# exp6 arm -> exp5 arm with the same trigger: the operator effect
CROSS_EXPERIMENT = [
    ("Z_reground", "Z_routed", "zero-carry trigger: reground vs compaction"),
    ("C_clock", "C_clock", "clock trigger: reground vs compaction"),
    ("C_ctx", "C_ctx", "ctx trigger: reground vs compaction"),
    ("C_judge", "C_judge", "judge trigger: reground vs compaction"),
    ("F_oracle", "F_oracle", "oracle trigger: reground vs compaction"),
]

KEEP_BY_DOMAIN = {("F_oracle", "A_no_reset"), ("Z_reground", "C_clock"),
                  ("Z_reground", "C_ctx"), ("Z_reground", "A_no_reset"),
                  ("G_dense", "A_no_reset"), ("Z_reground", "G_dense"),
                  ("C_clock", "A_no_reset"), ("Z_replay", "Z_reground"),
                  ("Z_reground", "C_judge")}


def pool_ids():
    return {d: [t["task_id"] for t in select_tasks(d)] for d in DOMAINS}


def load_all(model):
    ids = pool_ids()
    out = {}
    for arm in ARMS:
        recs = {}
        for d in DOMAINS:
            for tid, r in load_arm(model, d, arm, ids[d]).items():
                recs[(d, tid)] = r
        if recs:
            out[arm] = recs
    return out


def load_exp5(model):
    ids = pool_ids()
    arms5 = sorted({b for _, b, _ in CROSS_EXPERIMENT})
    out = {}
    for arm in arms5:
        recs = {}
        for d in DOMAINS:
            for tid, r in load_arm5(model, d, arm, ids[d]).items():
                recs[(d, tid)] = r
        if recs:
            out[arm] = recs
    return out


def compute(model="gpt-oss-20b"):
    all_recs = load_all(model)
    exp5 = load_exp5(model)
    config6.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    common = sorted(set.intersection(*[set(v) for v in all_recs.values()]))
    by_domain = {d: [k for k in common if k[0] == d] for d in DOMAINS}

    stats, dstats = {}, {}
    for arm, recs in all_recs.items():
        stats[arm] = arm_stats([recs[k] for k in common])
        dstats[arm] = {d: arm_stats([recs[k] for k in by_domain[d]])
                       for d in DOMAINS if by_domain[d]}

    contrasts, dcontrasts = [], []
    for a, b, lab in CONTRAST_PAIRS:
        if a in all_recs and b in all_recs:
            contrasts.append(contrast(all_recs[a], all_recs[b],
                                      f"{a} - {b}: {lab}", common))
            for d in DOMAINS:
                if by_domain[d]:
                    c = contrast(all_recs[a], all_recs[b], lab, by_domain[d])
                    dcontrasts.append({"domain": d, "a": a, "b": b, **c})

    cross = []
    for a6, a5, lab in CROSS_EXPERIMENT:
        if a6 in all_recs and a5 in exp5:
            keys = sorted(set(all_recs[a6]) & set(exp5[a5]) & set(common))
            if keys:
                c = contrast(all_recs[a6], exp5[a5], f"{a6}(6) - {a5}(5): {lab}",
                             keys)
                cross.append(c)
                for d in DOMAINS:
                    dk = [k for k in keys if k[0] == d]
                    if dk:
                        cd = contrast(all_recs[a6], exp5[a5], lab, dk)
                        cross.append({"domain": d, **cd})

    out = {"model": model, "n_tasks": len(common),
           "per_domain_n": {d: len(by_domain[d]) for d in DOMAINS},
           "arms": stats, "arms_by_domain": dstats,
           "contrasts": contrasts, "contrasts_by_domain": dcontrasts,
           "cross_experiment": cross}
    (config6.RESULTS_DIR / "metrics.json").write_text(json.dumps(out, indent=2))
    write_summary(out)
    return out


def write_summary(out):
    stats, model = out["arms"], out["model"]
    L = ["# Experiment 6 — Sentinel-triggered re-grounding, deployed", "",
         f"Model {model} — {out['n_tasks']} tasks paired across every arm "
         f"({', '.join(f'{v} {k}' for k, v in out['per_domain_n'].items())}), "
         "full horizon, no early stop.",
         "Reset operator: deterministic re-grounding from the external store (R1)",
         "or verbatim user-log replay (R2). No LLM call at reset time; no probe",
         "carried in any arm. A_no_reset imported verbatim from runs5.", "",
         "## Arms", "",
         "| arm | policy | operator | accuracy | success@0.9 | resets/task | prompt tok |",
         "|---|---|---|---|---|---|---|"]
    for a in [x for x in ARM_DISPLAY if x in stats]:
        s = stats[a]
        L.append(f"| {a} | {ARMS[a]['policy']} | {ARMS[a]['operator'] or '—'} "
                 f"| {s['accuracy']:.3f} | {s['success_rate']:.3f} "
                 f"| {s['resets_per_task']:.2f} | {s['prompt_tokens']:,} |")

    L += ["", "## Per-domain accuracy", "",
          "| arm | " + " | ".join(DOMAINS) + " |",
          "|---|" + "---|" * len(DOMAINS)]
    for a in [x for x in ARM_DISPLAY if x in out["arms_by_domain"]]:
        cells = []
        for d in DOMAINS:
            s = out["arms_by_domain"][a].get(d)
            cells.append(f"{s['accuracy']:.3f}" if s else "—")
        L.append(f"| {a} | " + " | ".join(cells) + " |")

    L += ["", "## Paired contrasts, pooled (bootstrap 95% CI on the per-task delta)",
          "",
          "| contrast | delta accuracy | 95% CI | significant | better/worse/tied |",
          "|---|---|---|---|---|"]
    for c in out["contrasts"]:
        L.append(f"| {c['contrast']} | {c['mean_delta']:+.4f} | "
                 f"[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] | "
                 f"{'**yes**' if c['significant'] else 'no'} | "
                 f"{c['tasks_better']}/{c['tasks_worse']}/{c['tasks_tied']} |")

    L += ["", "## Operator effect: same trigger, exp-6 operator minus exp-5 compaction",
          "", "(paired on task across experiments; positive = re-grounding beats",
          "compaction at that trigger)", "",
          "| contrast | delta | 95% CI | sig |", "|---|---|---|---|"]
    for c in out["cross_experiment"]:
        name = (f"{c['domain']}: {c['contrast']}" if "domain" in c
                else f"**pooled**: {c['contrast']}")
        L.append(f"| {name} | {c['mean_delta']:+.4f} | "
                 f"[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] | "
                 f"{'**yes**' if c['significant'] else 'no'} |")

    L += ["", "## Key contrasts per domain", "",
          "| domain | contrast | delta | 95% CI | sig |", "|---|---|---|---|---|"]
    for c in out["contrasts_by_domain"]:
        if (c["a"], c["b"]) in KEEP_BY_DOMAIN:
            L.append(f"| {c['domain']} | {c['a']} - {c['b']} | "
                     f"{c['mean_delta']:+.4f} | "
                     f"[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] | "
                     f"{'**yes**' if c['significant'] else 'no'} |")

    (config6.RESULTS_DIR / "SUMMARY.md").write_text("\n".join(L) + "\n",
                                                    encoding="utf-8")
    print("\n".join(L).encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    compute()
