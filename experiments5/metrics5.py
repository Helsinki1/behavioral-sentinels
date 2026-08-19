"""Experiment-5 analysis: paired comparison of reset policies on task accuracy,
pooled over the mixed pool and broken out per domain.

Primary outcome  : mean per-turn accuracy (share of turns with zero errors)
Primary contrasts: D_routed  vs each traditional trigger (clock, ctx, judge,
                   random) -- the routing claim in deployment terms -- and
                   Z_routed vs the same four (the zero-carry variant).
Routing controls : D_routed vs D_blanket (one probe everywhere, the exp-1..4
                   approach) and vs D_rotated (same probes, mis-assigned).
Decomposition    : C_prime_routed - C_clock  = carrying cost of routed probe
                   D_routed - C_prime_routed = value of its timing
All tests are paired over (domain, task_id) with a bootstrap CI on the delta.
"""
import json
import random
import statistics

from .config5 import (ARMS, DOMAINS, INTENDED_GENRE, N_PER_DOMAIN,
                      RESULTS_DIR, SUCCESS_THRESHOLD)
from .run_all5 import load_arm, load_router, select_tasks

ARM_DISPLAY = ["A_no_reset", "B_random", "C_clock", "C_ctx", "C_judge",
               "C_prime_routed", "D_routed", "D_labeled", "D_blanket",
               "D_rotated", "Z_routed", "F_oracle"]

CONTRAST_PAIRS = [
    ("D_routed", "C_clock", "routed sentinel vs turn-count clock"),
    ("D_routed", "C_ctx", "routed sentinel vs context-growth trigger"),
    ("D_routed", "C_judge", "routed sentinel vs LLM judge"),
    ("D_routed", "B_random", "routed sentinel vs random, budget-matched"),
    ("Z_routed", "C_clock", "ZERO-CARRY routed vs turn-count clock"),
    ("Z_routed", "C_ctx", "zero-carry routed vs context-growth trigger"),
    ("Z_routed", "C_judge", "zero-carry routed vs LLM judge"),
    ("Z_routed", "A_no_reset", "zero-carry routed vs never resetting"),
    ("D_routed", "D_blanket", "ROUTING vs blanket probe (exp-1..4 approach)"),
    ("D_routed", "D_rotated", "ROUTING vs mis-assigned probes (anti-routing)"),
    ("D_labeled", "D_routed", "LABELED routing vs LLM router (router noise cost)"),
    ("D_labeled", "C_clock", "labeled-routed sentinel vs turn-count clock"),
    ("D_labeled", "D_rotated", "labeled routing vs mis-assigned probes"),
    ("C_prime_routed", "C_clock", "carrying cost of the routed probe"),
    ("D_routed", "C_prime_routed", "timing value of the routed probe"),
    ("C_clock", "A_no_reset", "does clock resetting help at all"),
    ("F_oracle", "C_clock", "PERFECT predictor vs clock (headroom)"),
    ("F_oracle", "A_no_reset", "perfect predictor vs no reset"),
]


def pool_ids():
    return {d: [t["task_id"] for t in select_tasks(d)] for d in DOMAINS}


def load_all(model):
    """{arm: {(domain, tid): record}} restricted to complete trajectories."""
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


def boot_ci(deltas, n=10000, seed=7):
    if not deltas:
        return (None, None)
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        s = [deltas[rng.randrange(len(deltas))] for _ in range(len(deltas))]
        means.append(sum(s) / len(s))
    means.sort()
    return (means[int(0.025 * n)], means[int(0.975 * n)])


def arm_stats(recs):
    acc = [r["accuracy"] for r in recs]
    return {
        "n": len(recs),
        "accuracy": round(statistics.mean(acc), 4),
        "success_rate": round(sum(1 for a in acc if a >= SUCCESS_THRESHOLD) / len(acc), 4),
        "resets_per_task": round(statistics.mean(r["n_resets"] for r in recs), 3),
        "prompt_tokens": round(statistics.mean(r["prompt_tokens"] for r in recs)),
        "completion_tokens": round(statistics.mean(r["completion_tokens"] for r in recs)),
    }


def contrast(a_map, b_map, label, keys):
    d = [a_map[k]["accuracy"] - b_map[k]["accuracy"] for k in keys]
    lo, hi = boot_ci(d)
    wins = sum(1 for v in d if v > 1e-9)
    losses = sum(1 for v in d if v < -1e-9)
    return {
        "contrast": label, "n": len(d),
        "mean_delta": round(statistics.mean(d), 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "significant": bool(lo > 0 or hi < 0),
        "tasks_better": wins, "tasks_worse": losses,
        "tasks_tied": len(d) - wins - losses,
    }


def router_report(model):
    ids = pool_ids()
    rows = {}
    for d in DOMAINS:
        router = load_router(model, d, ids[d])
        n = len(router)
        agree = sum(1 for v in router.values()
                    if v.get("genre") == INTENDED_GENRE[d])
        probes = sorted(v["probe"] for v in router.values())
        rows[d] = {"n": n, "agree_intended": agree,
                   "probes": {p: probes.count(p) for p in dict.fromkeys(probes)}}
    return rows


def compute(model="gpt-oss-20b"):
    all_recs = load_all(model)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # common keys across every present arm (paired analysis)
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

    out = {"model": model, "n_tasks": len(common),
           "per_domain_n": {d: len(by_domain[d]) for d in DOMAINS},
           "router": router_report(model), "arms": stats,
           "arms_by_domain": dstats, "contrasts": contrasts,
           "contrasts_by_domain": dcontrasts}
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(out, indent=2))
    write_summary(out)
    return out


def write_summary(out):
    stats, model = out["arms"], out["model"]
    L = ["# Experiment 5 — Task-conditioned sentinel routing, deployed", "",
         f"Model {model} — {out['n_tasks']} tasks paired across every arm "
         f"({', '.join(f'{v} {k}' for k, v in out['per_domain_n'].items())}), "
         "full horizon, no early stop.",
         "Primary outcome: share of turns with zero errors. Compaction = the agent's own",
         "validated structured state snapshot (no ground truth injected), identical in",
         "every resetting arm.", "",
         "## Router",
         ""]
    for d, r in out["router"].items():
        L.append(f"- **{d}**: {r['agree_intended']}/{r['n']} tasks routed to the "
                 f"intended genre; probes assigned: "
                 + ", ".join(f"{p} x{c}" for p, c in r["probes"].items()))
    L += ["", "## Arms", "",
          "| arm | policy | probe | accuracy | success@0.9 | resets/task | prompt tok |",
          "|---|---|---|---|---|---|---|"]
    for a in [x for x in ARM_DISPLAY if x in stats]:
        s = stats[a]
        L.append(f"| {a} | {ARMS[a]['policy']} | {ARMS[a]['probe']} "
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

    L += ["", "## Key contrasts per domain", "",
          "| domain | contrast | delta | 95% CI | sig |", "|---|---|---|---|---|"]
    keep = {("D_routed", "C_clock"), ("D_routed", "D_rotated"),
            ("D_routed", "D_blanket"), ("D_labeled", "D_routed"),
            ("D_labeled", "C_clock"), ("Z_routed", "C_clock"),
            ("Z_routed", "C_judge"), ("C_prime_routed", "C_clock"),
            ("F_oracle", "C_clock")}
    for c in out["contrasts_by_domain"]:
        if (c["a"], c["b"]) in keep:
            L.append(f"| {c['domain']} | {c['a']} - {c['b']} | "
                     f"{c['mean_delta']:+.4f} | "
                     f"[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] | "
                     f"{'**yes**' if c['significant'] else 'no'} |")

    (RESULTS_DIR / "SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L).encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    compute()
