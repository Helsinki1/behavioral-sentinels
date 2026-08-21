"""Experiment 10, Study B0 — the carrying-cost x timing 2x2.

Experiment 4 estimated "even a perfect carried probe barely pays" by
subtracting the measured carrying cost from the measured timing prize. That
assumes the two effects are ADDITIVE, which was never measured. This completes
the factorial so the interaction is estimated rather than assumed:

                     no useful reset        oracle-timed reset
    no sentinel      A_no_reset      (A)    F_oracle        (B)
    carries probe    P_carry_noreset (C)    P_carry_oracle  (D)

    C - A   carrying cost
    B - A   maximum timing value
    D - C   timing value WHILE carrying the probe        <- previously unknown
    D - A   can a perfect carried sentinel pay at all?   <- the decision
    (D-C)-(B-A)  the interaction; zero means additive
"""
import json
import statistics

from experiments4.config4 import FACTORIAL, RESULTS_DIR as R4
from experiments4.metrics4 import boot_ci
from experiments4.run_all4 import load_arm
from .config10 import RESULTS_DIR


def compute(model="gpt-oss-20b", n=40):
    ids = list(range(n))
    arms = {k: load_arm(model, v, ids) for k, v in FACTORIAL.items()}
    common = sorted(set.intersection(*[set(v) for v in arms.values()]))
    acc = {k: {t: arms[k][t]["accuracy"] for t in common} for k in arms}

    def delta(x, y, label):
        d = [acc[x][t] - acc[y][t] for t in common]
        lo, hi = boot_ci(d)
        return {"contrast": f"{x} - {y}", "meaning": label,
                "mean": round(statistics.mean(d), 4),
                "ci95": [round(lo, 4), round(hi, 4)],
                "significant": bool(lo > 0 or hi < 0)}

    cells = {k: round(statistics.mean(acc[k].values()), 4) for k in arms}
    res = [delta("C", "A", "carrying cost (no reset in either)"),
           delta("B", "A", "maximum timing value (no probe in either)"),
           delta("D", "C", "timing value WHILE carrying the probe"),
           delta("D", "A", "can a PERFECT CARRIED sentinel pay for itself?"),
           delta("D", "B", "cost of carrying, given oracle timing")]

    # interaction: (D-C) - (B-A), paired per task
    inter = [(acc["D"][t] - acc["C"][t]) - (acc["B"][t] - acc["A"][t]) for t in common]
    lo, hi = boot_ci(inter)
    interaction = {"contrast": "(D-C) - (B-A)", "meaning": "interaction; 0 = additive",
                   "mean": round(statistics.mean(inter), 4),
                   "ci95": [round(lo, 4), round(hi, 4)],
                   "significant": bool(lo > 0 or hi < 0)}

    additive_pred = round(cells["A"] + (cells["C"] - cells["A"]) + (cells["B"] - cells["A"]), 4)
    out = {"model": model, "n_tasks": len(common), "cells": cells,
           "contrasts": res, "interaction": interaction,
           "additive_prediction_for_D": additive_pred,
           "observed_D": cells["D"],
           "additivity_error": round(cells["D"] - additive_pred, 4)}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "factorial.json").write_text(json.dumps(out, indent=2))

    L = ["# Experiment 10, Study B0 — does a perfect carried sentinel pay?", "",
         f"Model {model}, {len(common)} coding tasks paired across all four cells,",
         "experiment-4 regime (self-summary compaction — the operator under which a",
         "timing prize exists at all). Primary outcome: share of turns with zero errors.",
         "", "## The 2x2", "",
         "| | no useful reset | oracle-timed reset |", "|---|---|---|",
         f"| **no sentinel** | A = {cells['A']:.3f} | B = {cells['B']:.3f} |",
         f"| **carries probe** | C = {cells['C']:.3f} | D = {cells['D']:.3f} |", "",
         "## Contrasts (paired, bootstrap 95% CI)", "",
         "| contrast | meaning | delta | 95% CI | significant |", "|---|---|---|---|---|"]
    for c in res + [interaction]:
        L.append(f"| {c['contrast']} | {c['meaning']} | {c['mean']:+.4f} | "
                 f"[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] | "
                 f"{'**yes**' if c['significant'] else 'no'} |")
    L += ["", "## Additivity check", "",
          f"- additive prediction for D (A + carrying cost + timing prize): **{additive_pred:.4f}**",
          f"- observed D: **{cells['D']:.4f}**",
          f"- error: **{out['additivity_error']:+.4f}**", ""]
    (RESULTS_DIR / "STUDY_B0.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return out


if __name__ == "__main__":
    compute()
