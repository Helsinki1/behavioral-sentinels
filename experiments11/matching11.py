"""Experiment 11 — matched vs mismatched probes.

Analysis is exactly the plan fixed in README_EXPERIMENT11.md before the
mismatched arms were run. Primary statistic is the domain x probe interaction
(P2), which is free of per-domain baseline differences. Everything else is
secondary and labelled as such.

    Delta_carry(probe, domain) = accuracy(probe arm) - accuracy(C_clock)
    paired by task, identical policy / operator / schedule / pool / model.
"""
import json
import statistics
from pathlib import Path

from experiments4.metrics4 import boot_ci

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results11"
DOMAINS = ["coding", "babi"]                      # registers excluded by screen
MATCHED = {"coding": "staircase", "babi": "lag_span"}
MISMATCHED = {"coding": "lag_span", "babi": "staircase"}


def load(run_root, domain, arm):
    out = {}
    d = ROOT / run_root / "gpt-oss-20b" / domain / arm
    for f in sorted(d.glob("task_*.json")):
        r = json.loads(f.read_text())
        if r.get("complete"):
            out[r["task_id"]] = r["accuracy"]
    return out


def compute():
    RESULTS.mkdir(parents=True, exist_ok=True)
    per_domain, deltas = {}, {}
    for dom in DOMAINS:
        ctrl = load("runs6", dom, "C_clock")          # no probe
        matched = load("runs8", dom, "ACT_carry_clock")
        mismatched = load("runs8", dom, "MM_carry_clock")
        common = sorted(set(ctrl) & set(matched) & set(mismatched))
        dm = [matched[t] - ctrl[t] for t in common]
        dx = [mismatched[t] - ctrl[t] for t in common]
        diff = [a - b for a, b in zip(dm, dx)]        # matched - mismatched
        # P2 is defined in the pre-registration by PROBE IDENTITY
        # (staircase - lag_span) in each domain. The matched probe differs
        # between domains, so that quantity is +(matched-mismatched) in coding
        # and -(matched-mismatched) in babi. Keep both, and build P2 from the
        # probe-identity form so it matches what was pre-registered.
        by_identity = diff if MATCHED[dom] == "staircase" else [-v for v in diff]
        mlo, mhi = boot_ci(dm); xlo, xhi = boot_ci(dx); dlo, dhi = boot_ci(diff)
        per_domain[dom] = {
            "n": len(common),
            "control_acc": round(statistics.mean(ctrl[t] for t in common), 4),
            "matched_probe": MATCHED[dom], "mismatched_probe": MISMATCHED[dom],
            "matched_acc": round(statistics.mean(matched[t] for t in common), 4),
            "mismatched_acc": round(statistics.mean(mismatched[t] for t in common), 4),
            "delta_matched": {"mean": round(statistics.mean(dm), 4),
                              "ci95": [round(mlo, 4), round(mhi, 4)],
                              "significant": bool(mlo > 0 or mhi < 0)},
            "delta_mismatched": {"mean": round(statistics.mean(dx), 4),
                                 "ci95": [round(xlo, 4), round(xhi, 4)],
                                 "significant": bool(xlo > 0 or xhi < 0)},
            "matched_minus_mismatched": {"mean": round(statistics.mean(diff), 4),
                                         "ci95": [round(dlo, 4), round(dhi, 4)],
                                         "significant": bool(dlo > 0 or dhi < 0)},
        }
        deltas[dom] = by_identity

    # P2: the pre-registered primary statistic. Unpaired across domains (they
    # are different tasks), so bootstrap the difference of the two means.
    import random
    rng = random.Random(11)
    # P2 = [stair-lag]_coding - [stair-lag]_babi   (pre-registered form)
    a, b = deltas["coding"], deltas["babi"]
    boots = []
    for _ in range(10000):
        sa = [a[rng.randrange(len(a))] for _ in a]
        sb = [b[rng.randrange(len(b))] for _ in b]
        boots.append(sum(sa) / len(sa) - sum(sb) / len(sb))
    boots.sort()
    p2 = {"mean": round(statistics.mean(a) - statistics.mean(b), 4),
          "ci95": [round(boots[250], 4), round(boots[9750], 4)],
          "significant": bool(boots[250] > 0 or boots[9750] < 0)}

    out = {"per_domain": per_domain, "P2_interaction": p2}
    (RESULTS / "matching.json").write_text(json.dumps(out, indent=2))

    L = ["# Experiment 11 — matched vs mismatched probes (results)", "",
         "Pre-registered in README_EXPERIMENT11.md before these runs. gpt-oss-20b,",
         "R1 re-grounding, clock reset every 6 turns, full horizon. Identical policy,",
         "operator, schedule and pool in every cell; only the carried chore varies.",
         "`registers` excluded by the degradation screen.", "",
         "| domain | no probe | matched probe | Δ matched | mismatched probe | Δ mismatched | matched − mismatched |",
         "|---|---|---|---|---|---|---|"]
    for dom in DOMAINS:
        d = per_domain[dom]
        dm, dx, mm = d["delta_matched"], d["delta_mismatched"], d["matched_minus_mismatched"]
        L.append(
            f"| {dom} (n={d['n']}) | {d['control_acc']:.3f} | "
            f"{d['matched_probe']} {d['matched_acc']:.3f} | "
            f"{dm['mean']:+.4f} [{dm['ci95'][0]:+.3f},{dm['ci95'][1]:+.3f}]"
            f"{' **sig**' if dm['significant'] else ''} | "
            f"{d['mismatched_probe']} {d['mismatched_acc']:.3f} | "
            f"{dx['mean']:+.4f} [{dx['ci95'][0]:+.3f},{dx['ci95'][1]:+.3f}]"
            f"{' **sig**' if dx['significant'] else ''} | "
            f"{mm['mean']:+.4f} [{mm['ci95'][0]:+.3f},{mm['ci95'][1]:+.3f}]"
            f"{' **sig**' if mm['significant'] else ''} |")
    L += ["", "## P2 — the pre-registered primary statistic", "",
          "Domain x probe interaction: [Δ(staircase,coding) − Δ(lag_span,coding)] −",
          "[Δ(staircase,babi) − Δ(lag_span,babi)], i.e. matched-minus-mismatched in",
          "coding minus the same in babi. H1 predicts > 0.", "",
          f"**P2 = {p2['mean']:+.4f}, 95% CI [{p2['ci95'][0]:+.3f}, {p2['ci95'][1]:+.3f}]"
          f" — {'SIGNIFICANT' if p2['significant'] else 'not significant'}**", ""]
    (RESULTS / "SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L).encode("ascii", "replace").decode("ascii"))
    return out


if __name__ == "__main__":
    compute()
