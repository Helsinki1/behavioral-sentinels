"""Experiment 10, Study A — the sentinel break-even surface.

Six experiments produced a horse race: which trigger gets the highest accuracy.
That framing hides the actual deployment question, because it prices a reset at
zero. In a real system a restart costs re-onboarding latency, a destroyed
prompt cache, and sometimes human attention -- and the sentinel's whole
remaining value proposition is that it buys most of the clock's benefit with a
quarter of the restarts.

So instead of asking "which policy wins", ask "which policy wins AT WHAT PRICE".
Each policy is a point (accuracy, resets/task, tokens/task). Under a linear
utility

    U = accuracy - R * resets_per_task - T * prompt_tokens_per_task/1000

every policy is a plane over the (R, T) cost plane, and the winner is the upper
envelope. Sweeping R and T produces a decision map: given YOUR cost of a
restart and YOUR cost of a token, it names the policy you should run.

R and T are in accuracy-equivalent units: R = 0.01 means one restart costs as
much as one point of task accuracy.

This study needs no new API calls -- it re-reads experiments 5 and 6.
"""
import json

from .config10 import (COST_PER_RESET_GRID, COST_PER_KTOK_GRID, REGIMES,
                      RESULTS_DIR)


def load_regime(regime):
    """Return {policy: (accuracy, resets_per_task, ktokens)} for one operator regime."""
    spec = REGIMES[regime]
    arms = json.loads(open(spec["metrics"]).read())["arms"]
    out = {}
    for name in spec["policies"]:
        if name in arms:
            a = arms[name]
            out[spec["policies"][name]] = (a["accuracy"], a["resets_per_task"],
                                           a["prompt_tokens"] / 1000.0)
    return out


def utility(point, R, T):
    acc, resets, ktok = point
    return acc - R * resets - T * ktok


def winner(points, R, T):
    return max(points, key=lambda k: utility(points[k], R, T))


def breakeven_R(points, a, b, T=0.0):
    """Cost per reset at which policy `a` overtakes policy `b`. None if `a`
    never overtakes (it uses at least as many resets and is no more accurate)."""
    accA, resA, tokA = points[a]
    accB, resB, tokB = points[b]
    d_acc = accA - accB
    d_res = resA - resB
    d_tok = (tokA - tokB) * T
    if d_res >= 0:
        return None if d_acc - d_tok <= 0 else 0.0   # already dominant
    return max(0.0, (d_acc - d_tok) / (-d_res) * -1) if False else (d_tok - d_acc) / (-d_res)


def compute():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for regime in REGIMES:
        pts = load_regime(regime)
        grid = []
        for T in COST_PER_KTOK_GRID:
            row = [winner(pts, R, T) for R in COST_PER_RESET_GRID]
            grid.append(row)
        sentinel = next((k for k in pts if k.startswith("sentinel")), None)
        clock = "clock"
        be = {}
        if sentinel and clock in pts:
            for T in (0.0, 0.001, 0.002):
                be[str(T)] = breakeven_R(pts, sentinel, clock, T)
        out[regime] = {
            "points": {k: {"accuracy": v[0], "resets": v[1], "ktokens": v[2]}
                       for k, v in pts.items()},
            "grid": grid,
            "R_grid": COST_PER_RESET_GRID,
            "T_grid": COST_PER_KTOK_GRID,
            "sentinel_breakeven_R_vs_clock": be,
        }
    (RESULTS_DIR / "breakeven.json").write_text(json.dumps(out, indent=2))
    return out


def report():
    out = compute()
    L = ["# Experiment 10, Study A — the sentinel break-even surface", "",
         "Re-analysis of experiments 5 and 6; no new API calls. Each policy is scored by",
         "", "    U = accuracy - R x resets_per_task - T x prompt_ktokens_per_task", "",
         "with R the cost of one restart and T the cost of 1k prompt tokens, both in",
         "accuracy-equivalent units (R = 0.01 means a restart costs one accuracy point).",
         "The question is not which policy wins, but at what price each one wins.", ""]
    for regime, d in out.items():
        L += [f"## {REGIMES[regime]['label']}", "",
              "| policy | accuracy | resets/task | prompt ktok |", "|---|---|---|---|"]
        for k, v in sorted(d["points"].items(), key=lambda x: -x[1]["accuracy"]):
            L.append(f"| {k} | {v['accuracy']:.3f} | {v['resets']:.2f} | {v['ktokens']:.1f} |")
        be = d["sentinel_breakeven_R_vs_clock"]
        L.append("")
        if be:
            r0 = be.get("0.0")
            if r0 is None:
                L.append("The sentinel never overtakes the clock in this regime.")
            elif r0 <= 0:
                L.append("**The sentinel dominates the clock outright** — higher accuracy *and* "
                         "fewer restarts, so it wins at any restart price including zero.")
            else:
                L.append(f"**Break-even: the sentinel overtakes the clock once one restart costs "
                         f"more than {r0:.4f} accuracy-equivalents** "
                         f"({r0*100:.2f} accuracy points per restart), at zero token cost.")
        L.append("")
    (RESULTS_DIR / "STUDY_A.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return out


if __name__ == "__main__":
    report()
