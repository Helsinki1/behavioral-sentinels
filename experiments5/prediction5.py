"""Experiment-5 prediction layer: same-trajectory precision/recall for every
reset signal. No new API calls -- everything is re-scored from the per-turn
fields the harness already recorded.

Experiment 3 compared signals ACROSS conditions: each probe was scored on the
trajectories that carried it while the traditional baselines were scored on
baseline trajectories, so a probe with an observer effect was graded on a
different (harder) trajectory distribution than the clock it was compared to.
Here that pitfall is handled by construction: every row of a table is computed
on ONE trajectory set, with the turn-number / context-length / random
thresholds re-scored on that same set.

Scoring unit: the PRE-FIRST-RESET segment of each trajectory (all of it when
the arm never reset), so no reset ever separates a fire from its outcome.
S = first turn the signal fires inside the segment, H = first hallucinated
turn inside the segment; TP/FP/FN/TN and the K-window rule are
experiments/metrics.py's classify(), primary window K = 5.

Censoring caveat, stated rather than hidden: in an arm whose OWN signal
triggers the reset (the probe on D_routed / D_labeled, the judge on C_judge)
the segment ends right after the first fire, so that signal's lead time is
right-censored there and outcomes the reset prevented are unobservable. The
clean reads are (a) the zero-carry monitor on A_no_reset, which never resets,
and (b) the carried probe on C_prime_routed, where the clock -- not the
probe -- ends the segment.
"""
import json
import statistics

from experiments.metrics import classify, summarize, summarize_random

from . import config5
from .config5 import DOMAINS
from .metrics5 import pool_ids
from .run_all5 import load_arm

K = 5
TURN_THRESHOLDS = [3, 5, 8, 10, 12, 15, 20, 25]
CTX_THRESHOLDS = [800, 1000, 1500, 2000, 2500, 3000, 4000, 6000]

# trajectory set -> signals recorded there (key, display name)
SET_SIGNALS = {
    "A_no_reset":     [("zerocarry", "zero-carry monitor")],
    "D_routed":       [("probe", "routed probe"), ("zerocarry", "zero-carry monitor")],
    "C_prime_routed": [("probe", "routed probe"), ("zerocarry", "zero-carry monitor")],
    "D_labeled":      [("probe", "labeled probe"), ("zerocarry", "zero-carry monitor")],
    "C_judge":        [("judge", "LLM judge"), ("zerocarry", "zero-carry monitor")],
}

# in these sets, this signal is the reset trigger -> its own lead is censored
SELF_TRIGGERED = {"D_routed": "probe", "D_labeled": "probe", "C_judge": "judge"}

ROUTED_SIGNALS = {("D_routed", "probe"), ("D_labeled", "probe"),
                  ("A_no_reset", "zerocarry"), ("C_prime_routed", "probe")}

FIRE = {
    "probe": lambda r: bool(r.get("probe_fail")),
    "zerocarry": lambda r: bool(r.get("zerocarry_fired")),
    "judge": lambda r: bool(r.get("judge_yes")),
}


def segment(traj):
    """Records before the first reset (all records if the arm never reset)."""
    cut = traj["reset_turns"][0] if traj["reset_turns"] else None
    return [r for r in traj["records"] if cut is None or r["turn"] < cut]


def make_items(model, arm):
    ids = pool_ids()
    items = []
    for d in DOMAINS:
        for tid, traj in load_arm(model, d, arm, ids[d]).items():
            recs = segment(traj)
            H = next((r["turn"] for r in recs if r["hallucination"]), None)
            items.append({
                "domain": d, "tid": tid, "recs": recs, "H": H,
                "hz": recs[-1]["turn"], "truncated": bool(traj["reset_turns"]),
            })
    return items


def first_fire(recs, key):
    return next((r["turn"] for r in recs if FIRE[key](r)), None)


def ctx_fire(recs, th):
    return next((r["turn"] for r in recs
                 if r.get("prompt_tokens") is not None
                 and r["prompt_tokens"] >= th), None)


def leads_of(pairs):
    return [H - S for S, H, _ in pairs
            if S is not None and H is not None and S <= H]


def score(pairs):
    m = summarize(pairs, K)
    m["leads"] = leads_of(pairs)
    return m


def best_threshold(items, S_fn, thresholds):
    """Sweep a threshold on THESE items and keep the best F1 -- maximally
    generous to the baseline (it is tuned on the evaluation set itself)."""
    best = None
    for th in thresholds:
        pairs = [(S_fn(it, th), it["H"], it["hz"]) for it in items]
        m = score(pairs)
        m["threshold"] = th
        if best is None or (m["f1"] or 0) > (best["f1"] or 0):
            best = m
    return best


def random_metrics(items):
    fake = [{"first_hallucination": it["H"], "horizon": it["hz"]} for it in items]
    m = summarize_random(fake, K)
    m["leads"] = []
    return m


def compute(model="gpt-oss-20b"):
    out = {"model": model, "K": K, "sets": {}}
    for arm, signals in SET_SIGNALS.items():
        items = make_items(model, arm)
        if not items:
            continue
        by_dom = {d: [it for it in items if it["domain"] == d] for d in DOMAINS}
        seg_lens = [it["hz"] for it in items]
        entry = {
            "n_segments": len(items),
            "n_by_domain": {d: len(v) for d, v in by_dom.items()},
            "median_segment_turns": statistics.median(seg_lens),
            "share_truncated_by_reset": round(
                sum(it["truncated"] for it in items) / len(items), 3),
            "self_triggered_signal": dict(signals).get(SELF_TRIGGERED.get(arm)),
            "signals": {},
        }

        def all_scopes(name, scorer):
            entry["signals"][name] = {
                "pooled": scorer(items),
                "by_domain": {d: scorer(v) for d, v in by_dom.items() if v},
            }

        for key, disp in signals:
            all_scopes(disp, lambda its, key=key: score(
                [(first_fire(it["recs"], key), it["H"], it["hz"]) for it in its]))
        all_scopes("turn_number", lambda its: best_threshold(
            its, lambda it, th: th if it["hz"] >= th else None, TURN_THRESHOLDS))
        all_scopes("context_length", lambda its: best_threshold(
            its, lambda it, th: ctx_fire(it["recs"], th), CTX_THRESHOLDS))
        all_scopes("random (expected)", lambda its: random_metrics(its))
        out["sets"][arm] = entry

    config5.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config5.RESULTS_DIR / "prediction.json").write_text(json.dumps(out, indent=2))
    write_report(out)
    return out


def _f(x):
    return "-" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def _row(name, m, mark=""):
    th = f" (th={m['threshold']})" if "threshold" in m else ""
    return (f"| {name}{th}{mark} | {_f(m['precision'])} | {_f(m['recall'])} | "
            f"{_f(m['f1'])} | {_f(m['fire_rate'])} | {_f(m['lead_time_median'])} "
            f"| {m['n']} |")


HEADER = ("| signal | precision | recall | F1 | fire rate | median lead | n |\n"
          "|---|---|---|---|---|---|---|")

SET_NOTES = {
    "A_no_reset": "never resets: full-horizon trajectories, the cleanest read",
    "D_routed": "the probe itself triggers resets: its lead is right-censored "
                "at the fire; the same-set baselines share the segments",
    "C_prime_routed": "the CLOCK ends segments (first reset at turn 6), so "
                      "the probe's fires are not self-censored -- but "
                      "segments are short",
    "D_labeled": "deterministically routed probe triggers resets (same "
                 "censoring as D_routed, no router noise)",
    "C_judge": "the judge triggers resets: its lead is right-censored at "
               "the fire",
}


def write_report(out):
    L = [f"# Experiment 5 — same-trajectory prediction metrics (K={K})", "",
         f"Model `{out['model']}`. Signal quality separated from intervention "
         "value: each table below re-scores every signal — including the "
         "turn-number / context-length / random baselines — on ONE set of "
         "trajectories (the exp-3 cross-condition pitfall handled by "
         "construction). Scoring unit = the pre-first-reset segment; "
         "S = first fire, H = first hallucination in the segment; "
         f"TP within K={K} turns (experiments/metrics.py rule). Thresholded "
         "baselines are tuned to best F1 on the evaluation set itself — "
         "maximally generous to them.", "",
         "Censoring: where a set's own signal triggers the reset, that "
         "signal's segment ends at its first fire (lead ≈ 0 by construction, "
         "prevented outcomes unobservable). Clean reads: the zero-carry "
         "monitor on `A_no_reset`, the carried probe on `C_prime_routed`.",
         ""]
    for arm, e in out["sets"].items():
        L += [f"## Trajectory set `{arm}` — {SET_NOTES.get(arm, '')}", "",
              f"{e['n_segments']} segments ("
              + ", ".join(f"{n} {d}" for d, n in e["n_by_domain"].items())
              + f"), median length {e['median_segment_turns']} turns, "
              f"{e['share_truncated_by_reset']:.0%} truncated by a reset."
              + (f" Self-triggered signal: `{e['self_triggered_signal']}`."
                 if e["self_triggered_signal"] else ""), "",
              "### Pooled", "", HEADER]
        for name, s in e["signals"].items():
            mark = " ✂" if name == e["self_triggered_signal"] else ""
            L.append(_row(name, s["pooled"], mark))
        L += ["", "### Per domain", "",
              "| domain | signal | precision | recall | F1 | fire rate | "
              "median lead | n |", "|---|---|---|---|---|---|---|---|"]
        for d in DOMAINS:
            for name, s in e["signals"].items():
                m = s["by_domain"].get(d)
                if m:
                    L.append(f"| {d} " + _row(name, m))
        L.append("")
    L += ["✂ = self-triggered on this set: the segment ends at this signal's "
          "first fire, so its lead time is right-censored.", ""]
    p = config5.RESULTS_DIR / "PREDICTION.md"
    p.write_text("\n".join(L), encoding="utf-8")
    print("wrote", p)


if __name__ == "__main__":
    compute()
