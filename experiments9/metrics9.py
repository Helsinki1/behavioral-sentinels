"""Experiment-9 analysis: per-model arms/contrasts/prediction on the sharded-
math pool, plus the cross-model headline table.

Outputs: results9/metrics.json, prediction.json, SUMMARY.md, PREDICTION.md.
"""
import json
import statistics

from experiments.metrics import summarize, summarize_random
from experiments8.metrics8 import boot_ci, contrast as contrast8

from .config9 import (ARMS, ARM_DISPLAY, MODEL_ORDER, RESULTS_DIR,
                      SUCCESS_THRESHOLD)
from .run_all9 import load_arm, load_pool
from .shadow9 import load_shadow

K = 5
TURN_THRESHOLDS = [3, 5, 8, 10, 12, 15]
CTX_THRESHOLDS = [400, 600, 800, 1000, 1500, 2000]
QUIZ_FAIL_SWEEP = [1, 2, 3]

CONTRAST_PAIRS = [
    ("QUIZ", "C_clock", "frozen-state QUIZ vs clock"),
    ("QUIZ", "Z_trace", "quiz vs zero-carry trace monitor"),
    ("QUIZ", "A_no_reset", "quiz vs never resetting"),
    ("QUIZ", "F_oracle", "quiz vs oracle"),
    ("ACT_probe", "QUIZ", "ACTIVE probe vs passive quiz"),
    ("ACT_probe", "C_clock", "active probe vs clock"),
    ("ACT_probe", "A_no_reset", "active probe vs never resetting"),
    ("ACT_carry_clock", "C_clock", "OBSERVER-EFFECT COST"),
    ("ACT_probe", "ACT_carry_clock", "timing value of the active signal"),
    ("Z_trace", "C_clock", "trace monitor vs clock"),
    ("Z_trace", "A_no_reset", "trace monitor vs never resetting"),
    ("C_clock", "A_no_reset", "clock vs never resetting"),
    ("F_oracle", "A_no_reset", "oracle vs never resetting"),
]

FIRE = {
    "probe": lambda r: bool(r.get("probe_fail")),
    "zerocarry": lambda r: bool(r.get("zerocarry_fired")),
    "quiz": lambda r: bool(r.get("quiz_fail")),
}
SET_SIGNALS = {
    "A_no_reset":      [("zerocarry", "zero-carry trace monitor")],
    "QUIZ":            [("quiz", "frozen-state quiz (live)"),
                        ("zerocarry", "zero-carry trace monitor")],
    "ACT_carry_clock": [("probe", "carried probe (clock-truncated read)"),
                        ("zerocarry", "zero-carry trace monitor")],
    "ACT_probe":       [("probe", "carried probe"),
                        ("zerocarry", "zero-carry trace monitor")],
}
SELF_TRIGGERED = {"QUIZ": "quiz", "ACT_probe": "probe"}


def arm_stats(recs):
    acc = [r["accuracy"] for r in recs]
    return {
        "n": len(recs),
        "accuracy": round(statistics.mean(acc), 4),
        "success_rate": round(
            sum(1 for a in acc if a >= SUCCESS_THRESHOLD) / len(acc), 4),
        "resets_per_task": round(
            statistics.mean(r["n_resets"] for r in recs), 3),
        "prompt_tokens": round(
            statistics.mean(r["prompt_tokens"] for r in recs)),
        "completion_tokens": round(
            statistics.mean(r["completion_tokens"] for r in recs)),
        "quiz_tokens": round(statistics.mean(
            (r.get("quiz_prompt_tokens", 0) or 0)
            + (r.get("quiz_completion_tokens", 0) or 0) for r in recs)),
    }


def segment(traj):
    cut = traj["reset_turns"][0] if traj["reset_turns"] else None
    return [r for r in traj["records"] if cut is None or r["turn"] < cut]


def make_items(recs_map):
    items = []
    for tid, traj in sorted(recs_map.items()):
        recs = segment(traj)
        if not recs:
            continue
        H = next((r["turn"] for r in recs if r["hallucination"]), None)
        items.append({"tid": tid, "recs": recs, "H": H,
                      "hz": recs[-1]["turn"],
                      "truncated": bool(traj["reset_turns"])})
    return items


def score(pairs):
    return summarize(pairs, K)


def first_fire(recs, key):
    return next((r["turn"] for r in recs if FIRE[key](r)), None)


def best_threshold(items, S_fn, thresholds):
    best = None
    for th in thresholds:
        m = score([(S_fn(it, th), it["H"], it["hz"]) for it in items])
        m["threshold"] = th
        if best is None or (m["f1"] or 0) > (best["f1"] or 0):
            best = m
    return best


def ctx_fire(recs, th):
    return next((r["turn"] for r in recs
                 if r.get("prompt_tokens") is not None
                 and r["prompt_tokens"] >= th), None)


def shadow_S(sh, fail_min):
    return next((c["turn"] for c in sh["checkpoints"]
                 if c["n_wrong"] >= fail_min), None)


def prediction_for(model, all_recs, ids):
    out = {"sets": {}, "shadow": {}}
    shadows = load_shadow(model, ids)
    if shadows:
        for fail_min in QUIZ_FAIL_SWEEP:
            out["shadow"][f"fail>={fail_min}"] = score(
                [(shadow_S(sh, fail_min), sh["first_hallucination"],
                  sh["horizon"]) for sh in shadows.values()])
    for arm, signals in SET_SIGNALS.items():
        if arm not in all_recs:
            continue
        items = make_items(all_recs[arm])
        if not items:
            continue
        entry = {"n_segments": len(items),
                 "median_segment_turns": statistics.median(
                     it["hz"] for it in items),
                 "self_triggered_signal": dict(signals).get(
                     SELF_TRIGGERED.get(arm)),
                 "signals": {}}
        for key, disp in signals:
            entry["signals"][disp] = score(
                [(first_fire(it["recs"], key), it["H"], it["hz"])
                 for it in items])
        if arm == "A_no_reset":
            if shadows:
                entry["signals"]["frozen-state quiz (shadow)"] = score(
                    [(shadow_S(shadows[it["tid"]], 2), it["H"], it["hz"])
                     for it in items if it["tid"] in shadows])
            entry["signals"]["turn_number"] = best_threshold(
                items, lambda it, th: th if it["hz"] >= th else None,
                TURN_THRESHOLDS)
            entry["signals"]["context_length"] = best_threshold(
                items, lambda it, th: ctx_fire(it["recs"], th),
                CTX_THRESHOLDS)
            entry["signals"]["random (expected)"] = summarize_random(
                [{"first_hallucination": it["H"], "horizon": it["hz"]}
                 for it in items], K)
        out["sets"][arm] = entry
    return out


def compute():
    pool = load_pool()
    ids = [t["task_id"] for t in pool]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {"models": {}}
    pred = {"models": {}}
    for model in MODEL_ORDER:
        all_recs = {arm: load_arm(model, arm, ids) for arm in ARMS}
        all_recs = {a: r for a, r in all_recs.items() if r}
        if not all_recs:
            continue
        common = sorted(set.intersection(*[set(v) for v in all_recs.values()]))
        stats = {arm: arm_stats([recs[k] for k in common])
                 for arm, recs in all_recs.items()}
        contrasts = []
        for a, b, lab in CONTRAST_PAIRS:
            if a in all_recs and b in all_recs:
                contrasts.append(contrast8(all_recs[a], all_recs[b],
                                           f"{a} - {b}: {lab}", common))
        out["models"][model] = {"n_tasks": len(common), "arms": stats,
                                "contrasts": contrasts}
        pred["models"][model] = prediction_for(model, all_recs, ids)

    (RESULTS_DIR / "metrics.json").write_text(json.dumps(out, indent=2))
    (RESULTS_DIR / "prediction.json").write_text(json.dumps(pred, indent=2))
    write_summary(out, pred)
    return out, pred


def _f(x):
    return "-" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def _prow(name, m, mark=""):
    th = f" (th={m['threshold']})" if "threshold" in m else ""
    return (f"| {name}{th}{mark} | {_f(m['precision'])} | {_f(m['recall'])} | "
            f"{_f(m['f1'])} | {_f(m['fire_rate'])} | "
            f"{_f(m['lead_time_median'])} | {m['n']} |")


PRED_HEADER = ("| signal | precision | recall | F1 | fire rate | median lead "
               "| n |\n|---|---|---|---|---|---|---|")


def write_summary(out, pred):
    L = ["# Experiment 9 — active vs passive observation on sharded GSM8K, "
         "four models", "",
         "Sessions of 3 sharded math problems (lost_in_conversation `math`, "
         "arXiv:2505.06120), one verbatim shard per turn, WAIT/ANSWER "
         "protocol; hallucination = premature ANSWER, missing WAIT, or "
         "missing/wrong final ANSWER vs the GSM8K key. R1 reground resets; "
         "same arms/policies as exp 8.", ""]

    L += ["## Cross-model headline", "",
          "| model | " + " | ".join(ARM_DISPLAY) + " | observer Δ (sig) "
          "| quiz prec (shadow≥2) | trace-monitor prec |",
          "|---|" + "---|" * (len(ARM_DISPLAY) + 3)]
    for model, m in out["models"].items():
        cells = []
        for a in ARM_DISPLAY:
            s = m["arms"].get(a)
            cells.append(f"{s['accuracy']:.3f}" if s else "—")
        oc = next((c for c in m["contrasts"]
                   if c["contrast"].startswith("ACT_carry_clock - C_clock")),
                  None)
        occ = (f"{oc['mean_delta']:+.3f} "
               f"({'y' if oc['significant'] else 'n'})") if oc else "—"
        p = pred["models"].get(model, {})
        qz = p.get("shadow", {}).get("fail>=2", {}).get("precision")
        zc = (p.get("sets", {}).get("A_no_reset", {}).get("signals", {})
               .get("zero-carry trace monitor", {}).get("precision"))
        L.append(f"| {model} | " + " | ".join(cells)
                 + f" | {occ} | {_f(qz)} | {_f(zc)} |")

    for model, m in out["models"].items():
        L += ["", f"## {model} — {m['n_tasks']} sessions paired", "",
              "| arm | category | accuracy | success@0.9 | resets/task "
              "| prompt tok | quiz tok |", "|---|---|---|---|---|---|---|"]
        for a in [x for x in ARM_DISPLAY if x in m["arms"]]:
            s = m["arms"][a]
            L.append(f"| {a} | {ARMS[a]['category']} | {s['accuracy']:.3f} "
                     f"| {s['success_rate']:.3f} | {s['resets_per_task']:.2f} "
                     f"| {s['prompt_tokens']:,} | {s['quiz_tokens']:,} |")
        L += ["", "| contrast | delta | 95% CI | sig | better/worse/tied |",
              "|---|---|---|---|---|"]
        for c in m["contrasts"]:
            L.append(f"| {c['contrast']} | {c['mean_delta']:+.4f} | "
                     f"[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] | "
                     f"{'**yes**' if c['significant'] else 'no'} | "
                     f"{c['tasks_better']}/{c['tasks_worse']}/"
                     f"{c['tasks_tied']} |")
    (RESULTS_DIR / "SUMMARY.md").write_text("\n".join(L) + "\n",
                                            encoding="utf-8")

    P = [f"# Experiment 9 — signal quality per model (K={K})", "",
         "Quiz checkpoints every 3 turns (3-turn S granularity); thresholded "
         "baselines tuned to best F1 on the evaluation set itself. ✂ = the "
         "signal triggers that set's resets (lead right-censored).", ""]
    for model, p in pred["models"].items():
        P += [f"## {model}", ""]
        if p.get("shadow"):
            P += ["Quiz fail-threshold ablation (shadow pass):", "",
                  PRED_HEADER]
            for name, s in p["shadow"].items():
                P.append(_prow(f"quiz {name}", s))
            P.append("")
        for arm, e in p["sets"].items():
            P += [f"### set `{arm}` — {e['n_segments']} segments, median "
                  f"{e['median_segment_turns']} turns", "", PRED_HEADER]
            for name, s in e["signals"].items():
                mark = " ✂" if name == e.get("self_triggered_signal") else ""
                P.append(_prow(name, s, mark))
            P.append("")
    (RESULTS_DIR / "PREDICTION.md").write_text("\n".join(P) + "\n",
                                               encoding="utf-8")
    print("wrote", RESULTS_DIR / "SUMMARY.md", "and PREDICTION.md")


if __name__ == "__main__":
    compute()
