"""Trajectory-level prediction metrics.

For each trajectory we have:
  H = first hallucination turn (None if the task finished clean)
  S = first signal firing turn (first canary failure / threshold crossing /
      judge YES / random compaction turn)

Classification at window K ("signal at turn S predicts a hallucination
within the next K turns", firing at or before the hallucination):
  TP: H exists and S exists and S <= H and H - S <= K
  FN: H exists, not predicted (no S, S fired late, or window exceeded)
  FP: S exists but no hallucination ever occurred
  TN: neither
Lead time = H - S over trajectories where both exist and S <= H.
"""
import json
import statistics

from .config import (BASELINE_CONDITION, CANARY_CONDITIONS,
                     CONTEXT_LENGTH_THRESHOLDS, K_VALUES, MODELS, PRIMARY_K,
                     RESULTS_DIR, RUNS_DIR, TRADITIONAL_SIGNALS,
                     TURN_NUMBER_THRESHOLDS)
from .judge import judge_path
from .runner import traj_path


def classify(S, H, K):
    if H is not None:
        if S is not None and S <= H and (K == "inf" or H - S <= K):
            return "TP"
        return "FN"
    return "FP" if S is not None else "TN"


def summarize(pairs, K):
    """pairs: list of (S, H, horizon). Returns metric dict for window K."""
    counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    leads = []
    for S, H, _ in pairs:
        counts[classify(S, H, K)] += 1
        if S is not None and H is not None and S <= H:
            leads.append(H - S)
    tp, fp, fn, tn = counts["TP"], counts["FP"], counts["FN"], counts["TN"]
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = (2 * prec * rec / (prec + rec)) if prec and rec else 0.0
    return {
        "K": K, "n": len(pairs), **counts,
        "precision": round(prec, 4) if prec is not None else None,
        "recall": round(rec, 4) if rec is not None else None,
        "f1": round(f1, 4),
        "accuracy": round((tp + tn) / len(pairs), 4) if pairs else None,
        "hallucination_rate": round((tp + fn) / len(pairs), 4) if pairs else None,
        "fire_rate": round(sum(1 for S, _, _ in pairs if S is not None) / len(pairs), 4) if pairs else None,
        "lead_time_median": statistics.median(leads) if leads else None,
        "lead_time_mean": round(statistics.mean(leads), 2) if leads else None,
        "n_lead_pairs": len(leads),
    }


def summarize_random(trajs, K):
    """Analytic expectation for a compaction trigger at a uniformly random
    turn s in {1..horizon} (the policy does not know about early stops)."""
    exp = {"TP": 0.0, "FP": 0.0, "FN": 0.0, "TN": 0.0}
    leads = []
    for tr in trajs:
        H, horizon = tr["first_hallucination"], tr["horizon"]
        for s in range(1, horizon + 1):
            exp[classify(s, H, K)] += 1.0 / horizon
            if H is not None and s <= H:
                leads.append(H - s)
    tp, fp, fn, tn = exp["TP"], exp["FP"], exp["FN"], exp["TN"]
    n = len(trajs)
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = (2 * prec * rec / (prec + rec)) if prec and rec else 0.0
    return {
        "K": K, "n": n,
        "TP": round(tp, 2), "FP": round(fp, 2), "FN": round(fn, 2), "TN": round(tn, 2),
        "precision": round(prec, 4) if prec is not None else None,
        "recall": round(rec, 4) if rec is not None else None,
        "f1": round(f1, 4),
        "accuracy": round((tp + tn) / n, 4) if n else None,
        "hallucination_rate": round((tp + fn) / n, 4) if n else None,
        "fire_rate": 1.0,
        "lead_time_median": statistics.median(leads) if leads else None,
        "lead_time_mean": round(statistics.mean(leads), 2) if leads else None,
        "n_lead_pairs": len(leads),
        "note": "expected values over a uniform random firing turn in [1, horizon]",
    }


def load_trajs(model_name, condition, task_ids):
    out = []
    for tid in task_ids:
        p = traj_path(model_name, condition, tid)
        if p.exists():
            d = json.loads(p.read_text())
            if d.get("complete"):
                out.append(d)
    return out


def context_length_S(traj, threshold):
    for rec in traj["records"]:
        pt = rec.get("prompt_tokens")
        if pt is not None and pt >= threshold:
            return rec["turn"]
    return None


def turn_number_S(traj, threshold):
    return threshold if traj["turns_run"] >= threshold else None


def pairs_for(trajs, S_fn):
    return [(S_fn(tr), tr["first_hallucination"], tr["horizon"]) for tr in trajs]


def fmt_row(name, m):
    def f(x):
        return "-" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))
    return (f"| {name} | {f(m['precision'])} | {f(m['recall'])} | {f(m['f1'])} | "
            f"{f(m['TP'])} | {f(m['FP'])} | {f(m['FN'])} | {f(m['TN'])} | "
            f"{f(m['fire_rate'])} | {f(m['lead_time_median'])} |")

TABLE_HEADER = ("| signal | precision | recall | F1 | TP | FP | FN | TN | fire rate | median lead |\n"
                "|---|---|---|---|---|---|---|---|---|---|")


def compute_all(task_ids, model_names=None):
    model_names = model_names or list(MODELS)
    RESULTS_DIR.mkdir(exist_ok=True)
    headline = {}  # (model, signal) -> metrics at PRIMARY_K

    # ---- canary conditions
    for cond in CANARY_CONDITIONS:
        folder = RESULTS_DIR / cond
        folder.mkdir(parents=True, exist_ok=True)
        summary = [f"# Canary: {cond}", ""]
        for m in model_names:
            trajs = load_trajs(m, cond, task_ids)
            pairs = [(tr["first_canary_fail"], tr["first_hallucination"], tr["horizon"]) for tr in trajs]
            per_k = {str(K): summarize(pairs, K) for K in K_VALUES}
            (folder / f"metrics_{m}.json").write_text(json.dumps({
                "signal": cond, "model": m, "kind": MODELS[m]["kind"],
                "metrics_by_K": per_k,
                "per_task": [{"task_id": tr["task_id"], "S": tr["first_canary_fail"],
                              "H": tr["first_hallucination"], "horizon": tr["horizon"]}
                             for tr in trajs],
            }, indent=2))
            headline[(m, cond)] = per_k[str(PRIMARY_K)]
            summary += [f"## model: {m} ({MODELS[m]['kind']}), n={len(trajs)} tasks", "", TABLE_HEADER]
            for K in K_VALUES:
                summary.append(fmt_row(f"K={K}", per_k[str(K)]))
            summary.append("")
        (folder / "summary.md").write_text("\n".join(summary))

    # ---- traditional signals (on baseline trajectories)
    trad_root = RESULTS_DIR / "Traditional"
    for sig in TRADITIONAL_SIGNALS:
        (trad_root / sig).mkdir(parents=True, exist_ok=True)

    for m in model_names:
        trajs = load_trajs(m, BASELINE_CONDITION, task_ids)

        # context length + turn number: threshold sweeps, headline = best F1 at PRIMARY_K
        for sig, thresholds, S_maker, unit in [
            ("context_length", CONTEXT_LENGTH_THRESHOLDS,
             lambda th: (lambda tr: context_length_S(tr, th)), "prompt tokens"),
            ("turn_number", TURN_NUMBER_THRESHOLDS,
             lambda th: (lambda tr: turn_number_S(tr, th)), "turns"),
        ]:
            sweep = {}
            for th in thresholds:
                pairs = pairs_for(trajs, S_maker(th))
                sweep[str(th)] = {str(K): summarize(pairs, K) for K in K_VALUES}
            best_th = max(thresholds, key=lambda th: sweep[str(th)][str(PRIMARY_K)]["f1"])
            headline[(m, f"Traditional/{sig}")] = {**sweep[str(best_th)][str(PRIMARY_K)],
                                                  "threshold": best_th}
            folder = trad_root / sig
            (folder / f"metrics_{m}.json").write_text(json.dumps({
                "signal": sig, "model": m, "unit": unit, "sweep": sweep,
                "best_threshold_at_primary_K": best_th}, indent=2))
            lines = [f"# Traditional signal: {sig} (threshold in {unit})",
                     f"\n## model: {m} ({MODELS[m]['kind']}), n={len(trajs)} tasks, K={PRIMARY_K}",
                     "", TABLE_HEADER]
            for th in thresholds:
                mark = " *best F1*" if th == best_th else ""
                lines.append(fmt_row(f"θ={th}{mark}", sweep[str(th)][str(PRIMARY_K)]))
            lines.append("")
            mode = "a" if (folder / "summary.md").exists() else "w"
            with open(folder / "summary.md", mode) as fh:
                fh.write("\n".join(lines) + "\n")

        # LLM judge
        pairs = []
        for tr in trajs:
            jp = judge_path(m, tr["task_id"])
            S = json.loads(jp.read_text()).get("first_yes") if jp.exists() else None
            pairs.append((S, tr["first_hallucination"], tr["horizon"]))
        per_k = {str(K): summarize(pairs, K) for K in K_VALUES}
        headline[(m, "Traditional/LLM_judge")] = per_k[str(PRIMARY_K)]
        folder = trad_root / "LLM_judge"
        (folder / f"metrics_{m}.json").write_text(json.dumps({
            "signal": "LLM_judge", "model": m, "metrics_by_K": per_k}, indent=2))
        lines = [f"# Traditional signal: LLM judge (gpt-4o-mini, last-8-turn window)",
                 f"\n## model: {m} ({MODELS[m]['kind']}), n={len(trajs)} tasks", "", TABLE_HEADER]
        for K in K_VALUES:
            lines.append(fmt_row(f"K={K}", per_k[str(K)]))
        mode = "a" if (folder / "summary.md").exists() else "w"
        with open(folder / "summary.md", mode) as fh:
            fh.write("\n".join(lines) + "\n\n")

        # random compaction (analytic expectation)
        per_k = {str(K): summarize_random(trajs, K) for K in K_VALUES}
        headline[(m, "Traditional/random_compaction")] = per_k[str(PRIMARY_K)]
        folder = trad_root / "random_compaction"
        (folder / f"metrics_{m}.json").write_text(json.dumps({
            "signal": "random_compaction", "model": m, "metrics_by_K": per_k}, indent=2))
        lines = [f"# Traditional signal: random compaction (uniform random firing turn)",
                 f"\n## model: {m} ({MODELS[m]['kind']}), n={len(trajs)} tasks", "", TABLE_HEADER]
        for K in K_VALUES:
            lines.append(fmt_row(f"K={K}", per_k[str(K)]))
        mode = "a" if (folder / "summary.md").exists() else "w"
        with open(folder / "summary.md", mode) as fh:
            fh.write("\n".join(lines) + "\n\n")

    # ---- top-level summary
    sigs = CANARY_CONDITIONS + [f"Traditional/{s}" for s in TRADITIONAL_SIGNALS]
    lines = ["# Behavioral Sentinels — Results Summary",
             f"\n200 synthetic state book-keeping tasks, horizons 15-35 turns. "
             f"Primary prediction window K={PRIMARY_K} turns. "
             "TP = signal fired at/before the first hallucination and within K turns of it; "
             "FP = fired on a clean trajectory; FN = hallucination not predicted "
             "(no firing, fired late, or window exceeded); TN = clean and silent. "
             "Context-length/turn-number rows use the best-F1 threshold from their sweep "
             "(see Traditional/*/summary.md). Random compaction is an analytic expectation.", ""]
    for m in model_names:
        lines += [f"\n## model: {m} ({MODELS[m]['kind']}), K={PRIMARY_K}", "", TABLE_HEADER]
        for sig in sigs:
            met = headline.get((m, sig))
            if met:
                name = sig + (f" (θ={met['threshold']})" if "threshold" in met else "")
                lines.append(fmt_row(name, met))
        base = headline.get((m, CANARY_CONDITIONS[0]))
        if base:
            lines.append(f"\nHallucination base rate (canary runs vary slightly): "
                         f"{base['hallucination_rate']}")
    (RESULTS_DIR / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote results to {RESULTS_DIR}")
    return headline
