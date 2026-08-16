"""Experiment-2 prediction metrics.

The classification rule is imported verbatim from experiment 1 so the two
experiments are scored identically:

  TP: hallucination happened, signal fired at/before it and within K turns
  FN: hallucination happened but was not predicted in time
  FP: signal fired on a trajectory that never hallucinated
  TN: clean and silent

Experiment 2 adds one thing experiment 1 did not have: a breakdown of WHICH
kind of coding hallucination fired first (wrong signature, fabricated dead
function, invented helper call, syntax error, abandoned edit).
"""
import collections
import json
import statistics

from experiments.metrics import (TABLE_HEADER, classify, fmt_row, summarize,
                                 summarize_random)

from .config2 import (ALL_CONDITIONS, BASELINE_CONDITION, CANARY_CONDITIONS,
                      CONTEXT_LENGTH_THRESHOLDS, K_VALUES, MODELS, PRIMARY_K,
                      RESULTS_DIR, TRADITIONAL_SIGNALS, TURN_NUMBER_THRESHOLDS)
from .judge2 import judge_path
from .runner2 import traj_path


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


def error_mix(trajs):
    """Which hallucination kind fired first, across trajectories."""
    c = collections.Counter()
    for tr in trajs:
        if tr["first_hallucination"] is None:
            continue
        rec = tr["records"][-1]
        for kind, _ in rec["errors"]:
            c[kind] += 1
    return dict(c.most_common())


def paired_task_ids(model_name, task_ids):
    """Task ids that completed under EVERY condition for this model.

    The two model arms may cover different numbers of tasks (API quota), so
    every number reported for a model is computed on one common, paired task
    set rather than on whatever happened to finish per condition."""
    sets = []
    for c in ALL_CONDITIONS:
        sets.append({tr["task_id"] for tr in load_trajs(model_name, c, task_ids)})
    return sorted(set.intersection(*sets)) if sets and all(sets) else []


def compute_all(task_ids, model_names=None):
    model_names = model_names or list(MODELS)
    paired = {m: paired_task_ids(m, task_ids) for m in model_names}
    for m in list(model_names):
        print(f"  {m}: {len(paired[m])} tasks complete under all "
              f"{len(ALL_CONDITIONS)} conditions (paired set)")
    # a model with no complete paired set contributes nothing scoreable
    model_names = [m for m in model_names if paired[m]]
    if not model_names:
        raise SystemExit("no model has a complete paired task set yet")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    headline = {}
    mixes = {}

    # ---- dynamic canary conditions
    for cond in CANARY_CONDITIONS:
        folder = RESULTS_DIR / cond
        folder.mkdir(parents=True, exist_ok=True)
        summary = [f"# Experiment 2 canary: {cond}", ""]
        for m in model_names:
            trajs = load_trajs(m, cond, paired[m])
            pairs = [(tr["first_canary_fail"], tr["first_hallucination"], tr["horizon"])
                     for tr in trajs]
            per_k = {str(K): summarize(pairs, K) for K in K_VALUES}
            (folder / f"metrics_{m}.json").write_text(json.dumps({
                "signal": cond, "model": m, "kind": MODELS[m]["kind"],
                "metrics_by_K": per_k,
                "first_error_mix": error_mix(trajs),
                "per_task": [{"task_id": tr["task_id"], "S": tr["first_canary_fail"],
                              "H": tr["first_hallucination"], "horizon": tr["horizon"]}
                             for tr in trajs],
            }, indent=2))
            headline[(m, cond)] = per_k[str(PRIMARY_K)]
            mixes[(m, cond)] = error_mix(trajs)
            summary += [f"## model: {m} ({MODELS[m]['kind']}), n={len(trajs)} tasks", "",
                        TABLE_HEADER]
            for K in K_VALUES:
                summary.append(fmt_row(f"K={K}", per_k[str(K)]))
            summary += ["", f"first-hallucination error mix: {error_mix(trajs)}", ""]
        (folder / "summary.md").write_text("\n".join(summary), encoding="utf-8")

    # ---- traditional signals, measured on the baseline (no-canary) runs
    trad_root = RESULTS_DIR / "Traditional"
    for sig in TRADITIONAL_SIGNALS:
        (trad_root / sig).mkdir(parents=True, exist_ok=True)
        p = trad_root / sig / "summary.md"
        if p.exists():
            p.unlink()

    for m in model_names:
        trajs = load_trajs(m, BASELINE_CONDITION, paired[m])
        mixes[(m, "baseline")] = error_mix(trajs)

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
                lines.append(fmt_row(f"th={th}{mark}", sweep[str(th)][str(PRIMARY_K)]))
            with open(folder / "summary.md", "a") as fh:
                fh.write("\n".join(lines) + "\n\n")

        # LLM judge
        pairs, judged, judge_model = [], 0, None
        for tr in trajs:
            jp = judge_path(m, tr["task_id"])
            S = None
            if jp.exists():
                jd = json.loads(jp.read_text())
                S = jd.get("first_yes")
                judge_model = jd.get("judge_model", judge_model)
                judged += 1
            pairs.append((S, tr["first_hallucination"], tr["horizon"]))
        per_k = {str(K): summarize(pairs, K) for K in K_VALUES}
        if judged:
            headline[(m, "Traditional/LLM_judge")] = per_k[str(PRIMARY_K)]
        else:
            print(f"  [skip] LLM_judge for {m}: no judge files -- row omitted "
                  f"rather than reported as a spurious zero")
        folder = trad_root / "LLM_judge"
        (folder / f"metrics_{m}.json").write_text(json.dumps({
            "signal": "LLM_judge", "model": m, "judge_model": judge_model,
            "n_judged": judged, "metrics_by_K": per_k}, indent=2))
        lines = [f"# Traditional signal: LLM judge ({judge_model}, last-8-turn "
                 f"window, n judged = {judged})",
                 f"\n## model: {m} ({MODELS[m]['kind']}), n={len(trajs)} tasks", "", TABLE_HEADER]
        for K in K_VALUES:
            lines.append(fmt_row(f"K={K}", per_k[str(K)]))
        with open(folder / "summary.md", "a") as fh:
            fh.write("\n".join(lines) + "\n\n")

        # random compaction (analytic expectation)
        per_k = {str(K): summarize_random(trajs, K) for K in K_VALUES}
        headline[(m, "Traditional/random_compaction")] = per_k[str(PRIMARY_K)]
        folder = trad_root / "random_compaction"
        (folder / f"metrics_{m}.json").write_text(json.dumps({
            "signal": "random_compaction", "model": m, "metrics_by_K": per_k}, indent=2))
        lines = ["# Traditional signal: random compaction (uniform random firing turn)",
                 f"\n## model: {m} ({MODELS[m]['kind']}), n={len(trajs)} tasks", "", TABLE_HEADER]
        for K in K_VALUES:
            lines.append(fmt_row(f"K={K}", per_k[str(K)]))
        with open(folder / "summary.md", "a") as fh:
            fh.write("\n".join(lines) + "\n\n")

    # ---- top-level summary
    sigs = CANARY_CONDITIONS + [f"Traditional/{s}" for s in TRADITIONAL_SIGNALS]
    n_tasks = ", ".join(f"{len(paired[m])} ({m})" for m in model_names)
    lines = ["# Experiment 2 — Dynamic Canaries on a Coding Task",
             "",
             f"{n_tasks} synthetic incremental-coding tasks (maintain a Python module across "
             f"12-30 turns of add/rename/delete/re-signature edits). Primary prediction window "
             f"K={PRIMARY_K} turns. TP = signal fired at/before the first hallucination and "
             "within K turns of it; FP = fired on a clean trajectory; FN = hallucination not "
             "predicted (no firing, fired late, or window exceeded); TN = clean and silent. "
             "Context-length/turn-number rows use the best-F1 threshold from their sweep. "
             "Random compaction is an analytic expectation over a uniform firing turn.",
             "",
             "Canaries are DYNAMIC (the required output changes over the trajectory) except "
             "`static_trailer`, which is the experiment-1-style fixed-string control.", ""]
    for m in model_names:
        lines += [f"\n## model: {m} ({MODELS[m]['kind']}), n={len(paired[m])} paired tasks, "
                  f"K={PRIMARY_K}", "", TABLE_HEADER]
        for sig in sigs:
            met = headline.get((m, sig))
            if met:
                name = sig + (f" (th={met['threshold']})" if "threshold" in met else "")
                lines.append(fmt_row(name, met))
        base = headline.get((m, CANARY_CONDITIONS[0]))
        if base:
            lines.append(f"\nHallucination base rate (canary runs vary slightly): "
                         f"{base['hallucination_rate']}")
        mix = mixes.get((m, "baseline"))
        if mix:
            lines.append(f"\nFirst-hallucination error mix (baseline runs): {mix}")
    (RESULTS_DIR / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote experiment-2 results to {RESULTS_DIR}")
    return headline
