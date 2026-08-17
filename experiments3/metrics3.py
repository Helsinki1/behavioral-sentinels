"""Experiment-3 prediction metrics, computed independently per task set.

The classification rule (classify / summarize / summarize_random) is imported
verbatim from experiments/metrics.py -- third experiment scored by the same
code.  New here:

  * canaries are scored, so graded canaries additionally get a score-threshold
    sweep (fire when score <= theta), mirroring the treatment the traditional
    context-length / turn-number signals have always received
  * an ensemble row (any member fails) and an ensemble_2of3 row (two members
    failing on the same turn), from the single combined-instruction runs
  * an observer-effect table: hallucination rate per condition vs baseline --
    does carrying the canary itself accelerate task failure?
  * a secondary table excluding trajectories that hallucinate on turn 1
    (no signal can lead those; ~20 percent of trajectories in experiment 2)
  * confab_trap fabrication share and sparse_recall rehearsal rate
"""
import collections
import json

from experiments.metrics import (TABLE_HEADER, fmt_row, summarize,
                                 summarize_random)

from .config3 import (BASELINE_CONDITION, CANARY_CONDITIONS,
                      CONTEXT_LENGTH_THRESHOLDS, K_VALUES, MODELS, PRIMARY_K,
                      RESULTS_DIR, TASK_SETS, TRADITIONAL_SIGNALS,
                      TURN_NUMBER_THRESHOLDS, conditions_for, task_file)
from .judge3 import judge_path
from .runner3 import traj_path

GRADED = ["lag_span", "multi_counter", "multi_counter_heavy", "staircase"]
SCORE_THRESHOLDS = [0.99, 0.67, 0.34, 0.0]  # fire when score <= theta

AXIS = {
    "lag_span": "memory distance",
    "multi_counter": "memory breadth",
    "multi_counter_heavy": "memory breadth (heavy)",
    "chain_checksum": "reasoning composition",
    "interference_twin": "interference",
    "confab_trap": "abstention",
    "sparse_recall": "unrehearsed retention",
    "staircase": "headroom",
    "ensemble": "ensemble (lag+checksum+confab)",
    "ensemble_2of3": "ensemble, 2-of-3 vote",
    "static_trailer": "null control",
}


def load_trajs(task_set, model_name, condition, task_ids):
    out = []
    for tid in task_ids:
        p = traj_path(task_set, model_name, condition, tid)
        if p.exists():
            d = json.loads(p.read_text())
            if d.get("complete"):
                out.append(d)
    return out


def paired_task_ids(task_set, model_name, task_ids):
    sets = [{tr["task_id"] for tr in load_trajs(task_set, model_name, c, task_ids)}
            for c in conditions_for(task_set)]
    return sorted(set.intersection(*sets)) if sets and all(sets) else []


# ------------------------------------------------------------ signal makers

def S_score(traj, theta):
    """First turn with an applicable canary score <= theta."""
    for rec in traj["records"]:
        s = rec.get("canary_score")
        if s is not None and s <= theta:
            return rec["turn"]
    return None


def S_strict(traj):
    return traj["first_canary_fail"]


def S_ensemble_2of3(traj):
    for rec in traj["records"]:
        subs = rec.get("canary_subs") or {}
        failing = sum(1 for s in subs.values() if s is not None and s < 1.0)
        if failing >= 2:
            return rec["turn"]
    return None


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
    c = collections.Counter()
    for tr in trajs:
        if tr["first_hallucination"] is None:
            continue
        for kind, _ in tr["records"][-1]["errors"]:
            c[kind] += 1
    return dict(c.most_common())


def hallu_rate(trajs):
    n = len(trajs)
    return round(sum(1 for tr in trajs if tr["first_hallucination"] is not None) / n, 4) \
        if n else None


def drop_turn1(pairs):
    return [(S, H, hz) for S, H, hz in pairs if H != 1]


# ------------------------------------------------------------------ compute

def compute_set(task_set, model_names):
    task_ids = [t["task_id"] for t in json.loads(task_file(task_set).read_text())]
    paired = {m: paired_task_ids(task_set, m, task_ids) for m in model_names}
    for m in model_names:
        print(f"  [{task_set}] {m}: {len(paired[m])} tasks paired across all "
              f"{len(conditions_for(task_set))} conditions")
    model_names = [m for m in model_names if paired[m]]
    if not model_names:
        print(f"  [{task_set}] no model has a complete paired set yet -- skipped")
        return None

    set_dir = RESULTS_DIR / task_set
    set_dir.mkdir(parents=True, exist_ok=True)
    headline = {}      # (model, signal) -> metrics at PRIMARY_K
    headline_no1 = {}  # same, excluding turn-1 hallucinations
    observer = {}      # (model, condition) -> hallucination rate
    extras = {}        # (model, condition) -> extra notes
    mixes = {}

    canary_conds = [c for c in conditions_for(task_set) if c != BASELINE_CONDITION]

    for cond in canary_conds:
        folder = set_dir / cond
        folder.mkdir(parents=True, exist_ok=True)
        summary = [f"# Experiment 3 canary: {cond}  ({AXIS.get(cond, '')})", ""]
        for m in model_names:
            trajs = load_trajs(task_set, m, cond, paired[m])
            pairs = pairs_for(trajs, S_strict)
            per_k = {str(K): summarize(pairs, K) for K in K_VALUES}
            headline[(m, cond)] = per_k[str(PRIMARY_K)]
            headline_no1[(m, cond)] = summarize(drop_turn1(pairs), PRIMARY_K)
            observer[(m, cond)] = hallu_rate(trajs)

            sweep = {}
            if cond in GRADED:
                for th in SCORE_THRESHOLDS:
                    sp = pairs_for(trajs, lambda tr, th=th: S_score(tr, th))
                    sweep[str(th)] = {str(K): summarize(sp, K) for K in K_VALUES}

            note = {}
            if cond == "confab_trap":
                fab = tot = 0
                for tr in trajs:
                    fc = tr["first_canary_fail"]
                    if fc is None:
                        continue
                    tot += 1
                    rec = next(r for r in tr["records"] if r["turn"] == fc)
                    if rec.get("canary_fabricated"):
                        fab += 1
                note["fabricated_share_of_first_fails"] = round(fab / tot, 3) if tot else None
            if cond == "sparse_recall":
                reh = tot = 0
                for tr in trajs:
                    for rec in tr["records"]:
                        if rec.get("canary_rehearsed") is not None:
                            tot += 1
                            reh += bool(rec.get("canary_rehearsed"))
                note["rehearsal_rate"] = round(reh / tot, 4) if tot else None
            if cond == "ensemble":
                p23 = pairs_for(trajs, S_ensemble_2of3)
                note["ensemble_2of3"] = {str(K): summarize(p23, K) for K in K_VALUES}
                headline[(m, "ensemble_2of3")] = note["ensemble_2of3"][str(PRIMARY_K)]
                headline_no1[(m, "ensemble_2of3")] = summarize(drop_turn1(p23), PRIMARY_K)
            extras[(m, cond)] = note

            (folder / f"metrics_{m}.json").write_text(json.dumps({
                "task_set": task_set, "signal": cond, "axis": AXIS.get(cond),
                "model": m, "kind": MODELS[m]["kind"],
                "metrics_by_K": per_k,
                "metrics_no_turn1_at_primary_K": headline_no1[(m, cond)],
                "score_threshold_sweep": sweep or None,
                "hallucination_rate": observer[(m, cond)],
                "notes": note,
                "per_task": [{"task_id": tr["task_id"], "S": tr["first_canary_fail"],
                              "H": tr["first_hallucination"], "horizon": tr["horizon"]}
                             for tr in trajs],
            }, indent=2))
            summary += [f"## model: {m}, n={len(trajs)} tasks", "", TABLE_HEADER]
            for K in K_VALUES:
                summary.append(fmt_row(f"K={K}", per_k[str(K)]))
            if sweep:
                summary += ["", f"score-threshold sweep at K={PRIMARY_K} "
                            "(fire when score <= theta; strict rule above is theta<1):",
                            TABLE_HEADER]
                for th in SCORE_THRESHOLDS:
                    summary.append(fmt_row(f"theta={th}", sweep[str(th)][str(PRIMARY_K)]))
            if extras[(m, cond)]:
                for k, v in extras[(m, cond)].items():
                    if k != "ensemble_2of3":
                        summary.append(f"\n{k}: {v}")
            summary.append("")
        (folder / "summary.md").write_text("\n".join(summary), encoding="utf-8")

    # ---- traditional signals on the baseline runs
    trad_root = set_dir / "Traditional"
    for sig in TRADITIONAL_SIGNALS:
        (trad_root / sig).mkdir(parents=True, exist_ok=True)
        p = trad_root / sig / "summary.md"
        if p.exists():
            p.unlink()

    for m in model_names:
        trajs = load_trajs(task_set, m, BASELINE_CONDITION, paired[m])
        observer[(m, BASELINE_CONDITION)] = hallu_rate(trajs)
        mixes[m] = error_mix(trajs)

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
            best = max(thresholds, key=lambda th: sweep[str(th)][str(PRIMARY_K)]["f1"])
            headline[(m, f"Traditional/{sig}")] = {**sweep[str(best)][str(PRIMARY_K)],
                                                   "threshold": best}
            headline_no1[(m, f"Traditional/{sig}")] = summarize(
                drop_turn1(pairs_for(trajs, S_maker(best))), PRIMARY_K)
            folder = trad_root / sig
            (folder / f"metrics_{m}.json").write_text(json.dumps({
                "task_set": task_set, "signal": sig, "model": m, "unit": unit,
                "sweep": sweep, "best_threshold_at_primary_K": best}, indent=2))
            lines = [f"# Traditional signal: {sig} (threshold in {unit})",
                     f"\n## model: {m}, n={len(trajs)} tasks, K={PRIMARY_K}",
                     "", TABLE_HEADER]
            for th in thresholds:
                mark = " *best F1*" if th == best else ""
                lines.append(fmt_row(f"th={th}{mark}", sweep[str(th)][str(PRIMARY_K)]))
            with open(folder / "summary.md", "a") as fh:
                fh.write("\n".join(lines) + "\n\n")

        # judge
        pairs, judged, judge_model = [], 0, None
        for tr in trajs:
            jp = judge_path(task_set, m, tr["task_id"])
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
            headline_no1[(m, "Traditional/LLM_judge")] = summarize(
                drop_turn1(pairs), PRIMARY_K)
        folder = trad_root / "LLM_judge"
        (folder / f"metrics_{m}.json").write_text(json.dumps({
            "task_set": task_set, "signal": "LLM_judge", "model": m,
            "judge_model": judge_model, "n_judged": judged,
            "metrics_by_K": per_k}, indent=2))
        lines = [f"# Traditional signal: LLM judge ({judge_model}, "
                 f"last-8-turn window, n judged = {judged})",
                 f"\n## model: {m}, n={len(trajs)} tasks", "", TABLE_HEADER]
        for K in K_VALUES:
            lines.append(fmt_row(f"K={K}", per_k[str(K)]))
        with open(folder / "summary.md", "a") as fh:
            fh.write("\n".join(lines) + "\n\n")

        # random compaction
        per_k = {str(K): summarize_random(trajs, K) for K in K_VALUES}
        headline[(m, "Traditional/random_compaction")] = per_k[str(PRIMARY_K)]
        folder = trad_root / "random_compaction"
        (folder / f"metrics_{m}.json").write_text(json.dumps({
            "task_set": task_set, "signal": "random_compaction", "model": m,
            "metrics_by_K": per_k}, indent=2))
        lines = ["# Traditional signal: random compaction",
                 f"\n## model: {m}, n={len(trajs)} tasks", "", TABLE_HEADER]
        for K in K_VALUES:
            lines.append(fmt_row(f"K={K}", per_k[str(K)]))
        with open(folder / "summary.md", "a") as fh:
            fh.write("\n".join(lines) + "\n\n")

    # ---- per-set summary
    sig_order = ([c for c in CANARY_CONDITIONS if c != "ensemble"] +
                 (["multi_counter_heavy"] if task_set == "coding" else []) +
                 ["ensemble", "ensemble_2of3"] +
                 [f"Traditional/{s}" for s in TRADITIONAL_SIGNALS])
    lines = [f"# Experiment 3 -- task set: {task_set}", "",
             f"Primary prediction window K={PRIMARY_K}. Signal = first turn with an "
             "applicable canary score < 1. Same TP/FP/FN/TN rule as experiments 1-2 "
             "(experiments/metrics.py). Each canary isolates one axis (second column "
             "of the table).", ""]
    for m in model_names:
        lines += [f"\n## model: {m}, n={len(paired[m])} paired tasks, K={PRIMARY_K}", "",
                  "| signal | axis | precision | recall | F1 | fire rate | median lead |",
                  "|---|---|---|---|---|---|---|"]
        for sig in sig_order:
            met = headline.get((m, sig))
            if not met:
                continue
            def f(x):
                return "-" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))
            name = sig + (f" (th={met['threshold']})" if "threshold" in met else "")
            lines.append(f"| {name} | {AXIS.get(sig, '')} | {f(met['precision'])} | "
                         f"{f(met['recall'])} | {f(met['f1'])} | {f(met['fire_rate'])} | "
                         f"{f(met['lead_time_median'])} |")
        lines += ["", f"### excluding turn-1 hallucinations (K={PRIMARY_K})", "",
                  TABLE_HEADER]
        for sig in sig_order:
            met = headline_no1.get((m, sig))
            if met:
                lines.append(fmt_row(sig, met))
        lines += ["", "### observer effect: hallucination rate by condition", "",
                  "| condition | hallucination rate |", "|---|---|"]
        base = observer.get((m, BASELINE_CONDITION))
        lines.append(f"| baseline (no canary) | {base} |")
        for cond in canary_conds:
            r = observer.get((m, cond))
            if r is not None:
                delta = "" if base is None else f" ({'+' if r - base >= 0 else ''}{r - base:.3f})"
                lines.append(f"| {cond} | {r}{delta} |")
        if mixes.get(m):
            lines.append(f"\nFirst-hallucination error mix (baseline runs): {mixes[m]}")
    (set_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  [{task_set}] wrote {set_dir}/SUMMARY.md")
    return {"headline": headline, "models": model_names, "paired": paired}


def compute_all(task_sets=None, model_names=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    task_sets = task_sets or TASK_SETS
    model_names = model_names or list(MODELS)
    results = {}
    for ts in task_sets:
        if task_file(ts).exists():
            r = compute_set(ts, model_names)
            if r:
                results[ts] = r

    # ---- top-level cross-set summary
    lines = ["# Experiment 3 -- Results Summary (axis-isolating canaries x 3 task sets)",
             "",
             "One row per (task set, model, signal) at the primary window "
             f"K={PRIMARY_K}. Full per-set tables (score sweeps, turn-1-excluded, "
             "observer effect) live in results3/<set>/SUMMARY.md.", ""]
    for ts, r in results.items():
        for m in r["models"]:
            lines += [f"\n## {ts} / {m} (n={len(r['paired'][m])})", "",
                      "| signal | axis | precision | recall | F1 | fire rate | median lead |",
                      "|---|---|---|---|---|---|---|"]
            order = ([c for c in CANARY_CONDITIONS if c != "ensemble"] +
                     (["multi_counter_heavy"] if ts == "coding" else []) +
                     ["ensemble", "ensemble_2of3"] +
                     [f"Traditional/{s}" for s in TRADITIONAL_SIGNALS])
            for sig in order:
                met = r["headline"].get((m, sig))
                if not met:
                    continue
                def f(x):
                    return "-" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))
                name = sig + (f" (th={met['threshold']})" if "threshold" in met else "")
                lines.append(f"| {name} | {AXIS.get(sig, '')} | {f(met['precision'])} | "
                             f"{f(met['recall'])} | {f(met['f1'])} | "
                             f"{f(met['fire_rate'])} | {f(met['lead_time_median'])} |")
    (RESULTS_DIR / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {RESULTS_DIR}/SUMMARY.md")
    return results
