"""Experiment-8 analysis: the three claims of the active-vs-passive writeup.

  1  SIGNAL QUALITY  precision/recall (K=5) of every observation method at
     predicting the first hallucination -- quiz scored on the SAME
     full-horizon A_no_reset trajectories as every passive baseline (via the
     shadow pass), the carried probe on its own arms with censoring marked.
  2  DOWNSTREAM GAIN end-task accuracy when each method's signal routes
     R1 re-grounded restarts, all arms paired on the same 90 tasks; reused
     arms are read verbatim from runs6.
  3  COST OF OBSERVATION  the observer-effect accuracy delta
     (ACT_carry_clock - C_clock: carrying the probe at an identical reset
     schedule) and per-arm monitoring tokens (quiz fork tokens; judge calls;
     zero for trace monitors by construction).

Outputs: results8/metrics.json + SUMMARY.md, prediction.json + PREDICTION.md.
"""
import json
import random
import statistics

from experiments.metrics import summarize, summarize_random

from .config8 import (ARMS, ARM_DISPLAY, DOMAINS, REUSED_ARMS, RESULTS_DIR,
                      RUNS6_DIR, SUCCESS_THRESHOLD)
from .run_all8 import load_arm as load_arm8
from .shadow8 import load_shadow
from experiments5.run_all5 import select_tasks

K = 5
TURN_THRESHOLDS = [3, 5, 8, 10, 12, 15, 20, 25]
CTX_THRESHOLDS = [800, 1000, 1500, 2000, 2500, 3000, 4000, 6000]
QUIZ_FAIL_SWEEP = [1, 2, 3]

CATEGORY = {**{a: ARMS[a]["category"] for a in ARMS}, **REUSED_ARMS}

CONTRAST_PAIRS = [
    ("QUIZ", "C_clock", "frozen-state QUIZ vs turn-count clock"),
    ("QUIZ", "C_ctx", "quiz vs context-growth trigger"),
    ("QUIZ", "C_judge", "quiz vs LLM judge (passive-observational)"),
    ("QUIZ", "B_random", "quiz vs random resets"),
    ("QUIZ", "Z_reground", "quiz vs zero-carry trace monitor"),
    ("QUIZ", "G_dense", "quiz vs densest schedule"),
    ("QUIZ", "A_no_reset", "quiz vs never resetting"),
    ("QUIZ", "F_oracle", "quiz vs perfect-timing oracle"),
    ("ACT_probe", "QUIZ", "ACTIVE probe vs passive-behavioural quiz"),
    ("ACT_probe", "C_clock", "active probe vs clock"),
    ("ACT_probe", "Z_reground", "active probe vs zero-carry trace monitor"),
    ("ACT_probe", "A_no_reset", "active probe vs never resetting"),
    ("ACT_carry_clock", "C_clock",
     "OBSERVER-EFFECT COST: carrying the probe at an identical schedule"),
    ("ACT_probe", "ACT_carry_clock", "timing value of the active signal"),
    ("Z_reground", "C_clock", "zero-carry trace monitor vs clock (anchor)"),
    ("Z_reground", "A_no_reset", "zero-carry vs never resetting (anchor)"),
    ("C_clock", "A_no_reset", "clock vs never resetting (anchor)"),
    ("G_dense", "A_no_reset", "densest schedule vs never resetting (anchor)"),
    ("F_oracle", "A_no_reset", "oracle vs never resetting (anchor)"),
]


def pool_ids():
    return {d: [t["task_id"] for t in select_tasks(d)] for d in DOMAINS}


def load_reused(model, domain, arm, task_ids):
    out = {}
    for tid in task_ids:
        p = RUNS6_DIR / model / domain / arm / f"task_{tid:03d}.json"
        if p.exists():
            try:
                d = json.loads(p.read_text())
            except json.JSONDecodeError:
                continue
            if d.get("complete"):
                out[tid] = d
    return out


def load_all(model):
    ids = pool_ids()
    out = {}
    for arm in list(ARMS) + list(REUSED_ARMS):
        loader = load_arm8 if arm in ARMS else load_reused
        recs = {}
        for d in DOMAINS:
            for tid, r in loader(model, d, arm, ids[d]).items():
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
    qp = [r.get("quiz_prompt_tokens", 0) or 0 for r in recs]
    qc = [r.get("quiz_completion_tokens", 0) or 0 for r in recs]
    return {
        "n": len(recs),
        "accuracy": round(statistics.mean(acc), 4),
        "success_rate": round(sum(1 for a in acc if a >= SUCCESS_THRESHOLD) / len(acc), 4),
        "resets_per_task": round(statistics.mean(r["n_resets"] for r in recs), 3),
        "prompt_tokens": round(statistics.mean(r["prompt_tokens"] for r in recs)),
        "completion_tokens": round(statistics.mean(r["completion_tokens"] for r in recs)),
        "quiz_tokens": round(statistics.mean(p + c for p, c in zip(qp, qc))),
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


# ------------------------------------------------------- signal quality (P/R)

def segment(traj):
    cut = traj["reset_turns"][0] if traj["reset_turns"] else None
    return [r for r in traj["records"] if cut is None or r["turn"] < cut]


FIRE = {
    "probe": lambda r: bool(r.get("probe_fail")),
    "zerocarry": lambda r: bool(r.get("zerocarry_fired")),
    "judge": lambda r: bool(r.get("judge_yes")),
    "quiz": lambda r: bool(r.get("quiz_fail")),
}

# trajectory set -> recorded signals; ✂-marked ones trigger the set's resets
SET_SIGNALS = {
    "A_no_reset":      [("zerocarry", "zero-carry trace monitor")],
    "QUIZ":            [("quiz", "frozen-state quiz (live)"),
                        ("zerocarry", "zero-carry trace monitor")],
    "ACT_carry_clock": [("probe", "carried probe (clock-truncated read)"),
                        ("zerocarry", "zero-carry trace monitor")],
    "ACT_probe":       [("probe", "carried probe"),
                        ("zerocarry", "zero-carry trace monitor")],
    "C_judge":         [("judge", "LLM judge"),
                        ("zerocarry", "zero-carry trace monitor")],
}
SELF_TRIGGERED = {"QUIZ": "quiz", "ACT_probe": "probe", "C_judge": "judge"}


def make_items(all_recs, arm):
    items = []
    for (d, tid), traj in sorted(all_recs.get(arm, {}).items()):
        recs = segment(traj)
        if not recs:
            continue
        H = next((r["turn"] for r in recs if r["hallucination"]), None)
        items.append({"domain": d, "tid": tid, "recs": recs, "H": H,
                      "hz": recs[-1]["turn"],
                      "truncated": bool(traj["reset_turns"])})
    return items


def first_fire(recs, key):
    return next((r["turn"] for r in recs if FIRE[key](r)), None)


def score(pairs):
    m = summarize(pairs, K)
    m["leads"] = [H - S for S, H, _ in pairs
                  if S is not None and H is not None and S <= H]
    return m


def best_threshold(items, S_fn, thresholds):
    best = None
    for th in thresholds:
        pairs = [(S_fn(it, th), it["H"], it["hz"]) for it in items]
        m = score(pairs)
        m["threshold"] = th
        if best is None or (m["f1"] or 0) > (best["f1"] or 0):
            best = m
    return best


def ctx_fire(recs, th):
    return next((r["turn"] for r in recs
                 if r.get("prompt_tokens") is not None
                 and r["prompt_tokens"] >= th), None)


def shadow_items(model):
    ids = pool_ids()
    items = []
    for d in DOMAINS:
        for tid, sh in sorted(load_shadow(model, d, ids[d]).items()):
            items.append({"domain": d, "tid": tid, "H": sh["first_hallucination"],
                          "hz": sh["horizon"], "checkpoints": sh["checkpoints"]})
    return items


def shadow_S(item, fail_min):
    return next((c["turn"] for c in item["checkpoints"]
                 if c["n_wrong"] >= fail_min), None)


def prediction(model, all_recs):
    out = {"model": model, "K": K, "sets": {}, "shadow": {}}

    # -- the same-trajectory table: everything scored on A_no_reset ----------
    sh = shadow_items(model)
    if sh:
        by_dom = {d: [it for it in sh if it["domain"] == d] for d in DOMAINS}
        entry = {"n_segments": len(sh), "signals": {}}
        for fail_min in QUIZ_FAIL_SWEEP:
            name = f"frozen-state quiz (shadow, fail>={fail_min})"
            entry["signals"][name] = {
                "pooled": score([(shadow_S(it, fail_min), it["H"], it["hz"])
                                 for it in sh]),
                "by_domain": {d: score([(shadow_S(it, fail_min), it["H"], it["hz"])
                                        for it in v])
                              for d, v in by_dom.items() if v}}
        out["shadow"] = entry

    for arm, signals in SET_SIGNALS.items():
        items = make_items(all_recs, arm)
        if not items:
            continue
        by_dom = {d: [it for it in items if it["domain"] == d] for d in DOMAINS}
        entry = {
            "n_segments": len(items),
            "median_segment_turns": statistics.median(it["hz"] for it in items),
            "share_truncated_by_reset": round(
                sum(it["truncated"] for it in items) / len(items), 3),
            "self_triggered_signal": dict(signals).get(SELF_TRIGGERED.get(arm)),
            "signals": {},
        }

        def all_scopes(name, scorer):
            entry["signals"][name] = {
                "pooled": scorer(items),
                "by_domain": {d: scorer(v) for d, v in by_dom.items() if v}}

        for key, disp in signals:
            all_scopes(disp, lambda its, key=key: score(
                [(first_fire(it["recs"], key), it["H"], it["hz"]) for it in its]))
        if arm == "A_no_reset":
            if sh:
                sh_by = {(it["domain"], it["tid"]): it for it in sh}
                matched = [it for it in items
                           if (it["domain"], it["tid"]) in sh_by]
                all_scopes("frozen-state quiz (shadow)", lambda its: score(
                    [(shadow_S(sh_by[(it["domain"], it["tid"])], 2),
                      it["H"], it["hz"]) for it in its
                     if (it["domain"], it["tid"]) in sh_by]))
            all_scopes("turn_number", lambda its: best_threshold(
                its, lambda it, th: th if it["hz"] >= th else None,
                TURN_THRESHOLDS))
            all_scopes("context_length", lambda its: best_threshold(
                its, lambda it, th: ctx_fire(it["recs"], th), CTX_THRESHOLDS))
            all_scopes("random (expected)", lambda its: {
                **summarize_random(
                    [{"first_hallucination": it["H"], "horizon": it["hz"]}
                     for it in its], K), "leads": []})
        out["sets"][arm] = entry
    return out


# --------------------------------------------------------------- reporting

def _f(x):
    return "-" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def _row(name, m, mark=""):
    th = f" (th={m['threshold']})" if "threshold" in m else ""
    return (f"| {name}{th}{mark} | {_f(m['precision'])} | {_f(m['recall'])} | "
            f"{_f(m['f1'])} | {_f(m['fire_rate'])} | {_f(m['lead_time_median'])} "
            f"| {m['n']} |")


PRED_HEADER = ("| signal | precision | recall | F1 | fire rate | median lead | n |\n"
               "|---|---|---|---|---|---|---|")

SET_NOTES = {
    "A_no_reset": "never resets -- the same-trajectory table: every signal "
                  "(quiz via the shadow pass) on identical full-horizon "
                  "trajectories. THE Fig-2 source.",
    "QUIZ": "the quiz triggers this arm's resets: its live lead is "
            "right-censored at the fire",
    "ACT_carry_clock": "the CLOCK ends segments (first reset at turn 6), so "
                       "the carried probe's fires are not self-censored -- "
                       "the clean active-signal read, on observer-shifted "
                       "trajectories by necessity",
    "ACT_probe": "the probe triggers resets: its lead is right-censored",
    "C_judge": "the judge triggers resets: its lead is right-censored",
}


def write_prediction(pred):
    L = [f"# Experiment 8 — signal quality of every observation method (K={K})",
         "",
         f"Model `{pred['model']}`. S = first fire, H = first hallucination "
         "in the pre-first-reset segment; TP within K=5 turns "
         "(experiments/metrics.py rule). Quiz checkpoints occur every 3 "
         "turns, so quiz S has 3-turn granularity. Thresholded baselines are "
         "tuned to best F1 on the evaluation set itself (maximally generous "
         "to them).", ""]
    if pred.get("shadow", {}).get("signals"):
        L += ["## Quiz fail-threshold ablation (shadow pass, full-horizon "
              "A_no_reset trajectories)", "", PRED_HEADER]
        for name, s in pred["shadow"]["signals"].items():
            L.append(_row(name, s["pooled"]))
        L.append("")
    for arm, e in pred["sets"].items():
        L += [f"## Trajectory set `{arm}` — {SET_NOTES.get(arm, '')}", "",
              f"{e['n_segments']} segments, median "
              f"{e['median_segment_turns']} turns, "
              f"{e['share_truncated_by_reset']:.0%} truncated by a reset."
              + (f" Self-triggered signal: `{e['self_triggered_signal']}` ✂."
                 if e.get("self_triggered_signal") else ""), "", PRED_HEADER]
        for name, s in e["signals"].items():
            mark = " ✂" if name == e.get("self_triggered_signal") else ""
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
    L += ["✂ = the signal that triggers this set's resets: its lead time is "
          "right-censored at the first fire.", ""]
    p = RESULTS_DIR / "PREDICTION.md"
    p.write_text("\n".join(L), encoding="utf-8")
    print("wrote", p)


def compute(model="gpt-oss-20b"):
    all_recs = load_all(model)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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

    pred = prediction(model, all_recs)

    out = {"model": model, "n_tasks": len(common),
           "per_domain_n": {d: len(by_domain[d]) for d in DOMAINS},
           "arms": stats, "arms_by_domain": dstats,
           "categories": CATEGORY,
           "contrasts": contrasts, "contrasts_by_domain": dcontrasts}
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(out, indent=2))
    (RESULTS_DIR / "prediction.json").write_text(json.dumps(pred, indent=2))
    write_summary(out)
    write_prediction(pred)
    return out, pred


def write_summary(out):
    stats = out["arms"]
    L = ["# Experiment 8 — active vs passive observation, deployed", "",
         f"Model {out['model']} — {out['n_tasks']} tasks paired across every "
         f"arm ({', '.join(f'{v} {k}' for k, v in out['per_domain_n'].items())}), "
         "full horizon, no early stop, reset operator = R1 reground for every "
         "resetting arm. New arms (QUIZ, ACT_probe, ACT_carry_clock) from "
         "runs8; every other arm read verbatim from runs6.", "",
         "Categories: **active** = the observation writes into the agent's "
         "trajectory (carried probe); **passive-behavioural** = frozen-state "
         "quiz on a discarded fork (extra tokens, zero contamination); "
         "**passive-observational** = reads the existing trace only.", "",
         "## Arms", "",
         "| arm | category | policy | accuracy | success@0.9 | resets/task "
         "| prompt tok | quiz tok |",
         "|---|---|---|---|---|---|---|---|"]
    for a in [x for x in ARM_DISPLAY if x in stats]:
        s = stats[a]
        pol = ARMS[a]["policy"] if a in ARMS else "(runs6)"
        L.append(f"| {a} | {CATEGORY.get(a, '-')} | {pol} "
                 f"| {s['accuracy']:.3f} | {s['success_rate']:.3f} "
                 f"| {s['resets_per_task']:.2f} | {s['prompt_tokens']:,} "
                 f"| {s['quiz_tokens']:,} |")

    L += ["", "## Per-domain accuracy", "",
          "| arm | " + " | ".join(DOMAINS) + " |",
          "|---|" + "---|" * len(DOMAINS)]
    for a in [x for x in ARM_DISPLAY if x in out["arms_by_domain"]]:
        cells = []
        for d in DOMAINS:
            s = out["arms_by_domain"][a].get(d)
            cells.append(f"{s['accuracy']:.3f}" if s else "—")
        L.append(f"| {a} | " + " | ".join(cells) + " |")

    L += ["", "## Paired contrasts, pooled (bootstrap 95% CI on the per-task "
          "delta)", "",
          "| contrast | delta accuracy | 95% CI | significant | "
          "better/worse/tied |", "|---|---|---|---|---|"]
    for c in out["contrasts"]:
        L.append(f"| {c['contrast']} | {c['mean_delta']:+.4f} | "
                 f"[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] | "
                 f"{'**yes**' if c['significant'] else 'no'} | "
                 f"{c['tasks_better']}/{c['tasks_worse']}/{c['tasks_tied']} |")

    L += ["", "## Key contrasts per domain", "",
          "| domain | contrast | delta | 95% CI | sig |", "|---|---|---|---|---|"]
    keep = {("QUIZ", "C_clock"), ("QUIZ", "Z_reground"), ("QUIZ", "C_judge"),
            ("ACT_probe", "QUIZ"), ("ACT_probe", "C_clock"),
            ("ACT_carry_clock", "C_clock"), ("QUIZ", "F_oracle"),
            ("QUIZ", "A_no_reset")}
    for c in out["contrasts_by_domain"]:
        if (c["a"], c["b"]) in keep:
            L.append(f"| {c['domain']} | {c['a']} - {c['b']} | "
                     f"{c['mean_delta']:+.4f} | "
                     f"[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] | "
                     f"{'**yes**' if c['significant'] else 'no'} |")

    L += ["", "## Cost of observation", "",
          "The observer-effect accuracy delta is the `ACT_carry_clock - "
          "C_clock` contrast above (same trigger, same operator; the only "
          "difference is that the agent carries the probe). Monitoring "
          "tokens: `quiz tok` column for QUIZ (fork tokens, never in agent "
          "context); the judge's calls are folded into C_judge's totals; "
          "trace monitors cost 0 by construction."]

    (RESULTS_DIR / "SUMMARY.md").write_text("\n".join(L) + "\n",
                                            encoding="utf-8")
    print("\n".join(L).encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    compute()
