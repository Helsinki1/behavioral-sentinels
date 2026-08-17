"""Offline validation of every experiment-3 checker, across all three task
sets, before any API spend.

1. Synthesises the reply a PERFECT agent would give for every turn of every
   task under every condition and asserts zero hallucination errors and zero
   canary failures (score 1.0 or not-applicable on every turn).
2. Injects wrong-value faults into each canary line and each task answer and
   asserts the right checker fires.
3. Prints payload applicability rates (how often confab_trap / sparse_recall
   can fire at all) as a sanity check on the generator knobs.
"""
import collections
import json

from .canaries3 import STATIC_TRAILER, check_hallucination, score_canary
from .config3 import ENSEMBLE_MEMBERS, TASK_SETS, conditions_for, task_file


# ------------------------------------------------------------ perfect reply

def _task_reply(task_set, task, turn):
    if task_set == "coding":
        wired = {w["fn"]: w["callee"] for w in turn["wire_calls"]}
        body = []
        for fn in turn["defs_required"]:
            params = turn["defs_truth"][fn]
            call = wired.get(fn)
            body.append(f"def {fn}({', '.join(params)}):")
            body.append(f"    return {call}()" if call else "    return None")
        code = ("```python\n" + "\n".join(body) + "\n```") if body else ""
        sigs = [f"SIG {q['fn']} DELETED" if q["deleted"]
                else f"SIG {q['fn']}({', '.join(q['params'])})"
                for q in turn["queries"]]
        return ([code] if code else []) + sigs
    if task_set == "registers":
        return [f"VALUE {reg} = {val}" for reg, val in turn["truth"].items()]
    if task_set == "babi":
        return [f"ANSWER: {turn['answer']}"]
    raise ValueError(task_set)


def _canary_lines(condition, task, turn):
    c3 = turn["c3"]
    tl = task["c3"]
    if condition == "lag_span":
        return [f"ECHO: {c3['lag1']} {c3['lag3']} {c3['lag6']}"]
    if condition == "multi_counter":
        return ["COUNTS: " + " ".join(f"{c}={c3['counts'][c]}" for c in tl["colors"])]
    if condition == "multi_counter_heavy":
        return ["COUNTS: " + " ".join(f"{c}={c3['counts_heavy'][c]}"
                                      for c in tl["colors_heavy"])]
    if condition == "chain_checksum":
        return [f"CHECK: {c3['check']}"]
    if condition == "interference_twin":
        return [f"SHADOW: {c3['shadow_truth']}"]
    if condition == "confab_trap":
        return [f"TAG: {c3['tag_truth']}"] if c3["tag_query"] else []
    if condition == "sparse_recall":
        return [f"AUDIT: {tl['audit_code']}"] if c3["audit_probe"] else []
    if condition == "staircase":
        return ["LEDGER: " + " ".join(c3["stair_expect"])]
    if condition == "ensemble":
        out = []
        for m in ENSEMBLE_MEMBERS:
            out += _canary_lines(m, task, turn)
        return out
    if condition == "static_trailer":
        return [STATIC_TRAILER]
    if condition == "baseline":
        return []
    raise ValueError(condition)


def perfect_reply(task_set, condition, task, turn):
    return "\n".join(_task_reply(task_set, task, turn) + _canary_lines(condition, task, turn))


# ------------------------------------------------------------ perturbations

def perturbed(condition, task, turn, text):
    """Return a wrong-value variant of `text`, or None when not applicable."""
    c3 = turn["c3"]
    tl = task["c3"]
    if condition == "lag_span":
        if c3["lag1"] == "NONE":
            return None  # all slots vacuous -> canary N/A on this turn
        return text.replace(f"ECHO: {c3['lag1']}", f"ECHO: {c3['ticket']}", 1)
    if condition == "multi_counter":
        c = tl["colors"][0]
        return text.replace(f"{c}={c3['counts'][c]}", f"{c}={c3['counts'][c] + 2}", 1)
    if condition == "multi_counter_heavy":
        c = tl["colors_heavy"][0]
        return text.replace(f"{c}={c3['counts_heavy'][c]}",
                            f"{c}={c3['counts_heavy'][c] + 2}", 1)
    if condition == "chain_checksum":
        return text.replace(f"CHECK: {c3['check']}", f"CHECK: {(c3['check'] + 5) % 97}", 1)
    if condition == "interference_twin":
        return text.replace(f"SHADOW: {c3['shadow_truth']}", "SHADOW: ghost_entry", 1)
    if condition == "confab_trap":
        if not c3["tag_query"]:
            return None
        bad = "maple" if c3["tag_truth"] != "maple" else "quartz"
        return text.replace(f"TAG: {c3['tag_truth']}", f"TAG: {bad}", 1)
    if condition == "sparse_recall":
        if not c3["audit_probe"]:
            return None
        return text.replace(f"AUDIT: {tl['audit_code']}", "AUDIT: XK-0000-ZULU", 1)
    if condition == "staircase":
        want = " ".join(c3["stair_expect"])
        bad = " ".join(["99"] + c3["stair_expect"][1:])
        return text.replace(f"LEDGER: {want}", f"LEDGER: {bad}", 1)
    if condition == "ensemble":
        return text.replace(f"CHECK: {c3['check']}", f"CHECK: {(c3['check'] + 5) % 97}", 1)
    if condition == "static_trailer":
        return text.replace(STATIC_TRAILER, "Reviewed-by: Someone Else", 1)
    return None


def perturbed_task(task_set, task, turn, text):
    """Break the TASK part of the reply; the hallucination checker must fire."""
    if task_set == "coding":
        lq = [q for q in turn["queries"] if not q["deleted"]]
        if not lq:
            return None
        q0 = lq[0]
        return text.replace(f"SIG {q0['fn']}({', '.join(q0['params'])})",
                            f"SIG {q0['fn']}({', '.join(q0['params'] + ['phantom'])})", 1)
    if task_set == "registers":
        if not turn["truth"]:
            return None
        reg, val = next(iter(turn["truth"].items()))
        return text.replace(f"VALUE {reg} = {val}", f"VALUE {reg} = {val + 7}", 1)
    if task_set == "babi":
        return text.replace(f"ANSWER: {turn['answer']}", "ANSWER: mailbox", 1)
    return None


# ------------------------------------------------------------------ main

def main():
    failures = 0
    for task_set in TASK_SETS:
        path = task_file(task_set)
        if not path.exists():
            print(f"[{task_set}] SKIP: {path} missing (run python -m experiments3.tasks3)")
            continue
        tasks = json.loads(path.read_text())
        conds = [c for c in conditions_for(task_set) if c != "baseline"]

        fp_hallu = collections.Counter()
        fp_canary = collections.Counter()
        applicable = collections.Counter()
        n_turns = 0
        for task in tasks:
            for turn in task["turns"]:
                n_turns += 1
                for cond in conds:
                    text = perfect_reply(task_set, cond, task, turn)
                    for kind, _ in check_hallucination(task_set, text, turn):
                        fp_hallu[kind] += 1
                    r = score_canary(cond, text, task, turn)
                    if r["score"] is not None:
                        applicable[cond] += 1
                        if r["score"] < 1.0:
                            fp_canary[cond] += 1

        print(f"\n[{task_set}] {len(tasks)} tasks, {n_turns} turns")
        print(f"  FP hallucination on perfect agent: {dict(fp_hallu) or 'none'}")
        print(f"  FP canary on perfect agent       : {dict(fp_canary) or 'none'}")
        print("  applicability (turns where the canary can fire):")
        for cond in conds:
            print(f"    {cond:20s} {applicable[cond]:6d} / {n_turns}")
        failures += sum(fp_hallu.values()) + sum(fp_canary.values())

        # negative controls
        miss = collections.Counter()
        trials = collections.Counter()
        miss_task = trials_task = 0
        for task in tasks[:30]:
            for turn in task["turns"]:
                for cond in conds:
                    good = perfect_reply(task_set, cond, task, turn)
                    bad = perturbed(cond, task, turn, good)
                    if bad is None or bad == good:
                        continue
                    trials[cond] += 1
                    r = score_canary(cond, bad, task, turn)
                    if r["score"] is None or r["score"] >= 1.0:
                        miss[cond] += 1
                base = perfect_reply(task_set, "baseline", task, turn)
                badt = perturbed_task(task_set, task, turn, base)
                if badt and badt != base:
                    trials_task += 1
                    if not check_hallucination(task_set, badt, turn):
                        miss_task += 1
        print("  negative controls (missed injected faults):")
        for cond in conds:
            if trials[cond]:
                print(f"    {cond:20s} {miss[cond]}/{trials[cond]}")
        print(f"    {'task_hallucination':20s} {miss_task}/{trials_task}")
        failures += sum(miss.values()) + miss_task

    print(f"\n{'ALL CHECKS PASS' if failures == 0 else f'{failures} FAILURES'}")
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
