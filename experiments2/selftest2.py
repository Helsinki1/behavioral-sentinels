"""Offline validation of the experiment-2 checkers.

Synthesises the reply a PERFECT agent would give for every turn of every task
and asserts: zero hallucination errors, every canary passes. Then perturbs the
reply in each of the ways the taxonomy names and asserts the right error kind
fires. Catches checker false positives before any money is spent on API calls.
"""
import collections
import json
import random

from .canaries2 import (LEDGER_UPDATES, STATIC_TRAILER, check_canary,
                        check_hallucination)
from .config2 import CANARY_CONDITIONS, DATA_DIR


def perfect_reply(task, turn, condition):
    wired = {w["fn"]: w["callee"] for w in turn["wire_calls"]}
    body = []
    for fn in turn["defs_required"]:
        params = turn["defs_truth"][fn]
        call = wired.get(fn)
        body.append(f"def {fn}({', '.join(params)}):")
        body.append(f"    return {call}()" if call else "    return None")
    for fn in turn["deletes_required"]:
        body.append(f"# deleted: {fn}")
    code = ("```python\n" + "\n".join(body) + "\n```") if body else ""

    sigs = []
    for q in turn["queries"]:
        sigs.append(f"SIG {q['fn']} DELETED" if q["deleted"]
                    else f"SIG {q['fn']}({', '.join(q['params'])})")

    head, tail = [], []
    if condition == "rotating_prefix":
        head.append(f"BUILD {turn['turn']}")
    elif condition == "stochastic_policy":
        tail.append(f"POLICY: {turn['policy_token']}")
    elif condition == "lagged_echo":
        tail.append(f"PREV: {turn['prev_ticket'] or 'NONE'}")
    elif condition == "conditional_rule":
        tail.append(f"FLAG: {turn['flag']}")
    elif condition == "escalating_ledger":
        tail.append("LEDGER: " + " ".join(turn["ledger_expect"]))
    elif condition == "static_trailer":
        tail.append(STATIC_TRAILER)
    return "\n".join(head + ([code] if code else []) + sigs + tail)


def main():
    tasks = json.loads((DATA_DIR / "tasks2.json").read_text())
    rng = random.Random(0)
    fp_hallu = collections.Counter()
    fp_canary = collections.Counter()
    n_turns = 0

    for task in tasks:
        for turn in task["turns"]:
            n_turns += 1
            for cond in CANARY_CONDITIONS:
                text = perfect_reply(task, turn, cond)
                for kind, _ in check_hallucination(text, turn):
                    fp_hallu[kind] += 1
                if not check_canary(cond, text, task, turn, {}):
                    fp_canary[cond] += 1

    print(f"turns checked: {n_turns}")
    print("FALSE POSITIVES on a perfect agent:")
    print("  hallucination:", dict(fp_hallu) or "none")
    print("  canary       :", dict(fp_canary) or "none")

    # --- negative controls: each perturbation must fire the right error kind
    hits = collections.Counter()
    trials = collections.Counter()
    for task in tasks[:40]:
        for turn in task["turns"]:
            base = perfect_reply(task, turn, "rotating_prefix")
            checks = [
                ("no_code_block", base.split("```")[0] + "\n".join(
                    l for l in base.splitlines() if l.startswith("SIG"))),
            ]
            if "return " in base:
                checks.append(("syntax_error", base.replace("return ", "return ((", 1)))
            lq = [q for q in turn["queries"] if not q["deleted"]]
            if lq:
                q0 = lq[0]
                checks.append(("wrong_sig", base.replace(
                    f"SIG {q0['fn']}({', '.join(q0['params'])})",
                    f"SIG {q0['fn']}({', '.join(q0['params'] + ['phantom_arg'])})")))
            if turn["defs_required"]:
                fn = turn["defs_required"][0]
                p = turn["defs_truth"][fn]
                checks.append(("wrong_def_sig",
                               base.replace(f"def {fn}({', '.join(p)}):",
                                            f"def {fn}({', '.join(p + ['bogus_extra'])}):")))
            if turn["wire_calls"]:
                w = turn["wire_calls"][0]
                checks.append(("fabricated_symbol",
                               base.replace(f"return {w['callee']}()", "return ghost_helper_9()")))
            dq = [q for q in turn["queries"] if q["deleted"]]
            if dq:
                checks.append(("fabricated_sig",
                               base.replace(f"SIG {dq[0]['fn']} DELETED",
                                            f"SIG {dq[0]['fn']}(made, up)")))
            for kind, text in checks:
                trials[kind] += 1
                if kind in {k for k, _ in check_hallucination(text, turn)}:
                    hits[kind] += 1

    print("\nNEGATIVE CONTROLS (detection rate of injected faults):")
    for kind in sorted(trials):
        print(f"  {kind:20s} {hits[kind]:5d}/{trials[kind]:<5d} "
              f"{hits[kind]/trials[kind]:.3f}")

    # --- canary negative controls
    print("\nCANARY NEGATIVE CONTROLS (wrong-value detection):")
    for cond in CANARY_CONDITIONS:
        miss = 0
        tot = 0
        for task in tasks[:40]:
            for turn in task["turns"]:
                good = perfect_reply(task, turn, cond)
                bad = good
                if cond == "rotating_prefix":
                    bad = good.replace(f"BUILD {turn['turn']}", f"BUILD {turn['turn'] + 3}", 1)
                elif cond == "stochastic_policy":
                    bad = good.replace(f"POLICY: {turn['policy_token']}", "POLICY: WRONG-99")
                elif cond == "lagged_echo":
                    bad = good.replace(f"PREV: {turn['prev_ticket'] or 'NONE'}",
                                       f"PREV: {turn['ticket']}")  # echoing CURRENT ticket
                elif cond == "conditional_rule":
                    bad = good.replace(f"FLAG: {turn['flag']}",
                                       "FLAG: " + ("NONE" if turn["flag"] == "VOWEL" else "VOWEL"))
                elif cond == "escalating_ledger":
                    bad = good.replace("LEDGER: " + " ".join(turn["ledger_expect"]),
                                       "LEDGER: " + " ".join(turn["ledger_expect"][:-1] + ["99"])) \
                        if len(turn["ledger_expect"]) > 1 else good.replace(
                            "LEDGER: " + turn["ledger_expect"][0], "LEDGER: 999")
                elif cond == "static_trailer":
                    bad = good.replace(STATIC_TRAILER, "Reviewed-by: Someone Else")
                if bad == good:
                    continue
                tot += 1
                if check_canary(cond, bad, task, turn, {}):
                    miss += 1
        print(f"  {cond:20s} missed {miss}/{tot}")


if __name__ == "__main__":
    main()
