"""Reset policies. Each returns decide(state)->bool, called before each turn.

Every policy is capped at MAX_RESETS so no arm can win by intervening more,
and every behavioural trigger (probe, zero-carry, judge) observes a grace
period after each reset: the state was just re-seeded, so an immediate
re-trigger would be an artifact of the reset itself.
"""
import random

from .config5 import (CTX_GROWTH_TOKENS, MAX_RESETS, RESET_GRACE,
                      SCHEDULE_EVERY, SCHEDULE_FIRST)


def none_policy(task, **kw):
    return lambda s: False


def scheduled_policy(task, every=SCHEDULE_EVERY, first=SCHEDULE_FIRST, **kw):
    def decide(s):
        if s["n_resets"] >= MAX_RESETS:
            return False
        if s["n_resets"] == 0:
            return s["turn"] >= first
        return s["turns_since_reset"] >= every
    return decide


def ctx_growth_policy(task, threshold=CTX_GROWTH_TOKENS, **kw):
    """Reset when the prompt has grown by `threshold` tokens since the first
    turn after the last reset (or since the start). This is the deployed
    'compact when the context budget is spent' heuristic in relative form."""
    def decide(s):
        if s["n_resets"] >= MAX_RESETS or s["ctx_baseline"] is None:
            return False
        return s["last_prompt_tokens"] - s["ctx_baseline"] >= threshold
    return decide


def _behavioural(fire_fn):
    def decide(s):
        if s["n_resets"] >= MAX_RESETS or s["turns_since_reset"] < RESET_GRACE:
            return False
        prev = s["records"][-1] if s["records"] else None
        return bool(prev and fire_fn(prev))
    return decide


def probe_policy(task, **kw):
    """Reset when the carried probe failed on the previous turn."""
    return _behavioural(lambda prev: prev.get("probe_fail"))


def zerocarry_policy(task, **kw):
    """Reset when the zero-carry self-consistency monitor fired on the
    previous turn."""
    return _behavioural(lambda prev: prev.get("zerocarry_fired"))


def judge_policy(task, **kw):
    """Reset when the LLM judge said YES on the previous turn."""
    return _behavioural(lambda prev: prev.get("judge_yes"))


def random_policy(task, n_resets=0, seed=0, horizon=None, **kw):
    """Fire at n_resets turns drawn uniformly -- budget-matched per task to
    D_routed, so the contrast is timing, not frequency."""
    rng = random.Random((seed << 16) ^ task["task_id"])
    h = horizon or task["horizon"]
    turns = sorted(rng.sample(range(2, h + 1), min(n_resets, max(0, h - 1))))
    fired = set()

    def decide(s):
        if s["n_resets"] >= MAX_RESETS:
            return False
        if s["turn"] in turns and s["turn"] not in fired:
            fired.add(s["turn"])
            return True
        return False
    return decide


def oracle_policy(task, oracle_turn=None, **kw):
    """Upper bound: a PERFECT predictor. Resets immediately before the turn
    that hallucinated first in the no-reset arm."""
    fired = []

    def decide(s):
        if oracle_turn is None or fired or s["n_resets"] >= MAX_RESETS:
            return False
        if s["turn"] >= oracle_turn:
            fired.append(s["turn"])
            return True
        return False
    return decide


POLICIES = {"none": none_policy, "scheduled": scheduled_policy,
            "ctx_growth": ctx_growth_policy, "probe": probe_policy,
            "zerocarry": zerocarry_policy, "judge": judge_policy,
            "random": random_policy, "oracle": oracle_policy}
