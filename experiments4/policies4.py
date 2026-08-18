"""Reset policies. Each returns decide(state)->bool, called before each turn.

Every policy is capped at MAX_RESETS so no arm can win by intervening more,
and the sentinel arm additionally observes a grace period after each reset
(it has just been re-seeded, so an immediate re-trigger would be an artifact).
"""
import random

from .config4 import MAX_RESETS, RESET_GRACE, SCHEDULE_EVERY, SCHEDULE_FIRST


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


def sentinel_policy(task, **kw):
    """Reset when the canary failed on the previous turn."""
    def decide(s):
        if s["n_resets"] >= MAX_RESETS or s["turns_since_reset"] < RESET_GRACE:
            return False
        prev = s["records"][-1] if s["records"] else None
        return bool(prev and prev.get("canary_pass") is False)
    return decide


def random_policy(task, n_resets=0, seed=0, horizon=None, **kw):
    """Fire at n_resets turns drawn uniformly -- budget-matched per task to the
    sentinel arm, so the contrast is timing, not frequency."""
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
    that hallucinated first in the no-reset arm. Nothing that reads behaviour
    can beat this, so if it fails to beat the schedule, no sentinel can."""
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
            "sentinel": sentinel_policy, "random": random_policy,
            "oracle": oracle_policy}
