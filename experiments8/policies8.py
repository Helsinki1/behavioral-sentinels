"""Experiment-8 reset policies: exp 5's table plus the quiz trigger.

The quiz policy is behavioural like the probe/zero-carry/judge triggers:
reset before turn t+1 when the checkpoint after turn t failed, subject to the
same MAX_RESETS cap and RESET_GRACE window as every other behavioural arm.
"""
from experiments5.policies5 import POLICIES as POLICIES5, _behavioural


def quiz_policy(task, **kw):
    """Reset when the most recent turn's frozen-state quiz failed."""
    return _behavioural(lambda prev: bool(prev.get("quiz_fail")))


POLICIES = {**POLICIES5, "quiz": quiz_policy}
