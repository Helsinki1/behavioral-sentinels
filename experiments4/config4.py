"""Experiment 4: does a sentinel-triggered reset beat a clock-triggered one?

Experiments 1-3 were PREDICTION benchmarks: does a signal fire before a
hallucination. Experiment 4 is a DEPLOYMENT benchmark: does acting on the
signal produce a better end-task outcome than acting on a schedule, at a
matched intervention budget.

Three things differ from every earlier experiment:

  1. No early stop. Trajectories run the full horizon so that errors can be
     recovered from; "first hallucination" is no longer the end of the story.
  2. The outcome is task accuracy, not a prediction score.
  3. Arms are budget-matched on resets, so the comparison is about WHEN you
     intervene, not HOW OFTEN.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data2"          # reuse the validated experiment-2 coding tasks
RUNS_DIR = ROOT / "runs4"
RESULTS_DIR = ROOT / "results4"

SEED = 4242
N_TASKS = 40                        # small version; paired across every arm

MODELS = {
    "gpt-oss-20b": {
        "provider": "fireworks",
        "model": "accounts/fireworks/models/gpt-oss-20b",
        "base_url": "https://api.fireworks.ai/inference/v1",
        "api_key_env": "FIREWORKS_API_KEY",
        "kind": "open",
        "extra": {"reasoning_effort": "low"},
        "max_tokens": 1600,
        "concurrency": 8,
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "kind": "proprietary",
        "extra": {},
        "max_tokens": 900,
        "concurrency": 6,
    },
}

TEMPERATURE = 0.2

# ---------------------------------------------------------------- the arms
#   carries_sentinel -> the agent is given the escalating_ledger instruction
#   policy           -> what decides a reset
#
# C_prime is the arm the original plan was missing: it carries the sentinel but
# ignores it, resetting on the schedule. Without it, D - C confounds "the
# timing information was useless" with "carrying the probe hurt", because the
# observer effect (experiment 3) means carrying a probe is not free.
ARMS = {
    "A_no_reset":      {"policy": "none",      "carries_sentinel": False},
    "C_scheduled":     {"policy": "scheduled", "carries_sentinel": False},
    "C_prime_carried": {"policy": "scheduled", "carries_sentinel": True},
    "D_sentinel":      {"policy": "sentinel",  "carries_sentinel": True},
    "B_random":        {"policy": "random",    "carries_sentinel": False},
    "F_oracle":        {"policy": "oracle",    "carries_sentinel": False},
    # --- the 2x2 interaction cells (experiment 10, Study B0) ---------------
    # Exp 4 estimated "even a perfect carried probe barely pays" by SUBTRACTING
    # the carrying cost from the timing prize. That assumes the two effects are
    # additive, which was never tested. These two arms complete the factorial:
    #
    #                    no useful reset      oracle-timed reset
    #   no sentinel      A_no_reset  (A)      F_oracle          (B)
    #   carries probe    P_carry_noreset (C)  P_carry_oracle    (D)
    #
    #   C - A = carrying cost          B - A = maximum timing value
    #   D - C = timing value WHILE carrying the probe   <- the unknown
    #   D - A = can a perfect carried sentinel pay for itself at all?
    "P_carry_noreset": {"policy": "none",      "carries_sentinel": True},
    "P_carry_oracle":  {"policy": "oracle",    "carries_sentinel": True},
}

# the 2x2 factorial, in (row, col) order
FACTORIAL = {"A": "A_no_reset", "B": "F_oracle",
             "C": "P_carry_noreset", "D": "P_carry_oracle"}

# The go/no-go gate: run these three first. F_oracle resets at the TRUE first
# hallucination turn, so it upper-bounds what any signal could ever buy. If the
# oracle cannot beat the schedule, no sentinel can, and the full run is wasted.
GATE_ARMS = ["A_no_reset", "C_scheduled", "F_oracle"]

SCHEDULE_EVERY = 6      # C/C': reset every N turns (swept in the dev pass)
SCHEDULE_FIRST = 6
RESET_GRACE = 2         # turns after a reset during which the sentinel is not
                        # allowed to re-trigger (it has just been re-seeded)
MAX_RESETS = 6          # safety cap per trajectory, identical across arms

# primary outcome: fraction of turns with zero hallucination errors
# secondary binary: a trajectory "succeeds" if at least this share are clean
SUCCESS_THRESHOLD = 0.90


def load_env():
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())
