"""Experiment 2 configuration: coding tasks + dynamic canaries.

Deliberately parallel to experiments/config.py so results are comparable,
but every path is namespaced (data2/ runs2/ results2/) and the canary set
is disjoint from experiment 1's static set.

EXPERIMENT 1 (experiments/)   : state book-keeping task, STATIC canaries.
EXPERIMENT 2 (experiments2/)  : incremental CODING task, DYNAMIC canaries.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data2"
RUNS_DIR = ROOT / "runs2"
RESULTS_DIR = ROOT / "results2"

SEED = 1729
N_TASKS = 200

# Coding trajectories are heavier per turn than register book-keeping, so the
# horizon band is slightly tighter than experiment 1's (15-35).
HORIZON_RANGE = (12, 30)
NUM_FNS_RANGE = (4, 14)

# Same two models as experiment 1: one proprietary, one open.
MODELS = {
    "gpt-4o-mini": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "kind": "proprietary",
        "extra": {},
        "max_tokens": 900,
        # kept low: the OpenAI account's requests-per-day allowance is a rolling
        # window, and bursting at 24 exhausts it mid-run
        "concurrency": 6,
    },
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
}

# Experiment 1 judged with gpt-4o-mini. Experiment 2 falls back to the open
# model when the OpenAI account has no rolling-window budget left; the judge is
# a *baseline* signal, so which model played judge is recorded alongside the
# results (results2/Traditional/LLM_judge/metrics_*.json -> "judge_model").
JUDGE_MODEL = os.environ.get("SENTINEL_JUDGE_MODEL", "gpt-4o-mini")
JUDGE_WINDOW = 8

# --- the experiment-2 canary set -------------------------------------------
# All six are DYNAMIC: the correct canary output changes over the trajectory
# (except static_trailer, kept deliberately as the experiment-1-style anchor).
CANARY_CONDITIONS = [
    "rotating_prefix",     # content changes EVERY turn (self-incremented counter)
    "stochastic_policy",   # content changes at RANDOM turns, held in between
    "lagged_echo",         # must echo the PREVIOUS turn's ticket (one-turn lag)
    "conditional_rule",    # per-turn conditional branch on the current instruction
    "escalating_ledger",   # requirements accumulate every 4 turns (multi-resolution)
    "static_trailer",      # fixed string every turn (experiment-1-style control)
]
BASELINE_CONDITION = "baseline"
ALL_CONDITIONS = CANARY_CONDITIONS + [BASELINE_CONDITION]

TRADITIONAL_SIGNALS = ["context_length", "turn_number", "LLM_judge", "random_compaction"]

K_VALUES = [2, 5, 10, "inf"]
PRIMARY_K = 5

CONTEXT_LENGTH_THRESHOLDS = [1000, 2000, 3000, 4000, 6000, 8000, 12000, 16000]
TURN_NUMBER_THRESHOLDS = [3, 5, 8, 10, 12, 15, 20, 25]

TEMPERATURE = 0.2


def load_env():
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())
