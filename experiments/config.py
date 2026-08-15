import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"
RESULTS_DIR = ROOT / "results"

SEED = 42
N_TASKS = 200

# Horizon / difficulty ranges (per user: 15-35 turns, varied)
HORIZON_RANGE = (15, 35)
NUM_KEYS_RANGE = (5, 25)

MODELS = {
    "gpt-4o-mini": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "kind": "proprietary",
        "extra": {},
        "max_tokens": 600,
        "concurrency": 24,
    },
    "gpt-oss-20b": {
        "provider": "fireworks",
        "model": "accounts/fireworks/models/gpt-oss-20b",
        "base_url": "https://api.fireworks.ai/inference/v1",
        "api_key_env": "FIREWORKS_API_KEY",
        "kind": "open",
        "extra": {"reasoning_effort": "low"},
        "max_tokens": 1200,   # leaves room for reasoning tokens
        "concurrency": 8,     # Fireworks serverless rate limits bite above this
    },
}

JUDGE_MODEL = "gpt-4o-mini"  # used for the LLM-judge traditional signal
JUDGE_WINDOW = 8             # judge sees the last N turns (cannot recompute full state)

CANARY_CONDITIONS = [
    "say_my_name",
    "remember_fact",
    "format_response",
    "variable_check",
    "early_decision",
    "multi_resolution",
]
BASELINE_CONDITION = "baseline"  # no canary; used for traditional signals
ALL_CONDITIONS = CANARY_CONDITIONS + [BASELINE_CONDITION]

TRADITIONAL_SIGNALS = ["context_length", "turn_number", "LLM_judge", "random_compaction"]

# Prediction windows K (turns). "inf" = canary before hallucination at all.
K_VALUES = [2, 5, 10, "inf"]
PRIMARY_K = 5

# Threshold sweeps for traditional signals
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
