"""Experiment 9: the exp-8 active-vs-passive observation design on a
RESPECTED EXTERNAL BENCHMARK and MULTIPLE MODELS.

Tasks: the `math` split of Microsoft's "LLMs Get Lost in Multi-Turn
Conversation" sharded instructions (arXiv:2505.06120; GSM8K problems split
into 4-12 constraint shards revealed one per turn). To restore the long
horizons the reset/monitoring dynamics need, THREE sharded problems are
concatenated into one session (~17 turns), exactly as the bAbI set strings
multiple stories through one conversation. Ground truth stays fully
decidable: a wrong ANSWER attempt, a premature ANSWER, or a missing/wrong
final ANSWER is a per-turn hallucination, graded against the `#### <number>`
key.

Deviations from the paper's protocol, stated up front: shards are revealed
VERBATIM (no LLM user-simulator paraphrase), one shard per turn, and the
assistant is instructed to reply WAIT until the FINAL shard -- which makes
premature answers (the paper's headline failure mode) decidable per turn.

Arms are exp 8's core seven; the reset operator is R1 reground (briefing +
the current problem's shards so far -- all user-issued content). Four
models, all through the same harness.
"""
from pathlib import Path

from experiments5.config5 import (MAX_RESETS, RESET_GRACE, SCHEDULE_EVERY,
                                  SCHEDULE_FIRST, SEED, SUCCESS_THRESHOLD,
                                  TEMPERATURE, load_env)
from experiments8.config8 import QUIZ_EVERY, QUIZ_FAIL_MIN, QUIZ_FIRST

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data9"
RUNS_DIR = ROOT / "runs9"
RESULTS_DIR = ROOT / "results9"

DOMAIN = "shardmath"
SEED9 = 92929
EPISODES_PER_SESSION = 3
N_SESSIONS = 34

QUIZ_MAX_TOKENS = 300

MODELS = {
    "gpt-oss-120b": {
        "provider": "fireworks",
        "model": "accounts/fireworks/models/gpt-oss-120b",
        "base_url": "https://api.fireworks.ai/inference/v1",
        "api_key_env": "FIREWORKS_API_KEY",
        "kind": "open",
        "extra": {"reasoning_effort": "low"},
        "max_tokens": 700,
        "concurrency": 8,
    },
    "deepseek-v4-flash": {
        "provider": "fireworks",
        "model": "accounts/fireworks/models/deepseek-v4-flash-0731",
        "base_url": "https://api.fireworks.ai/inference/v1",
        "api_key_env": "FIREWORKS_API_KEY",
        "kind": "open",
        "extra": {"reasoning_effort": "none"},
        "max_tokens": 400,
        "concurrency": 8,
    },
    "qwen3p7-plus": {
        "provider": "fireworks",
        "model": "accounts/fireworks/models/qwen3p7-plus",
        "base_url": "https://api.fireworks.ai/inference/v1",
        "api_key_env": "FIREWORKS_API_KEY",
        "kind": "open",
        "extra": {"reasoning_effort": "none"},
        "max_tokens": 400,
        "concurrency": 8,
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "kind": "proprietary",
        "extra": {},
        "max_tokens": 400,
        "concurrency": 6,
    },
}
MODEL_ORDER = ["gpt-oss-120b", "deepseek-v4-flash", "qwen3p7-plus",
               "gpt-4o-mini"]
DEFAULT_MODEL = "gpt-oss-120b"

# active arms carry lag_span: the sharded failure mode is losing constraints
# revealed many turns ago (the paper's "lost in the middle turns"), which is
# exp 3's RETRIEVAL_DISTANCE genre.
ACTIVE_PROBE = "lag_span"

ARMS = {
    "A_no_reset":      {"policy": "none",      "probe": "baseline",
                        "category": "bound"},
    "C_clock":         {"policy": "scheduled", "probe": "baseline",
                        "category": "baseline"},
    "Z_trace":         {"policy": "zerocarry", "probe": "baseline",
                        "category": "passive-observational"},
    "QUIZ":            {"policy": "quiz",      "probe": "baseline",
                        "category": "passive-behavioural"},
    "ACT_carry_clock": {"policy": "scheduled", "probe": ACTIVE_PROBE,
                        "category": "active"},
    "ACT_probe":       {"policy": "probe",     "probe": ACTIVE_PROBE,
                        "category": "active"},
    "F_oracle":        {"policy": "oracle",    "probe": "baseline",
                        "category": "bound"},
}
ARM_ORDER = ["A_no_reset", "C_clock", "ACT_carry_clock", "ACT_probe",
             "Z_trace", "QUIZ", "F_oracle"]
GATE_ARMS = ["A_no_reset", "C_clock"]
ARM_DISPLAY = ["A_no_reset", "C_clock", "ACT_probe", "ACT_carry_clock",
               "QUIZ", "Z_trace", "F_oracle"]
