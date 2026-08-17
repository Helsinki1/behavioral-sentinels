"""Experiment 3 configuration: axis-isolating canaries across THREE task sets.

EXPERIMENT 1 (experiments/)  : register book-keeping, STATIC canaries.
EXPERIMENT 2 (experiments2/) : incremental coding, DYNAMIC canaries.
                               Finding: what matters is that the canary answer
                               cannot be COPIED from the model's own last reply.
EXPERIMENT 3 (experiments3/) : non-copyability is now a prerequisite, not a
                               variable. Each canary isolates ONE cognitive
                               axis (memory distance, memory breadth, reasoning
                               composition, interference, abstention, unrehearsed
                               retention, escalating headroom), replicated over
                               three task domains:

  coding    -- the experiment-2 incremental-coding tasks (data2/tasks2.json),
               augmented with experiment-3 canary payloads.  N=200.
  registers -- the experiment-1 register book-keeping tasks (data/tasks.json),
               same augmentation.  N=100.
  babi      -- Facebook's published bAbI QA benchmark (tasks 1-3, en-valid-10k
               items verbatim via the Muennighoff/babi jsonl mirror), wrapped
               into long sessions: 3-5 consecutive stories separated by an
               explicit "NEW STORY" reset, one question per turn.  N=100.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data3"
RAW_DIR = DATA_DIR / "raw"
RUNS_DIR = ROOT / "runs3"
RESULTS_DIR = ROOT / "results3"

SEED = 271828

TASK_SETS = ["coding", "registers", "babi"]
N_TASKS = {"coding": 200, "registers": 100, "babi": 100}

# ---------------------------------------------------------------- models
MODELS = {
    "gpt-4o-mini": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "kind": "proprietary",
        "extra": {},
        "max_tokens": 900,
        "concurrency": 6,   # rolling RPD window; see results2/FINDINGS.md
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
    # third rung of the size ladder (8B < 20B < 4o-mini); opt-in via --models
    "llama-v3p1-8b": {
        "provider": "fireworks",
        "model": "accounts/fireworks/models/llama-v3p1-8b-instruct",
        "base_url": "https://api.fireworks.ai/inference/v1",
        "api_key_env": "FIREWORKS_API_KEY",
        "kind": "open-small",
        "extra": {},
        "max_tokens": 1200,
        "concurrency": 8,
    },
}
# default arm: the model with a complete exp-1/exp-2 record and no RPD ceiling
DEFAULT_MODELS = ["gpt-oss-20b"]

JUDGE_MODEL = os.environ.get("SENTINEL_JUDGE_MODEL", "gpt-4o-mini")
JUDGE_WINDOW = 8

# ---------------------------------------------------------------- canaries
# One axis per canary.  All non-copyable by construction.
CANARY_CONDITIONS = [
    "lag_span",            # memory DISTANCE: echo tickets from 1, 3 and 6 turns ago
    "multi_counter",       # memory BREADTH: 3 sparse event counters
    "chain_checksum",      # reasoning COMPOSITION: running (sum + key) mod M
    "interference_twin",   # INTERFERENCE: shadow items with look-alike names
    "confab_trap",         # ABSTENTION: tag queries where the true answer is NONE
    "sparse_recall",       # UNREHEARSED RETENTION: fact probed only at rare turns
    "staircase",           # HEADROOM: ledger whose rule gains a field every P turns
    "ensemble",            # lag_span + chain_checksum + confab_trap in one run
    "static_trailer",      # null control (experiment-1-style fixed string)
]
# the load-titration control runs on the coding set only
CODING_ONLY_CONDITIONS = ["multi_counter_heavy"]

BASELINE_CONDITION = "baseline"

def conditions_for(task_set):
    extra = CODING_ONLY_CONDITIONS if task_set == "coding" else []
    return CANARY_CONDITIONS + extra + [BASELINE_CONDITION]

ENSEMBLE_MEMBERS = ["lag_span", "chain_checksum", "confab_trap"]

# ---- difficulty knobs (per-canary; tune from a --pilot fire-rate report)
LAGS = [1, 3, 6]                 # lag_span slots
N_COUNTERS = 3                   # multi_counter colors (light)
N_COUNTERS_HEAVY = 4             # multi_counter_heavy colors
EVENT_P = 0.45                   # P(an EVENT line appears on a turn)
EVENT_P_HEAVY = 0.60
CHECKSUM_MOD = 97                # chain_checksum modulus
SHADOW_SLOTS = 3                 # interference_twin shadow items
SHADOW_RENAME_P = 0.25
TAG_NOTE_P = 0.35                # confab_trap: P(current ticket gets a tag)
TAG_QUERY_P = 0.30               # confab_trap: P(a past ticket is queried), t>=3
N_AUDIT_PROBES = 4               # sparse_recall probes per trajectory
STAIR_PERIOD = 4                 # staircase: a new ledger field every P turns

TRADITIONAL_SIGNALS = ["context_length", "turn_number", "LLM_judge", "random_compaction"]

K_VALUES = [2, 5, 10, "inf"]
PRIMARY_K = 5

CONTEXT_LENGTH_THRESHOLDS = [1000, 2000, 3000, 4000, 6000, 8000, 12000, 16000]
TURN_NUMBER_THRESHOLDS = [3, 5, 8, 10, 12, 15, 20, 25]

TEMPERATURE = 0.2


def task_file(task_set):
    return DATA_DIR / f"tasks3_{task_set}.json"


def load_env():
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())
