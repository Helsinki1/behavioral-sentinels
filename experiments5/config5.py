"""Experiment 5: TASK-CONDITIONED SENTINEL ROUTING, tested as a deployment.

Experiments 1-3 established (a) a probe only fires if its answer cannot be
copied from the model's own previous reply, (b) which axis of strain degrades
first DEPENDS ON THE DOMAIN (exp 3: staircase wins on coding, chain_checksum
on registers, lag_span on babi), and (c) carrying a probe costs what it
measures. Experiment 4 priced the whole trade on one domain: a perfect
predictor beats the clock by +0.059 accuracy, but carrying the (blanket)
probe costs -0.043, so a blanket sentinel nets nothing.

Experiment 5 attacks the two bottlenecks that analysis exposed:

  1. SIGNAL-FAILURE COUPLING -> route the sentinel. Classify the active
     task's dominant state demand (artifact accumulation / update application
     / retrieval over distance / interference tracking / evidence abstention)
     and carry only the ONE probe that stresses the same mechanism the task
     will fail by. The router is a real system component (one LLM call per
     task), not an oracle: its label is used as-is, and its mistakes are part
     of the measured system.
  2. CARRYING COST -> a routed ZERO-CARRY arm. Monitors read degradation off
     output the agent already produces (self-contradicted signatures,
     re-query inconsistencies, answers not grounded in the active story), so
     the observer effect is zero by construction.

Like experiment 4 this is a DEPLOYMENT benchmark: full horizon, no early
stop, outcome = per-turn task accuracy, arms budget-capped identically, and
the routing claim is controlled twice (blanket probe = the exp-1..4 approach,
rotated probe = same probes deliberately mis-assigned).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data3"          # every domain already carries every probe's payloads
RUNS_DIR = ROOT / "runs5"
RESULTS_DIR = ROOT / "results5"

SEED = 52525

# ---------------------------------------------------------------- task pool
# A mixed pool is the point: with one domain there is nothing to route.
DOMAINS = ["coding", "registers", "babi"]
N_PER_DOMAIN = 30                  # difficulty-stratified sample per domain

MODELS = {
    "gpt-oss-20b": {
        "provider": "fireworks",
        "model": "accounts/fireworks/models/gpt-oss-20b",
        "base_url": "https://api.fireworks.ai/inference/v1",
        "api_key_env": "FIREWORKS_API_KEY",
        "kind": "open",
        "extra": {"reasoning_effort": "low"},
        "max_tokens": 1600,
        "concurrency": 12,
    },
    # cross-model arm; rolling-RPD-limited (see results2/FINDINGS.md), run last
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
DEFAULT_MODEL = "gpt-oss-20b"

TEMPERATURE = 0.2

# ---------------------------------------------------------------- compaction
# Structured, validated state snapshot (schema per domain, see harness5).
# Larger than a task reply: it must carry the full artifact plus every
# durable conversation constraint.
COMPACTION_MAX_TOKENS = 2600
COMPACTION_ATTEMPTS = 2

# ---------------------------------------------------------------- judge
JUDGE_MODEL = "gpt-oss-20b"        # same family as exp-2's strongest judge run
JUDGE_WINDOW = 8

# ---------------------------------------------------------------- the arms
#   probe: "none" | "routed" | "labeled" | "blanket" | "rotated" -> what is carried
#          ("labeled" = deterministic pre-labeled routing straight from
#           INTENDED_GENRE, no router call: the router-noise-free upper bound)
#   policy: what decides a reset
ARMS = {
    "A_no_reset":     {"policy": "none",      "probe": "none"},
    "B_random":       {"policy": "random",    "probe": "none"},
    "C_clock":        {"policy": "scheduled", "probe": "none"},
    "C_ctx":          {"policy": "ctx_growth","probe": "none"},
    "C_judge":        {"policy": "judge",     "probe": "none"},
    "C_prime_routed": {"policy": "scheduled", "probe": "routed"},
    "D_routed":       {"policy": "probe",     "probe": "routed"},
    "D_labeled":      {"policy": "probe",     "probe": "labeled"},
    "D_blanket":      {"policy": "probe",     "probe": "blanket"},
    "D_rotated":      {"policy": "probe",     "probe": "rotated"},
    "Z_routed":       {"policy": "zerocarry", "probe": "none"},
    "F_oracle":       {"policy": "oracle",    "probe": "none"},
}

# go/no-go gate: if D_routed cannot at least approach C_clock here, stop early
GATE_ARMS = ["A_no_reset", "C_clock", "D_routed"]

# dependency order for a full run
ARM_ORDER = ["A_no_reset", "C_clock", "D_routed", "Z_routed", "C_ctx",
             "C_judge", "C_prime_routed", "D_blanket", "D_rotated",
             "D_labeled", "B_random", "F_oracle"]

SCHEDULE_EVERY = 6       # clock cadence (exp-4 continuity)
SCHEDULE_FIRST = 6
CTX_GROWTH_TOKENS = 600  # ctx trigger: prompt grew this much since last reset.
                         # Coding context grows ~90 tok/turn (runs4 baseline),
                         # so this matches the clock's ~6-turn cadence; the
                         # absolute-threshold form collapsed on babi in exp 3.
RESET_GRACE = 2          # turns after a reset during which behavioural
                         # triggers may not re-fire (state was just re-seeded)
MAX_RESETS = 6           # identical cap in every arm

SUCCESS_THRESHOLD = 0.90 # binary success = at least this share of clean turns

BLANKET_PROBE = "staircase"   # the exp-2/4 escalating-ledger axis, everywhere

# anti-routing control: the same three probes, deliberately mis-assigned
ROTATED_TABLE = {"coding": "chain_checksum", "registers": "lag_span",
                 "babi": "staircase"}

# what the router SHOULD say; scores the router, and drives the D_labeled arm
# (deterministic routing) -- the LLM-routed arms never read it
INTENDED_GENRE = {"coding": "ARTIFACT_ACCUMULATION",
                  "registers": "UPDATE_APPLICATION",
                  "babi": "RETRIEVAL_DISTANCE"}


def load_env():
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())
