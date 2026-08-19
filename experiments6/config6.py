"""Experiment 6: SENTINEL-TRIGGERED RE-GROUNDING -- resets that read state
back from an external store, the faithful test of "when should you start a
new Claude Code session?".

Experiments 4-5 tested resets whose operator was COMPACTION: the agent's own
snapshot of state that lives only in context. That operator is maximally
lossy -- a reset can canonicalise the agent's errors -- and exp 5 showed it,
not the signal, had become the bottleneck (even the oracle lost to never
resetting). But the original motivating scenario has an external ground
truth: a fresh Claude Code session RE-READS the repo. Exp 6 tests that
regime with two reset operators, both deterministic (no LLM call at reset):

  R1 "reground": the conversation is replaced by the original briefing with
     the CURRENT true state substituted -- module source / register lines /
     current story -- materialised by a harness reducer that applies each
     user instruction, exactly as a file system applies edits. Every byte is
     derivable from prior user messages; nothing the agent got wrong is
     repaired beyond what re-reading the store would fix. store6.py's
     verifier proves store == generator truth on every turn of every task.

  R2 "replay": the conversation is replaced by the verbatim log of all prior
     user messages (assistant turns dropped). Zero harness intelligence --
     the conservative bracket for anyone who calls R1 oracle-feeding.

Same protocol as exp 5 (full horizon, no early stop, per-turn accuracy,
6-reset cap, 2-turn grace, same 90-task pool); A_no_reset is imported
verbatim from runs5, enabling cross-experiment pairing of operator effects
at a fixed trigger. All arms carry NO probe (exp 5's D_labeled closed that
design: carried probes lose on intrinsic carrying cost).
"""
from pathlib import Path

# protocol constants shared verbatim with exp 5 -- identity matters for the
# cross-experiment operator contrasts
from experiments5.config5 import (CTX_GROWTH_TOKENS, DOMAINS, JUDGE_MODEL,
                                  JUDGE_WINDOW, MAX_RESETS, MODELS,
                                  N_PER_DOMAIN, RESET_GRACE, SCHEDULE_EVERY,
                                  SCHEDULE_FIRST, SEED, SUCCESS_THRESHOLD,
                                  TEMPERATURE, load_env)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data3"
RUNS_DIR = ROOT / "runs6"
RUNS5_DIR = ROOT / "runs5"          # A_no_reset (and exp-5 arms) live here
RESULTS_DIR = ROOT / "results6"

DEFAULT_MODEL = "gpt-oss-20b"

# G_dense: the "restart freely" ceiling probe -- the densest schedule the
# 6-reset cap allows on these horizons (~21 turns): every 3 turns.
DENSE_EVERY = 3
DENSE_FIRST = 3

# ---------------------------------------------------------------- the arms
#   operator: "reground" (R1) | "replay" (R2) | None (never resets)
ARMS = {
    "A_no_reset":      {"policy": "none",       "operator": None},
    "B_random":        {"policy": "random",     "operator": "reground"},
    "C_clock":         {"policy": "scheduled",  "operator": "reground"},
    "C_ctx":           {"policy": "ctx_growth", "operator": "reground"},
    "C_judge":         {"policy": "judge",      "operator": "reground"},
    "Z_reground":      {"policy": "zerocarry",  "operator": "reground"},
    "F_oracle":        {"policy": "oracle",     "operator": "reground"},
    "G_dense":         {"policy": "scheduled",  "operator": "reground",
                        "extras": {"every": DENSE_EVERY, "first": DENSE_FIRST}},
    "Z_replay":        {"policy": "zerocarry",  "operator": "replay"},
    "C_clock_replay":  {"policy": "scheduled",  "operator": "replay"},
    "F_oracle_replay": {"policy": "oracle",     "operator": "replay"},
}

# go/no-go: does a PERFECT predictor beat never resetting once the reset
# operator is loss-free? If not, context rot in this pool is not recoverable
# by restart and the rest of the experiment is moot.
GATE_ARMS = ["A_no_reset", "C_clock", "F_oracle"]

ARM_ORDER = ["A_no_reset", "C_clock", "F_oracle", "Z_reground", "C_ctx",
             "C_judge", "G_dense", "B_random",
             "Z_replay", "C_clock_replay", "F_oracle_replay"]
