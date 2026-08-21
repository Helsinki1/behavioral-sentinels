"""Experiment 8: ACTIVE vs PASSIVE OBSERVATION of a long-horizon agent.

The taxonomy's dividing line is whether the observation WRITES INTO THE
AGENT'S TRAJECTORY:

  ACTIVE observation        -- the monitor injects work into the agent's own
                               context: the carried probes of exps 1-5. The
                               chore stays in context and the observation
                               itself degrades the trajectory.
  PASSIVE-BEHAVIOURAL (new) -- the harness freezes the conversation at a
                               checkpoint, FORKS it, asks a short quiz about
                               current state, and discards the exchange.
                               Nothing is ever written to agent state; the
                               cost is fork tokens only.
  PASSIVE-OBSERVATIONAL     -- no extra queries at all: monitors read the
                               trace the agent already produces (exp 5's
                               zero-carry monitors; the LLM judge).

Every method is scored three ways on ONE shared pool (the exp-5/6 90 tasks,
reset operator fixed to exp 6's R1 reground): signal precision/recall at
predicting the first hallucination, end-task accuracy when the signal routes
re-grounded restarts, and the total cost of observation (observer-effect
accuracy delta + monitoring tokens).

Only three arms are new (QUIZ, ACT_probe, ACT_carry_clock); every baseline,
bound, and passive-observational arm is read verbatim from runs6/runs5.
"""
from pathlib import Path

# protocol constants shared verbatim with exps 5/6 -- identity licenses the
# cross-arm pairing
from experiments5.config5 import (CTX_GROWTH_TOKENS, DOMAINS, JUDGE_MODEL,
                                  JUDGE_WINDOW, MAX_RESETS, MODELS,
                                  N_PER_DOMAIN, RESET_GRACE, SCHEDULE_EVERY,
                                  SCHEDULE_FIRST, SEED, SUCCESS_THRESHOLD,
                                  TEMPERATURE, INTENDED_GENRE, load_env)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data3"
RUNS_DIR = ROOT / "runs8"           # new arms + the shadow quiz pass
RUNS5_DIR = ROOT / "runs5"          # A_no_reset lives here
RUNS6_DIR = ROOT / "runs6"          # all reused re-grounding arms live here
RESULTS_DIR = ROOT / "results8"

DEFAULT_MODEL = "gpt-oss-20b"

# ------------------------------------------------------------------- quiz
# Checkpoint cadence matches G_dense (every 3 turns), so quiz frequency is
# never the confound against the densest baseline schedule.
QUIZ_EVERY = 3
QUIZ_FIRST = 3
QUIZ_MAX_TOKENS = 300
QUIZ_FAIL_MIN = 2        # checkpoint fails when >= this many of the 3
                         # deterministic questions are wrong (ablated offline)

# ---------------------------------------------------------------- the arms
# probe: carried condition ('baseline' = none); labeled = exp 5's D_labeled
# assignment (deterministic INTENDED_GENRE lookup -- router noise already
# priced at -0.005 ns there).  All resets use exp 6's R1 reground operator.
ARMS = {
    "QUIZ":            {"policy": "quiz",      "probe": "none",
                        "category": "passive-behavioural"},
    "ACT_probe":       {"policy": "probe",     "probe": "labeled",
                        "category": "active"},
    "ACT_carry_clock": {"policy": "scheduled", "probe": "labeled",
                        "category": "active"},
}

# arms read verbatim from runs6 (and runs5 for A_no_reset), never re-run
REUSED_ARMS = {
    "A_no_reset": "bound",                 "F_oracle": "bound",
    "C_clock": "baseline",                 "C_ctx": "baseline",
    "B_random": "baseline",                "G_dense": "baseline",
    "C_judge": "passive-observational",    "Z_reground": "passive-observational",
}

GATE_ARMS = ["QUIZ"]
ARM_ORDER = ["QUIZ", "ACT_carry_clock", "ACT_probe"]

# display order for every table/figure: bounds, baselines, then the three
# observation categories -- the shared x-axis of the writeup
ARM_DISPLAY = ["A_no_reset", "B_random", "C_clock", "C_ctx", "G_dense",
               "ACT_probe", "ACT_carry_clock", "QUIZ",
               "Z_reground", "C_judge", "F_oracle"]
