"""Experiment 10 configuration."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results10"
RUNS_DIR = ROOT / "runs7"

# Study A: the two operator regimes already measured, and the policies in each.
# Names are normalised so the two regimes are directly comparable.
REGIMES = {
    "lossless": {
        "label": "Lossless operator (exp 6 re-grounding: restart restores true external state)",
        "metrics": str(ROOT / "results6" / "metrics.json"),
        "policies": {
            "A_no_reset": "never_reset", "C_clock": "clock", "C_ctx": "ctx_growth",
            "C_judge": "llm_judge", "Z_reground": "sentinel_zerocarry",
            "F_oracle": "oracle", "G_dense": "dense_clock", "B_random": "random",
        },
    },
    "lossy": {
        "label": "Lossy operator (exp 5 compaction: restart keeps the agent's own summary)",
        "metrics": str(ROOT / "results5" / "metrics.json"),
        "policies": {
            "A_no_reset": "never_reset", "C_clock": "clock", "C_ctx": "ctx_growth",
            "C_judge": "llm_judge", "Z_routed": "sentinel_zerocarry",
            "F_oracle": "oracle", "B_random": "random", "D_routed": "sentinel_carried",
        },
    },
}

# cost of one restart, in accuracy-equivalent units (0.01 = one accuracy point)
COST_PER_RESET_GRID = [round(0.0005 * i, 5) for i in range(0, 61)]      # 0 .. 0.03
# cost of 1k prompt tokens, in accuracy-equivalent units
COST_PER_KTOK_GRID = [round(0.0002 * i, 5) for i in range(0, 26)]       # 0 .. 0.005

# ---- Study B: the operator-fidelity sweep (new runs)
# phi = fraction of state entries restored from the external store at a reset;
# the remainder are carried over from the agent's OWN belief. phi=1 reproduces
# exp 6's re-grounding, phi=0 reproduces exp 5's self-summary compaction.
FIDELITY_GRID = [1.0, 0.75, 0.5, 0.25, 0.0]

# ---- Study C: synthetic detectors of known quality, built by degrading the oracle
DETECTOR_RECALL = [0.2, 0.4, 0.6, 0.8, 1.0]
DETECTOR_FALSE_ALARM = [0.0, 0.25, 0.5]
