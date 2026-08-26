"""Authoritative filesystem layout for Experiment 12."""

from pathlib import Path


SCRIPTS_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SCRIPTS_ROOT.parent
REPOSITORY_ROOT = EXPERIMENT_ROOT.parent

DATA_RESULTS_ROOT = EXPERIMENT_ROOT / "data_results"
INPUTS_ROOT = DATA_RESULTS_ROOT / "inputs"
RUNS_ROOT = DATA_RESULTS_ROOT / "runs"
DERIVED_ROOT = DATA_RESULTS_ROOT / "derived"
GRAPHS_ROOT = EXPERIMENT_ROOT / "graphs"
TESTS_ROOT = SCRIPTS_ROOT / "tests"
POSTHOC_ROOT = SCRIPTS_ROOT / "posthoc"
INTEGRATIONS_ROOT = SCRIPTS_ROOT / "integrations"
