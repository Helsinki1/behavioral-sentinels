"""Experiment 12: active and passive observation of agent traces.

This package is intentionally independent of experiments 1--11.  Historical
modules may inform the design, but an Experiment 12 run never imports their
mutable configuration or infers its sample from old artifacts.

The implementation lives in :mod:`experiments12.scripts` so the repository is
easy to scan.  Extending ``__path__`` keeps the original, provenance-recorded
module names (for example ``experiments12.cli12``) working after that physical
reorganization.
"""

from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parent
SCRIPTS_ROOT = EXPERIMENT_ROOT / "scripts"
if SCRIPTS_ROOT.is_dir() and str(SCRIPTS_ROOT) not in __path__:
    __path__.append(str(SCRIPTS_ROOT))

SCHEMA_VERSION = "12.0"
