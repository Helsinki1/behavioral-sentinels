# External benchmark sources

Third-party repositories and virtual environments are intentionally not
committed inside Experiment 12. They are reproducible inputs, not experiment
results.

| source | pinned commit | local setting |
|---|---|---|
| Microsoft Evolving Intent | `993d6be9597ac03854b46362ccd647eb1bfd267a` | `EVOLVING_INTENT_ROOT` |
| Berkeley Gorilla / BFCL | `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8` | `BFCL_ROOT` |
| TurnBench-MS | `b3a9daa914e66f62048b62cff06bcaf4151aadb5` | `TURNBENCH_MS_ROOT` plus permission receipt |

The Evolving Intent bridge template is
`../../scripts/integrations/evolving_intent_bridge12.py`; copy it into the
pinned checkout before using the reproduction builder. The exact bridge Python
dependency lock is `evolving_intent_math_verify.lock` in this directory.
