"""Compatibility entry point for strict provider-free two-pass analysis.

The implementation lives in :mod:`experiments12.two_pass_analysis12`.  This
module preserves the shorter production CLI name used in the deployment
runbook while exposing the same public API.
"""

from experiments12.two_pass_analysis12 import *  # noqa: F403
from experiments12.two_pass_analysis12 import (
    TWO_PASS_ANALYSIS_TYPE as DEPLOYMENT_ANALYSIS_TYPE,
    TWO_PASS_ANALYSIS_VERSION as DEPLOYMENT_ANALYSIS_VERSION,
    extract_two_pass_run as extract_deployment_run,
    main,
    parser,
    summarize_two_pass_outcomes as summarize_deployment_outcomes,
    write_two_pass_figures as write_deployment_figures,
)


if __name__ == "__main__":
    raise SystemExit(main())
