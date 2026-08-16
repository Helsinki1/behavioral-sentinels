"""Experiment 2 reuses experiment 1's HTTP transport verbatim so that any
throughput/retry difference between the two experiments is ruled out."""
from experiments.llm import chat, get_client  # noqa: F401
