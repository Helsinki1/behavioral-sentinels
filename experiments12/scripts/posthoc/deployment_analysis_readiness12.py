#!/usr/bin/env python3
"""Provider-free readiness map for Experiment 12 deployment analyses."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments12.manifest12 import code_tree_hash  # noqa: E402


GENERATED = ROOT / "experiments12" / "data_results" / "derived"
ARTIFACTS = ROOT / "experiments12" / "data_results" / "runs"
README = ROOT / "README.md"
ONLINE_RUN = "e12-deploy-online-evolving-luna-40-v1"
YOKED_RUN = "e12-deploy-twopass-yoked-evolving-luna-40-v1"
EXPECTED_CODE_HASH = (
    "851d54e58d109321ba16e246cc5b3abb9a3601cb5c20a666bbda039dcddf085e"
)
OUTPUT_JSON = GENERATED / "deployment-analysis-readiness12.json"
OUTPUT_MD = GENERATED / "deployment-analysis-readiness12.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def result_names(path: Path) -> set[str]:
    if not path.is_dir():
        return set()
    return {item.name for item in path.iterdir() if item.is_file() and item.suffix == ".json"}


def event_names(path: Path, prefix: str) -> set[str]:
    if not path.is_dir():
        return set()
    return {
        item.name
        for item in path.iterdir()
        if item.is_file() and item.name.startswith(prefix) and item.suffix == ".jsonl"
    }


def run_status(run_id: str, *, mode: str) -> dict[str, Any]:
    run = ARTIFACTS / run_id
    manifest_path = run / "manifest.json"
    pairs_path = run / "pairs.jsonl"
    manifest = load_json(manifest_path)
    cells = load_jsonl(pairs_path)
    cell_ids = {str(row["cell_id"]) for row in cells}
    expected_json = {f"{cell_id}.json" for cell_id in cell_ids}
    if mode == "online_adaptive":
        output_subdir = "adaptive_deployment"
        job_subdir = "adaptive_deployment_jobs"
        event_prefix = "adaptive-"
    else:
        output_subdir = "deployment"
        job_subdir = "deployment_jobs"
        event_prefix = "deployment-"
    outputs = result_names(run / "results" / output_subdir)
    jobs = result_names(run / "results" / job_subdir)
    events = event_names(run / "events", event_prefix)
    expected_events = {f"{event_prefix}{cell_id}.jsonl" for cell_id in cell_ids}
    complete = outputs == expected_json and jobs == expected_json and events == expected_events
    return {
        "run_id": run_id,
        "mode": mode,
        "manifest_sha256": sha256(manifest_path),
        "pair_manifest_sha256": sha256(pairs_path),
        "expected_cells": len(cells),
        "declared_n_cells": manifest["extra_config"]["n_cells"],
        "n_tasks": manifest["extra_config"]["n_tasks"],
        "models": manifest["models"],
        "methods": manifest["arms"],
        "operators": manifest["operators"],
        "progress": {
            "outputs": len(outputs),
            "jobs": len(jobs),
            "event_logs": len(events),
        },
        "exact_cell_coverage_complete": complete,
        "analysis_can_run_now": complete,
        "analysis_output_exists": (
            (run / "results" / ("adaptive-analysis.json" if mode == "online_adaptive" else "two-pass-analysis.json")).is_file()
        ),
    }


def paper_section() -> str:
    text = README.read_text(encoding="utf-8")
    return text.split("\n---\n", 1)[0].strip()


def command_lines(online: Mapping[str, Any], yoked: Mapping[str, Any]) -> dict[str, list[str]]:
    online_root = f"experiments12/data_results/runs/{ONLINE_RUN}/results"
    yoked_root = f"experiments12/data_results/runs/{YOKED_RUN}/results"
    return {
        "online_extract": [
            "python3 -m experiments12.adaptive_analysis12 extract",
            f"--run-id {ONLINE_RUN}",
            f"--manifest-sha256 {online['manifest_sha256']}",
            f"--output {online_root}/adaptive-analysis.json",
            f"--figures {online_root}/adaptive-figures",
            "--artifacts experiments12/data_results/runs",
            "--bootstrap-iterations 2000",
            "--bootstrap-seed 12012",
        ],
        "yoked_validate": [
            "python3 -m experiments12.two_pass_analysis12 validate",
            f"--run-id {YOKED_RUN}",
            f"--manifest-sha256 {yoked['manifest_sha256']}",
            f"--output {yoked_root}/validation-two-pass.json",
            "--artifacts experiments12/data_results/runs",
        ],
        "yoked_extract": [
            "python3 -m experiments12.two_pass_analysis12 extract",
            f"--run-id {YOKED_RUN}",
            f"--manifest-sha256 {yoked['manifest_sha256']}",
            f"--output {yoked_root}/two-pass-analysis.json",
            f"--tables {yoked_root}/two-pass-tables",
            f"--figures {yoked_root}/two-pass-figures",
            "--artifacts experiments12/data_results/runs",
            "--bootstrap-iterations 2000",
            "--bootstrap-seed 12012",
        ],
    }


def claim_map() -> list[dict[str, Any]]:
    return [
        {
            "claim_or_figure": "Overall deployed task success by observation method and class",
            "primary_source": "online",
            "online_fields": [
                "metric_summaries[metric=success].{observation_class,method,operator,n_tasks,mean,ci_low,ci_high}",
                "rows[].{unit_id,method,operator,success}",
            ],
            "yoked_fields": [
                "metric_summaries[metric=success].{observation_class,method,operator,n_tasks,mean,ci_low,ci_high}",
                "method_effects[metric=success].{reference_method,comparison_method,operator,effect,ci_low,ci_high}",
            ],
            "built_in_figure": "online adaptive-figures/deployment-evolving_intent_gsm8k-gpt-5.6-luna.svg",
            "readiness": "direct_once_runs_complete",
            "post_run_gap": "Built-in online plot has four classes (baseline, active, passive-behavioral, passive-observational), while README asks for three; merge the two passive classes only in a clearly labeled paper plot.",
        },
        {
            "claim_or_figure": "Which method helps under none, lossy compaction, public-state regrounding, or GOOD/BAD/WATCH feedback",
            "primary_source": "online",
            "online_fields": [
                "metric_summaries[metric=success] keyed by method/operator",
                "operator_effects[metric=success].{method,operator,control_mean,operator_mean,effect,ci_low,ci_high}",
                "rows[].{unit_id,method,operator,success} for paired custom interactions",
            ],
            "yoked_fields": [
                "operator_effects[metric=success].{method,operator,effect,ci_low,ci_high}",
                "method_effects[metric=success].{reference_method,comparison_method,operator,effect,ci_low,ci_high}",
            ],
            "built_in_figure": "none for effects; use exact rows/effect arrays",
            "readiness": "fields_available_but_provider_free_interaction_plot_needed",
            "post_run_gap": "Neither analyzer emits method-by-operator difference-in-differences. Compute task-paired interactions from rows before claiming one method is specifically better in one regime.",
        },
        {
            "claim_or_figure": "GOOD/BAD/WATCH feedback can improve deployed performance",
            "primary_source": "online_secondary",
            "online_fields": [
                "metric_summaries[metric=success,operator=good_bad_watch_feedback]",
                "operator_effects[metric=success,operator=good_bad_watch_feedback]",
                "rows[operator=good_bad_watch_feedback].{selected_actions,success,total_tokens,actual_cost_usd}",
            ],
            "yoked_fields": [],
            "built_in_figure": "included as operator bars in the online success figure",
            "readiness": "direct_online_only_once_complete",
            "post_run_gap": "The implemented feedback is deterministic, current-prefix, exact-quote-only GOOD/BAD/WATCH—not an LLM-generated breakdown. Narrow the exposition or run a new operator; there is no yoked sensitivity for feedback.",
        },
        {
            "claim_or_figure": "Token, latency, and dollar cost of deployed observation",
            "primary_source": "online",
            "online_fields": [
                "metric_summaries[metric in task_tokens,observer_tokens,total_tokens,latency_ms,actual_cost_usd]",
                "rows[].{task_tokens,observer_tokens,total_tokens,latency_ms,actual_cost_usd}",
            ],
            "yoked_fields": [
                "metric_summaries[metric in total_tokens,latency_ms,actual_cost_usd]",
                "operator_effects and method_effects for the same metrics",
            ],
            "built_in_figure": "two-pass emits total_tokens, latency_ms, and actual_cost_usd figures; online emits success only",
            "readiness": "data_direct_but_online_cost_plot_needed",
            "post_run_gap": "Two-pass resources exclude frozen pass-one passive-observer cost, so use online rows for end-to-end cost; label yoked resource plots as pass-two-only sensitivity.",
        },
        {
            "claim_or_figure": "Natural firing/action frequency and its relationship to success",
            "primary_source": "online",
            "online_fields": [
                "metric_summaries[metric in threshold_firings,selected_actions]",
                "rows[].{observations,threshold_firings,selected_actions,applied_interventions,success}",
            ],
            "yoked_fields": [
                "metric_summaries[metric in action_rate,acted_on_task,scheduled_actions,applied_interventions]",
            ],
            "built_in_figure": "two-pass action_rate figure; no online action-rate figure",
            "readiness": "direct_but_must_be_reported_with_success",
            "post_run_gap": "Calibration used fixed-count ranking, but online execution uses score >= scalar cutoff. Existing diagnostic shows deployed scalar-implied rates from 17.5% to 100%, so never describe deployment as a uniform 20% firing policy.",
        },
        {
            "claim_or_figure": "Carrying an active recomputation probe degrades task performance",
            "primary_source": "confirmatory_observer_effect_plus_yoked_sensitivity",
            "online_fields": [
                "metric_summaries[metric=success] for active_recompute versus zero-carry methods (ecological, natural-policy comparison)",
            ],
            "yoked_fields": [
                "method_effects[metric=success, reference/comparison includes active_recompute] under the same task and checkpoint-1 schedule",
            ],
            "built_in_figure": "two-pass success figure; cross-model causal figure comes from confirmatory observer-effect artifacts, not deployment",
            "readiness": "luna_deployment_sensitivity_once_complete",
            "post_run_gap": "Deployment is one model and one benchmark. Cross-model observer-effect language must come from the completed confirmatory runs, not these deployment analyses.",
        },
        {
            "claim_or_figure": "Signal precision, recall, AUPRC, and early-warning lead time",
            "primary_source": "confirmatory_signal_scoring_not_deployment",
            "online_fields": [],
            "yoked_fields": [],
            "built_in_figure": "confirmatory results/signal-figures/*.svg",
            "readiness": "already_available_upstream",
            "post_run_gap": "Do not infer signal accuracy from deployment success. Use score-confirmatory.json metrics[].{precision,recall,auprc,firing_rate,lead_time} and disclose the scalar/rank diagnostic separately.",
        },
        {
            "claim_or_figure": "Complexity ladder across say-name, increment, and recomputation active chores",
            "primary_source": "unsupported_by_pending_deployments",
            "online_fields": [],
            "yoked_fields": [],
            "built_in_figure": None,
            "readiness": "not_supported",
            "post_run_gap": "Both deployment analyses contain only active_recompute. Retaining a complexity-ladder claim requires new provider runs or clearly labeled legacy evidence.",
        },
        {
            "claim_or_figure": "Deployment generalizes across reasoning and action traces, models, and benchmarks",
            "primary_source": "unsupported_by_pending_deployments",
            "online_fields": [],
            "yoked_fields": [],
            "built_in_figure": None,
            "readiness": "not_supported",
            "post_run_gap": "Both deployments are Luna on Evolving-Intent GSM8K only. BFCL supplies completed action-trace detection/observer-effect evidence, but no deployment evidence.",
        },
        {
            "claim_or_figure": "Ground-truth outcome provenance",
            "primary_source": "both",
            "online_fields": [
                "rows[].success; validated by exact adaptive replay and job/evaluation binding",
            ],
            "yoked_fields": [
                "rows[].{success,outcome_source}",
                "validation.{canonical_regraded_cells,cached_official_cells,primary_ready}",
            ],
            "built_in_figure": None,
            "readiness": "direct_once_runs_complete",
            "post_run_gap": "Explain benchmark-specific final-answer ground truth separately from the earlier event labels used for signal precision/recall.",
        },
    ]


def post_run_actions() -> list[dict[str, str]]:
    return [
        {
            "priority": "P0",
            "kind": "provider_free",
            "action": "Run both exact validation/extraction commands only after exact cell coverage reaches 1120/1120 and 480/480.",
        },
        {
            "priority": "P0",
            "kind": "exposition",
            "action": "Describe online thresholds as scalar-cutoff policies with observed action rates, not fixed 20% rank policies; pair every success plot with firing/action incidence.",
        },
        {
            "priority": "P0",
            "kind": "exposition",
            "action": "Rename GOOD/BAD/WATCH as deterministic quote-only feedback unless a genuinely LLM-generated feedback operator is rerun.",
        },
        {
            "priority": "P0",
            "kind": "exposition",
            "action": "Replace the README's stale Qwen-27B/GPT-4-mini model list with the actual frozen confirmatory models; deployments themselves are Luna-only.",
        },
        {
            "priority": "P1",
            "kind": "provider_free",
            "action": "Add task-paired method contrasts and method-by-operator interactions from rows; create success-effect, action-rate, and online resource figures.",
        },
        {
            "priority": "P1",
            "kind": "provider_free",
            "action": "Create the requested three-cluster paper plot by merging passive subclasses visually while retaining method labels and the four-class machine-readable source.",
        },
        {
            "priority": "P1",
            "kind": "exposition",
            "action": "Present yoked results as a checkpoint-1, one-action sensitivity isolating carry/operator effects; it does not test comparative trigger timing or natural firing quality.",
        },
        {
            "priority": "P1",
            "kind": "exposition",
            "action": "Call the yoked run a sensitivity analysis on the same 40 source tasks—not an independent replication—and fill the README's third passive method slot with trace_rules.",
        },
        {
            "priority": "P2",
            "kind": "new_provider_runs_only_if_claim_retained",
            "action": "Run missing active-complexity variants, multi-model/multi-benchmark deployments, expanded yoked methods/feedback, or LLM-generated feedback only if those broader claims remain central.",
        },
    ]


def markdown(payload: Mapping[str, Any]) -> str:
    online = payload["runs"]["online"]
    yoked = payload["runs"]["yoked"]
    commands = payload["commands"]
    lines = [
        "# Experiment 12 deployment-analysis readiness",
        "",
        f"Provider-free snapshot: `{payload['snapshot_utc']}`. Code tree: `{payload['code_tree_sha256']}`.",
        "",
        "## Run gates",
        "",
        "| analysis | required | outputs | jobs | events | ready |",
        "|---|---:|---:|---:|---:|---|",
        f"| online adaptive | {online['expected_cells']} | {online['progress']['outputs']} | {online['progress']['jobs']} | {online['progress']['event_logs']} | {'yes' if online['analysis_can_run_now'] else 'no'} |",
        f"| yoked two-pass | {yoked['expected_cells']} | {yoked['progress']['outputs']} | {yoked['progress']['jobs']} | {yoked['progress']['event_logs']} | {'yes' if yoked['analysis_can_run_now'] else 'no'} |",
        "",
        "Run the commands below only when each row has exact coverage.",
        "",
        "## Exact commands",
        "",
    ]
    for name in ("online_extract", "yoked_validate", "yoked_extract"):
        shell_command = " ".join(commands[name])
        lines.extend(
            [f"### {name.replace('_', ' ')}", "", "```bash", shell_command, "```", ""]
        )
    lines.extend(["## Claim and figure map", ""])
    for index, row in enumerate(payload["claim_map"], start=1):
        lines.extend(
            [
                f"{index}. **{row['claim_or_figure']}** — `{row['readiness']}`",
                f"   - Online: {('; '.join(row['online_fields']) if row['online_fields'] else 'not available')}",
                f"   - Yoked: {('; '.join(row['yoked_fields']) if row['yoked_fields'] else 'not available')}",
                f"   - Gap: {row['post_run_gap']}",
                "",
            ]
        )
    lines.extend(["## Post-run checklist", ""])
    for row in payload["post_run_actions"]:
        lines.append(f"- [ ] **{row['priority']} · {row['kind']}** — {row['action']}")
    lines.extend(
        [
            "",
            "## Built-in outputs",
            "",
            "- Online: one success SVG plus exact sidecar; JSON contains rows, metric summaries, and paired operator-minus-none effects.",
            "- Yoked: validation JSON; full analysis JSON; four CSV tables; success, action-rate, total-token, latency, and cost SVGs with sidecars.",
            "- Missing but derivable without providers: cross-method online effects, method-by-operator interactions, three-class paper plot, online action/resource plots.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    code_hash = code_tree_hash(ROOT / "experiments12")
    if code_hash != EXPECTED_CODE_HASH:
        raise RuntimeError(f"frozen code tree changed: {code_hash}")
    online = run_status(ONLINE_RUN, mode="online_adaptive")
    yoked = run_status(YOKED_RUN, mode="two_pass_frozen")
    schedule = load_json(
        ARTIFACTS / YOKED_RUN / "results" / "deployment_schedule.json"
    )
    groups = schedule["groups"]
    yoked_schedule_summary = {
        "groups": len(groups),
        "groups_with_one_action": sum(len(group["actions"]) == 1 for group in groups),
        "unique_action_checkpoints": sorted(
            {
                action["checkpoint"]
                for group in groups
                for action in group["actions"]
            }
        ),
        "trigger_methods": sorted(
            {
                action["trigger_method"]
                for group in groups
                for action in group["actions"]
            }
        ),
    }
    payload = {
        "artifact_type": "experiment12_deployment_analysis_readiness",
        "schema_version": 1,
        "snapshot_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "provider_calls_made": 0,
        "code_tree_sha256": code_hash,
        "reviewed_sources": {
            "paper_contents": {
                "path": "README.md#paper-contents",
                "section_sha256": hashlib.sha256(paper_section().encode("utf-8")).hexdigest(),
                "file_sha256": sha256(README),
            },
            "adaptive_analysis": {
                "path": "experiments12/adaptive_analysis12.py",
                "sha256": sha256(ROOT / "experiments12" / "adaptive_analysis12.py"),
            },
            "two_pass_analysis": {
                "path": "experiments12/two_pass_analysis12.py",
                "sha256": sha256(ROOT / "experiments12" / "two_pass_analysis12.py"),
            },
        },
        "runs": {"online": online, "yoked": yoked},
        "yoked_schedule_summary": yoked_schedule_summary,
        "commands": command_lines(online, yoked),
        "claim_map": claim_map(),
        "post_run_actions": post_run_actions(),
    }
    atomic_text(OUTPUT_JSON, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    atomic_text(OUTPUT_MD, markdown(payload))
    if code_tree_hash(ROOT / "experiments12") != EXPECTED_CODE_HASH:
        raise RuntimeError("code tree changed while writing readiness artifacts")
    print(json.dumps({"json": str(OUTPUT_JSON.relative_to(ROOT)), "markdown": str(OUTPUT_MD.relative_to(ROOT)), "online_ready": online["analysis_can_run_now"], "yoked_ready": yoked["analysis_can_run_now"]}, sort_keys=True))


if __name__ == "__main__":
    main()
