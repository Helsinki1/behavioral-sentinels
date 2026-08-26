from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from experiments12.adaptive_deployment12 import (
    AdaptiveDeploymentError,
    execute_adaptive_run,
    parser as adaptive_parser,
)
from experiments12.bfcl_run12 import _parse_args as parse_bfcl_args
from experiments12.bfcl_run12 import execute_bfcl_run
from experiments12.deployment12 import (
    DeploymentArtifactError,
    execute_deployment_run,
    parser as deployment_parser,
)
from experiments12.execution_sharding12 import ExecutionShard
from experiments12.runner12 import execute_scripted_run, parser as scripted_parser


class ExecutionShardUnitTests(unittest.TestCase):
    def test_invalid_shard_specs_fail_strictly(self) -> None:
        invalid = (
            (0, 0),
            (-1, 0),
            (1, -1),
            (2, 2),
            (True, 0),
            (1, False),
            (1.0, 0),
            (1, 0.0),
        )
        for count, index in invalid:
            with self.subTest(count=count, index=index):
                with self.assertRaises(ValueError):
                    ExecutionShard(count=count, index=index)  # type: ignore[arg-type]

    def test_disjoint_union_exactly_covers_declared_order(self) -> None:
        declared = tuple(f"cell-{position:02d}" for position in range(23))
        for count in (1, 2, 4, 8, 29):
            with self.subTest(count=count):
                selections = tuple(
                    ExecutionShard(count=count, index=index).select(declared)
                    for index in range(count)
                )
                flattened = tuple(item for selection in selections for item in selection)
                self.assertEqual(len(flattened), len(set(flattened)))
                self.assertEqual(set(flattened), set(declared))
                for index, selection in enumerate(selections):
                    self.assertEqual(selection, declared[index::count])

    def test_clean_shadow_cell_has_the_same_single_owner(self) -> None:
        declared = tuple(
            (f"cell-{position}", "clean" if position % 3 == 0 else "active")
            for position in range(31)
        )
        owners: dict[str, list[int]] = {
            cell_id: [] for cell_id, arm in declared if arm == "clean"
        }
        for index in range(8):
            for cell_id, arm in ExecutionShard(8, index).select(declared):
                if arm == "clean":
                    owners[cell_id].append(index)
        self.assertTrue(owners)
        self.assertTrue(all(len(indices) == 1 for indices in owners.values()))

    def test_all_paid_run_parsers_share_defaults_and_explicit_values(self) -> None:
        parser_cases = (
            (
                scripted_parser(),
                [
                    "run-evolving",
                    "--run-id", "r",
                    "--dataset", "d",
                    "--dataset-sha256", "a" * 64,
                    "--build-receipt", "b",
                    "--tasks", "t",
                ],
            ),
            (None, ["run", "--run-id", "r", "--tasks", "t"]),
            (
                adaptive_parser(),
                [
                    "run-evolving",
                    "--run-id", "r",
                    "--dataset", "d",
                    "--dataset-sha256", "a" * 64,
                    "--build-receipt", "b",
                    "--tasks", "t",
                    "--thresholds", "h",
                ],
            ),
            (
                deployment_parser(),
                [
                    "run-evolving",
                    "--run-id", "r",
                    "--dataset", "d",
                    "--dataset-sha256", "a" * 64,
                    "--build-receipt", "b",
                    "--tasks", "t",
                    "--pass-one", "p",
                    "--thresholds", "h",
                    "--schedule", "s",
                ],
            ),
        )
        for parser, argv in parser_cases:
            with self.subTest(argv=argv[0]):
                defaults = parse_bfcl_args(argv) if parser is None else parser.parse_args(argv)
                self.assertEqual((defaults.shard_count, defaults.shard_index), (1, 0))
                explicit_argv = [*argv, "--shard-count", "8", "--shard-index", "7"]
                explicit = (
                    parse_bfcl_args(explicit_argv)
                    if parser is None
                    else parser.parse_args(explicit_argv)
                )
                self.assertEqual((explicit.shard_count, explicit.shard_index), (8, 7))


class ExecutionShardPreflightTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_executors_reject_invalid_shards_before_artifact_reads(self) -> None:
        with tempfile.TemporaryDirectory(prefix="experiment12-shard-preflight-") as tmp:
            root = Path(tmp)
            missing = root / "missing"
            with self.assertRaisesRegex(ValueError, "shard_count"):
                await execute_scripted_run(
                    run_id="missing",
                    task_manifest_path=missing,
                    tasks=(),
                    artifacts_root=root,
                    shard_count=0,
                )
            with self.assertRaisesRegex(ValueError, "shard_index"):
                await execute_bfcl_run(
                    run_id="missing",
                    task_manifest_path=missing,
                    bridge=object(),  # type: ignore[arg-type]
                    benchmark_receipts=(),
                    yes_spend=True,
                    artifacts_root=root,
                    shard_count=2,
                    shard_index=2,
                )
            with self.assertRaisesRegex(AdaptiveDeploymentError, "shard_count"):
                await execute_adaptive_run(
                    run_id="missing",
                    task_manifest_path=missing,
                    threshold_lock_path=missing,
                    tasks=(),
                    yes_spend=True,
                    artifacts_root=root,
                    shard_count=0,
                )
            with self.assertRaisesRegex(DeploymentArtifactError, "shard_index"):
                await execute_deployment_run(
                    run_id="missing",
                    task_manifest_path=missing,
                    pass_one_path=missing,
                    threshold_lock_path=missing,
                    schedule_path=missing,
                    tasks=(),
                    yes_spend=True,
                    artifacts_root=root,
                    shard_count=3,
                    shard_index=3,
                )


if __name__ == "__main__":
    unittest.main()
