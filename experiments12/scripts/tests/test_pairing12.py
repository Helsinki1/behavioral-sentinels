from __future__ import annotations

import unittest

from experiments12.pairing12 import (
    JobCell,
    TaskRef,
    check_completeness,
    make_pair_manifest,
    manifest_sha256,
)


TASKS = (
    TaskRef("reasoning", "a", "a" * 64),
    TaskRef("reasoning", "b", "b" * 64),
)


class PairingTests(unittest.TestCase):
    def test_manifest_is_deterministic_complete_and_block_randomized(self):
        kwargs = dict(
            tasks=TASKS,
            models=("m1", "m2"),
            arms=("clean", "active"),
            operators=("none", "reground"),
            replicates=2,
            randomization_seed=12,
        )
        first = make_pair_manifest(**kwargs)
        second = make_pair_manifest(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(manifest_sha256(first), manifest_sha256(second))
        self.assertEqual(len(first), 2 * 2 * 2 * 2 * 2)
        self.assertEqual(len({cell.cell_id for cell in first}), len(first))
        blocks = {}
        for cell in first:
            blocks.setdefault(cell.block_id, []).append((cell.arm, cell.operator))
        self.assertTrue(all(len(values) == 4 for values in blocks.values()))

    def test_changed_seed_changes_order_not_cells(self):
        common = dict(tasks=TASKS, models=("m",), arms=("a", "b", "c"))
        first = make_pair_manifest(**common, randomization_seed=1)
        second = make_pair_manifest(**common, randomization_seed=2)
        self.assertNotEqual([c.arm for c in first], [c.arm for c in second])

    def test_completeness_fails_closed(self):
        cells = make_pair_manifest(
            tasks=TASKS,
            models=("m",),
            arms=("clean", "active"),
            randomization_seed=1,
        )
        partial = check_completeness(
            cells,
            [(cells[0].cell_id, "complete"), (cells[1].cell_id, "failed")],
        )
        self.assertFalse(partial.primary_ready)
        self.assertEqual(partial.complete, 1)
        self.assertEqual(partial.failed, 1)
        self.assertEqual(partial.missing, len(cells) - 2)
        complete = check_completeness(cells, [(cell.cell_id, "complete") for cell in cells])
        self.assertTrue(complete.primary_ready)

    def test_duplicate_result_is_not_accepted(self):
        cells = make_pair_manifest(
            tasks=TASKS[:1], models=("m",), arms=("clean",), randomization_seed=1
        )
        report = check_completeness(
            cells,
            [(cells[0].cell_id, "complete"), (cells[0].cell_id, "complete")],
        )
        self.assertFalse(report.primary_ready)
        self.assertEqual(report.duplicate_results, (cells[0].cell_id,))

    def test_job_cell_strict_round_trip(self):
        cell = make_pair_manifest(
            tasks=TASKS[:1], models=("m",), arms=("clean",), randomization_seed=1
        )[0]
        self.assertEqual(JobCell.from_dict(cell.as_dict()), cell)
        corrupted = cell.as_dict()
        corrupted["unexpected"] = True
        with self.assertRaises(ValueError):
            JobCell.from_dict(corrupted)


if __name__ == "__main__":
    unittest.main()
