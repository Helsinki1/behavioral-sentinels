from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from experiments12.core.artifacts import atomic_write_jsonl, sha256_file
from experiments12.manifest12 import (
    ArtifactReceipt,
    RunLayout,
    build_manifest,
    validate_manifest_files,
    write_manifest_once,
)
from experiments12.passive_spec12 import PASSIVE_MONITOR_SPEC_SHA256
from experiments12.spec12 import Stage


ROOT = Path(__file__).resolve().parent.parent


class ManifestTests(unittest.TestCase):
    def test_layout_rejects_path_escape(self):
        with self.assertRaises(ValueError):
            RunLayout.for_run("artifacts", "../escape")

    def test_external_receipt_redacts_parent_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "dataset.json"
            file.write_text("{}")
            receipt = ArtifactReceipt.from_file("data", file, workspace=ROOT)
            self.assertEqual(receipt.path, "external:dataset.json")
            self.assertEqual(receipt.sha256, sha256_file(file))

    def test_manifest_is_write_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            pair_path = Path(tmp) / "pairs.jsonl"
            atomic_write_jsonl(pair_path, [{"cell": 1}])
            manifest = build_manifest(
                run_id="smoke-1",
                stage=Stage.SMOKE,
                repository_root=ROOT,
                pair_manifest_sha256=sha256_file(pair_path),
                models=("gpt-5.6-luna",),
                arms=("clean",),
                operators=("none",),
                randomization_seed=12,
                benchmark_receipts=(
                    ArtifactReceipt(
                        name="synthetic",
                        path="external:synthetic.json",
                        sha256="c" * 64,
                        upstream_commit="deadbeef",
                        license_id="MIT",
                    ),
                ),
            )
            path = Path(tmp) / "manifest.json"
            write_manifest_once(path, manifest)
            write_manifest_once(path, manifest)
            changed = dict(manifest)
            changed["randomization_seed"] = 13
            with self.assertRaises(FileExistsError):
                write_manifest_once(path, changed)
            payload = json.loads(path.read_text())
            self.assertFalse(payload["secret_values_recorded"])
            self.assertEqual(payload["benchmark_receipts"][0]["license_id"], "MIT")
            self.assertEqual(
                payload["passive_monitor_spec"]["sha256"],
                PASSIVE_MONITOR_SPEC_SHA256,
            )

            for label, mutate in (
                ("missing", lambda value: value.pop("passive_monitor_spec")),
                (
                    "extra",
                    lambda value: value["passive_monitor_spec"]["spec"].__setitem__(
                        "undeclared", True
                    ),
                ),
                (
                    "changed",
                    lambda value: value["passive_monitor_spec"].__setitem__(
                        "sha256", "f" * 64
                    ),
                ),
            ):
                with self.subTest(label=label):
                    adversarial = json.loads(json.dumps(payload))
                    mutate(adversarial)
                    errors = validate_manifest_files(
                        adversarial,
                        repository_root=ROOT,
                        pair_manifest_path=pair_path,
                    )
                    self.assertTrue(
                        any("passive monitor" in error for error in errors), errors
                    )


if __name__ == "__main__":
    unittest.main()
