"""Unit tests for the Experiment 12 model catalog; no live requests are made."""

from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unittest

from experiments12.models12 import (
    CATALOG,
    CATALOG_PATH,
    CatalogValidationError,
    JUDGE_MODEL_NAME,
    TARGET_MODEL_NAMES,
    estimate_call_upper_bound_usd,
    load_model_catalog,
    preflight_model_availability,
)


class _Response:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self._body = json.dumps(payload).encode("utf-8")
        self.status = status

    def read(self, _limit: int = -1) -> bytes:
        return self._body

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class CatalogTests(unittest.TestCase):
    def test_frozen_slate_roles_endpoints_and_sources(self) -> None:
        self.assertEqual(len(TARGET_MODEL_NAMES), 5)
        self.assertEqual(JUDGE_MODEL_NAME, "gpt-5.6-sol-judge")
        self.assertEqual(len(CATALOG.models), 6)
        self.assertEqual(CATALOG.pricing_snapshot_date.isoformat(), "2026-08-26")
        self.assertEqual(
            CATALOG.models["qwen3p7-plus"].model,
            "accounts/fireworks/models/qwen3p7-plus",
        )
        for model in CATALOG.models.values():
            self.assertTrue(model.inference_url.startswith("https://api."))
            self.assertTrue(model.models_url.endswith("/models"))
            self.assertIn(model.api_key_env, {"FIREWORKS_API_KEY", "OPENAI_API_KEY"})
            self.assertTrue(model.native_function_calling)
            self.assertGreaterEqual(len(model.sources), 2)
            self.assertTrue(all(source.startswith("https://") for source in model.sources))

    def test_loader_rejects_changed_frozen_identity(self) -> None:
        raw = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        raw["models"][0]["api_key_env"] = "SOME_OTHER_KEY"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(CatalogValidationError):
                load_model_catalog(path)


class CostTests(unittest.TestCase):
    def test_no_cache_discount_and_full_output_are_assumed(self) -> None:
        self.assertEqual(
            estimate_call_upper_bound_usd(
                "gpt-oss-120b", 1_000_000, 1_000_000, input_headroom=0
            ),
            Decimal("0.750000"),
        )
        # The conservative OpenAI bound uses the $0.25 cache-write rate,
        # which is higher than Luna's $0.20 ordinary input rate.
        self.assertEqual(
            estimate_call_upper_bound_usd(
                "gpt-5.6-luna", 100_000, 100_000, input_headroom=0
            ),
            Decimal("0.145000"),
        )

    def test_headroom_long_context_and_batch(self) -> None:
        # 250k + default 10% headroom crosses the 272k long-context threshold.
        self.assertEqual(
            estimate_call_upper_bound_usd("gpt-5.6-luna", 250_000, 0),
            Decimal("0.137500"),
        )
        self.assertEqual(
            estimate_call_upper_bound_usd(
                "qwen3p7-plus", 1_000_000, 1_000_000,
                input_headroom=0, batch=True
            ),
            Decimal("1.000000"),
        )

    def test_invalid_estimate_inputs_fail(self) -> None:
        with self.assertRaises(ValueError):
            estimate_call_upper_bound_usd("gpt-oss-120b", -1, 1)
        with self.assertRaises(ValueError):
            estimate_call_upper_bound_usd("gpt-oss-120b", 1, 1, input_headroom="nan")
        with self.assertRaises(KeyError):
            estimate_call_upper_bound_usd("not-a-model", 1, 1)
        with self.assertRaises(ValueError):
            estimate_call_upper_bound_usd("gpt-oss-120b", 1, 1, batch="yes")


class PreflightTests(unittest.TestCase):
    def test_missing_credentials_performs_no_network_io(self) -> None:
        def should_not_run(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("network must not be called without credentials")

        results = preflight_model_availability(environ={}, urlopen=should_not_run)
        self.assertEqual(len(results), 6)
        self.assertTrue(all(result.status == "missing_credentials" for result in results))
        self.assertTrue(all(result.available is None for result in results))

    def test_one_free_models_request_per_provider_and_no_secret_in_results(self) -> None:
        secrets = {
            "FIREWORKS_API_KEY": "fw-super-secret",
            "OPENAI_API_KEY": "oa-super-secret",
        }
        calls = []

        def fake_urlopen(request, *, timeout):
            calls.append((request.full_url, request.get_header("Authorization"), timeout))
            if "fireworks.ai" in request.full_url:
                ids = [
                    model.model
                    for model in CATALOG.models.values()
                    if model.provider == "fireworks"
                ]
            else:
                ids = [
                    model.model
                    for model in CATALOG.models.values()
                    if model.provider == "openai"
                ]
            return _Response({"object": "list", "data": [{"id": value} for value in ids]})

        results = preflight_model_availability(environ=secrets, urlopen=fake_urlopen)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(result.available is True for result in results))
        rendered = repr(results)
        self.assertNotIn(secrets["FIREWORKS_API_KEY"], rendered)
        self.assertNotIn(secrets["OPENAI_API_KEY"], rendered)

    def test_transport_exception_text_is_not_returned(self) -> None:
        secret = "must-never-escape"

        def failing_urlopen(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError(secret)

        results = preflight_model_availability(
            environ={"FIREWORKS_API_KEY": secret, "OPENAI_API_KEY": secret},
            urlopen=failing_urlopen,
        )
        self.assertTrue(all(result.status == "unexpected_error" for result in results))
        self.assertNotIn(secret, repr(results))

if __name__ == "__main__":
    unittest.main()
