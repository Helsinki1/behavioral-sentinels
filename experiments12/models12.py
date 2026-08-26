"""Validated model catalog, cost guardrails, and a free availability check.

Importing this module reads only the bundled JSON catalog.  It never reads a
``.env`` file, inspects credentials, or performs network I/O.  The explicit
``preflight_model_availability`` function reads credentials only from the
environment mapping passed to it (or ``os.environ``) and sends one unactioned
``GET /models`` request per provider.  It never logs or returns an API key.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import json
import math
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping
import urllib.error
import urllib.request
from urllib.parse import urlparse


CATALOG_PATH = Path(__file__).with_name("model_prices.json")
TOKEN_UNIT = 1_000_000

_EXPECTED_MODELS = {
    "gpt-oss-120b": (
        "target",
        "fireworks",
        "accounts/fireworks/models/gpt-oss-120b",
    ),
    "deepseek-v4-flash-0731": (
        "target",
        "fireworks",
        "accounts/fireworks/models/deepseek-v4-flash-0731",
    ),
    "qwen3p7-plus": (
        "target",
        "fireworks",
        "accounts/fireworks/models/qwen3p7-plus",
    ),
    "gpt-5.6-luna": ("target", "openai", "gpt-5.6-luna"),
    "gpt-5.6-terra": ("target", "openai", "gpt-5.6-terra"),
    "gpt-5.6-sol-judge": ("judge", "openai", "gpt-5.6-sol"),
}

_PROVIDER_CONTRACT = {
    "fireworks": {
        "api_key_env": "FIREWORKS_API_KEY",
        "inference_url": "https://api.fireworks.ai/inference/v1/chat/completions",
        "models_url": "https://api.fireworks.ai/inference/v1/models",
        "source_hosts": frozenset(
            {"fireworks.ai", "app.fireworks.ai", "docs.fireworks.ai"}
        ),
    },
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "inference_url": "https://api.openai.com/v1/responses",
        "models_url": "https://api.openai.com/v1/models",
        "source_hosts": frozenset({"developers.openai.com"}),
    },
}


class CatalogValidationError(ValueError):
    """The bundled or supplied model catalog violates its schema."""


@dataclass(frozen=True)
class LongContextPricing:
    threshold_input_tokens: int
    input_per_million_usd: Decimal
    cached_input_per_million_usd: Decimal
    cache_write_per_million_usd: Decimal
    output_per_million_usd: Decimal


@dataclass(frozen=True)
class Pricing:
    tier: str
    input_per_million_usd: Decimal
    cached_input_per_million_usd: Decimal
    cache_write_per_million_usd: Decimal | None
    output_per_million_usd: Decimal
    batch_multiplier: Decimal
    long_context: LongContextPricing | None


@dataclass(frozen=True)
class ModelSpec:
    name: str
    role: str
    provider: str
    model: str
    inference_url: str
    models_url: str
    api_key_env: str
    context_window_tokens: int
    native_function_calling: bool
    pricing: Pricing
    sources: tuple[str, ...]


@dataclass(frozen=True)
class ModelCatalog:
    schema_version: int
    pricing_snapshot_date: date
    currency: str
    token_unit: int
    models: Mapping[str, ModelSpec]

    @property
    def targets(self) -> tuple[ModelSpec, ...]:
        return tuple(model for model in self.models.values() if model.role == "target")

    @property
    def judges(self) -> tuple[ModelSpec, ...]:
        return tuple(model for model in self.models.values() if model.role == "judge")


@dataclass(frozen=True)
class AvailabilityResult:
    """A deliberately credential-free result for one catalog entry."""

    name: str
    provider: str
    model: str
    role: str
    available: bool | None
    status: str
    http_status: int | None = None


def _fail(message: str) -> None:
    raise CatalogValidationError(message)


def _required(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        _fail(f"{where}: missing required field {key!r}")
    return mapping[key]


def _as_nonnegative_decimal(value: Any, where: str, *, positive: bool = False) -> Decimal:
    if isinstance(value, bool):
        _fail(f"{where}: expected a number")
    try:
        number = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError):
        _fail(f"{where}: expected a finite decimal number")
    if not number.is_finite() or number < 0 or (positive and number == 0):
        qualifier = "positive" if positive else "non-negative"
        _fail(f"{where}: expected a finite {qualifier} number")
    return number


def _validate_url(value: Any, where: str, allowed_hosts: frozenset[str]) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{where}: expected a non-empty URL")
    parsed = urlparse(value)
    if parsed.scheme != "https" or parsed.hostname not in allowed_hosts:
        _fail(f"{where}: expected an official HTTPS source URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        _fail(f"{where}: credentials, query strings, and fragments are not allowed")
    return value


def _parse_long_context(raw: Any, where: str) -> LongContextPricing | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        _fail(f"{where}: expected an object")
    threshold = _required(raw, "threshold_input_tokens", where)
    if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold <= 0:
        _fail(f"{where}.threshold_input_tokens: expected a positive integer")
    result = LongContextPricing(
        threshold_input_tokens=threshold,
        input_per_million_usd=_as_nonnegative_decimal(
            _required(raw, "input_per_million_usd", where),
            f"{where}.input_per_million_usd",
            positive=True,
        ),
        cached_input_per_million_usd=_as_nonnegative_decimal(
            _required(raw, "cached_input_per_million_usd", where),
            f"{where}.cached_input_per_million_usd",
        ),
        cache_write_per_million_usd=_as_nonnegative_decimal(
            _required(raw, "cache_write_per_million_usd", where),
            f"{where}.cache_write_per_million_usd",
            positive=True,
        ),
        output_per_million_usd=_as_nonnegative_decimal(
            _required(raw, "output_per_million_usd", where),
            f"{where}.output_per_million_usd",
            positive=True,
        ),
    )
    if result.cached_input_per_million_usd > result.input_per_million_usd:
        _fail(f"{where}: cached input cannot cost more than uncached input")
    if result.cache_write_per_million_usd < result.input_per_million_usd:
        _fail(f"{where}: cache-write rate cannot be below uncached-input rate")
    return result


def _parse_pricing(raw: Any, provider: str, where: str) -> Pricing:
    if not isinstance(raw, Mapping):
        _fail(f"{where}: expected an object")
    tier = _required(raw, "tier", where)
    if not isinstance(tier, str) or not tier:
        _fail(f"{where}.tier: expected a non-empty string")
    input_rate = _as_nonnegative_decimal(
        _required(raw, "input_per_million_usd", where),
        f"{where}.input_per_million_usd",
        positive=True,
    )
    cached_rate = _as_nonnegative_decimal(
        _required(raw, "cached_input_per_million_usd", where),
        f"{where}.cached_input_per_million_usd",
    )
    output_rate = _as_nonnegative_decimal(
        _required(raw, "output_per_million_usd", where),
        f"{where}.output_per_million_usd",
        positive=True,
    )
    batch_multiplier = _as_nonnegative_decimal(
        _required(raw, "batch_multiplier", where),
        f"{where}.batch_multiplier",
        positive=True,
    )
    if batch_multiplier > 1:
        _fail(f"{where}.batch_multiplier: cannot exceed 1")
    if cached_rate > input_rate:
        _fail(f"{where}: cached input cannot cost more than uncached input")

    raw_cache_write = raw.get("cache_write_per_million_usd")
    cache_write = None
    if raw_cache_write is not None:
        cache_write = _as_nonnegative_decimal(
            raw_cache_write,
            f"{where}.cache_write_per_million_usd",
            positive=True,
        )
        if cache_write < input_rate:
            _fail(f"{where}: cache-write rate cannot be below uncached-input rate")

    long_context = _parse_long_context(raw.get("long_context"), f"{where}.long_context")
    if provider == "openai" and (cache_write is None or long_context is None):
        _fail(f"{where}: OpenAI entries require cache-write and long-context rates")
    if provider == "fireworks" and (cache_write is not None or long_context is not None):
        _fail(f"{where}: Fireworks entries must not invent cache-write/long-context rates")

    return Pricing(
        tier=tier,
        input_per_million_usd=input_rate,
        cached_input_per_million_usd=cached_rate,
        cache_write_per_million_usd=cache_write,
        output_per_million_usd=output_rate,
        batch_multiplier=batch_multiplier,
        long_context=long_context,
    )


def load_model_catalog(path: str | Path = CATALOG_PATH) -> ModelCatalog:
    """Load and strictly validate an Experiment 12 price/model snapshot."""

    catalog_path = Path(path)
    try:
        raw = json.loads(catalog_path.read_text(encoding="utf-8"), parse_float=Decimal)
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise CatalogValidationError(f"could not read valid catalog JSON: {catalog_path}") from exc

    if not isinstance(raw, Mapping):
        _fail("catalog root: expected an object")
    schema_version = _required(raw, "schema_version", "catalog")
    if schema_version != 1:
        _fail("catalog.schema_version: expected 1")
    currency = _required(raw, "currency", "catalog")
    if currency != "USD":
        _fail("catalog.currency: expected 'USD'")
    token_unit = _required(raw, "token_unit", "catalog")
    if token_unit != TOKEN_UNIT:
        _fail(f"catalog.token_unit: expected {TOKEN_UNIT}")
    snapshot_text = _required(raw, "pricing_snapshot_date", "catalog")
    if not isinstance(snapshot_text, str):
        _fail("catalog.pricing_snapshot_date: expected YYYY-MM-DD")
    try:
        snapshot = date.fromisoformat(snapshot_text)
    except ValueError as exc:
        raise CatalogValidationError(
            "catalog.pricing_snapshot_date: expected a valid YYYY-MM-DD date"
        ) from exc

    raw_models = _required(raw, "models", "catalog")
    if not isinstance(raw_models, list):
        _fail("catalog.models: expected a list")
    parsed_models: dict[str, ModelSpec] = {}
    for index, raw_model in enumerate(raw_models):
        where = f"catalog.models[{index}]"
        if not isinstance(raw_model, Mapping):
            _fail(f"{where}: expected an object")
        name = _required(raw_model, "name", where)
        if not isinstance(name, str) or not name:
            _fail(f"{where}.name: expected a non-empty string")
        if name in parsed_models:
            _fail(f"{where}.name: duplicate name {name!r}")
        if name not in _EXPECTED_MODELS:
            _fail(f"{where}.name: unexpected Experiment 12 model {name!r}")

        expected_role, expected_provider, expected_model = _EXPECTED_MODELS[name]
        role = _required(raw_model, "role", where)
        provider = _required(raw_model, "provider", where)
        model = _required(raw_model, "model", where)
        if (role, provider, model) != (expected_role, expected_provider, expected_model):
            _fail(f"{where}: role/provider/model does not match the frozen Experiment 12 slate")
        contract = _PROVIDER_CONTRACT[provider]
        api_key_env = _required(raw_model, "api_key_env", where)
        context_window_tokens = _required(raw_model, "context_window_tokens", where)
        if (
            isinstance(context_window_tokens, bool)
            or not isinstance(context_window_tokens, int)
            or context_window_tokens < 1024
        ):
            _fail(f"{where}.context_window_tokens: expected an integer >= 1024")
        inference_url = _required(raw_model, "inference_url", where)
        models_url = _required(raw_model, "models_url", where)
        if api_key_env != contract["api_key_env"]:
            _fail(f"{where}.api_key_env: unexpected credential variable")
        if inference_url != contract["inference_url"]:
            _fail(f"{where}.inference_url: unexpected provider endpoint")
        if models_url != contract["models_url"]:
            _fail(f"{where}.models_url: unexpected provider endpoint")
        native_functions = _required(raw_model, "native_function_calling", where)
        if native_functions is not True:
            _fail(f"{where}.native_function_calling: all primary entries must support it")

        raw_sources = _required(raw_model, "sources", where)
        if not isinstance(raw_sources, list) or len(raw_sources) < 2:
            _fail(f"{where}.sources: expected at least model-card and pricing URLs")
        sources = tuple(
            _validate_url(source, f"{where}.sources[{source_index}]", contract["source_hosts"])
            for source_index, source in enumerate(raw_sources)
        )
        pricing = _parse_pricing(raw_model.get("pricing"), provider, f"{where}.pricing")
        parsed_models[name] = ModelSpec(
            name=name,
            role=role,
            provider=provider,
            model=model,
            inference_url=inference_url,
            models_url=models_url,
            api_key_env=api_key_env,
            context_window_tokens=context_window_tokens,
            native_function_calling=native_functions,
            pricing=pricing,
            sources=sources,
        )

    missing = set(_EXPECTED_MODELS).difference(parsed_models)
    if missing:
        _fail(f"catalog.models: missing frozen Experiment 12 models: {sorted(missing)!r}")
    if len(parsed_models) != len(_EXPECTED_MODELS):
        _fail("catalog.models: wrong number of entries")

    return ModelCatalog(
        schema_version=schema_version,
        pricing_snapshot_date=snapshot,
        currency=currency,
        token_unit=token_unit,
        models=MappingProxyType(parsed_models),
    )


CATALOG = load_model_catalog()
TARGET_MODEL_NAMES = tuple(model.name for model in CATALOG.targets)
JUDGE_MODEL_NAME = CATALOG.judges[0].name


def _whole_tokens(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def estimate_call_upper_bound_usd(
    model_name: str,
    input_tokens: int,
    output_token_ceiling: int,
    *,
    catalog: ModelCatalog = CATALOG,
    input_headroom: Decimal | float | str = Decimal("0.10"),
    batch: bool = False,
) -> Decimal:
    """Return a conservative, rounded-up dollar ceiling for one model call.

    The caller supplies cumulative prompt tokens and a *total* generated-token
    ceiling (visible plus reasoning tokens).  By default the function adds 10%
    prompt-token headroom, assumes the full output ceiling is used, ignores
    cached-input discounts, and prices cache-capable input at the higher of the
    normal input and cache-write rates.  OpenAI long-context rates are selected
    automatically after headroom.  ``batch=True`` applies the cataloged Batch
    multiplier; leave it false for any live agent turn.
    """

    try:
        spec = catalog.models[model_name]
    except KeyError as exc:
        raise KeyError(f"unknown model name: {model_name}") from exc
    input_tokens = _whole_tokens(input_tokens, "input_tokens")
    output_token_ceiling = _whole_tokens(output_token_ceiling, "output_token_ceiling")
    try:
        headroom = (
            input_headroom
            if isinstance(input_headroom, Decimal)
            else Decimal(str(input_headroom))
        )
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("input_headroom must be a finite non-negative number") from exc
    if not headroom.is_finite() or headroom < 0:
        raise ValueError("input_headroom must be a finite non-negative number")
    if not isinstance(batch, bool):
        raise ValueError("batch must be a boolean")

    buffered_input = math.ceil(Decimal(input_tokens) * (Decimal(1) + headroom))
    pricing = spec.pricing
    long_pricing = pricing.long_context
    if long_pricing is not None and buffered_input > long_pricing.threshold_input_tokens:
        input_rate = max(
            long_pricing.input_per_million_usd,
            long_pricing.cache_write_per_million_usd,
        )
        output_rate = long_pricing.output_per_million_usd
    else:
        input_rate = pricing.input_per_million_usd
        if pricing.cache_write_per_million_usd is not None:
            input_rate = max(input_rate, pricing.cache_write_per_million_usd)
        output_rate = pricing.output_per_million_usd

    cost = (
        Decimal(buffered_input) * input_rate
        + Decimal(output_token_ceiling) * output_rate
    ) / Decimal(catalog.token_unit)
    if batch:
        cost *= pricing.batch_multiplier
    return cost.quantize(Decimal("0.000001"), rounding=ROUND_CEILING)


def _response_model_ids(payload: Any) -> set[str]:
    if not isinstance(payload, Mapping) or not isinstance(payload.get("data"), list):
        raise ValueError("invalid model-list response")
    model_ids: set[str] = set()
    for item in payload["data"]:
        if not isinstance(item, Mapping):
            continue
        candidate = item.get("id", item.get("name"))
        if isinstance(candidate, str) and candidate:
            model_ids.add(candidate)
    return model_ids


def _listed(model: str, returned_ids: set[str]) -> bool:
    """Accept exact provider IDs and a provider's occasional terminal slug."""

    return model in returned_ids or model.rsplit("/", 1)[-1] in returned_ids


def preflight_model_availability(
    *,
    catalog: ModelCatalog = CATALOG,
    environ: Mapping[str, str] | None = None,
    timeout_seconds: float = 10.0,
    urlopen: Callable[..., Any] | None = None,
) -> tuple[AvailabilityResult, ...]:
    """Check the providers' free ``GET /models`` listings without inference.

    Credentials can only come from ``environ`` (default: ``os.environ``); no
    key argument and no ``.env`` loading exist.  One request is sent per
    provider.  Returned records and all error statuses are intentionally
    secret-free, including when a network implementation raises an exception
    whose message contains sensitive text.
    """

    if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool):
        raise ValueError("timeout_seconds must be a positive number")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be a positive number")
    environment = os.environ if environ is None else environ
    open_url = urllib.request.urlopen if urlopen is None else urlopen

    groups: dict[tuple[str, str, str], list[ModelSpec]] = {}
    for spec in catalog.models.values():
        key = (spec.provider, spec.models_url, spec.api_key_env)
        groups.setdefault(key, []).append(spec)

    results_by_name: dict[str, AvailabilityResult] = {}
    for (provider, models_url, api_key_env), specs in groups.items():
        api_key = environment.get(api_key_env, "")
        if not isinstance(api_key, str) or not api_key.strip():
            for spec in specs:
                results_by_name[spec.name] = AvailabilityResult(
                    name=spec.name,
                    provider=provider,
                    model=spec.model,
                    role=spec.role,
                    available=None,
                    status="missing_credentials",
                )
            continue

        request = urllib.request.Request(
            models_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
                "User-Agent": "behavioral-sentinels-exp12-preflight/1",
            },
            method="GET",
        )
        try:
            with open_url(request, timeout=timeout_seconds) as response:
                status_code = getattr(response, "status", 200)
                if not isinstance(status_code, int) or status_code >= 400:
                    raise urllib.error.HTTPError(
                        models_url, int(status_code), "model-list failure", {}, None
                    )
                body = response.read(4_000_001)
                if len(body) > 4_000_000:
                    raise ValueError("model-list response too large")
                payload = json.loads(body.decode("utf-8"))
                returned_ids = _response_model_ids(payload)
        except urllib.error.HTTPError as exc:
            safe_status = exc.code if isinstance(exc.code, int) else None
            for spec in specs:
                results_by_name[spec.name] = AvailabilityResult(
                    name=spec.name,
                    provider=provider,
                    model=spec.model,
                    role=spec.role,
                    available=None,
                    status="http_error",
                    http_status=safe_status,
                )
            continue
        except urllib.error.URLError:
            safe_error = "network_error"
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            safe_error = "invalid_response"
        except Exception:
            # Deliberately do not expose exception text: custom transports can
            # include request headers (and therefore credentials) in it.
            safe_error = "unexpected_error"
        else:
            for spec in specs:
                available = _listed(spec.model, returned_ids)
                results_by_name[spec.name] = AvailabilityResult(
                    name=spec.name,
                    provider=provider,
                    model=spec.model,
                    role=spec.role,
                    available=available,
                    status="available" if available else "not_listed",
                    http_status=status_code,
                )
            continue

        for spec in specs:
            results_by_name[spec.name] = AvailabilityResult(
                name=spec.name,
                provider=provider,
                model=spec.model,
                role=spec.role,
                available=None,
                status=safe_error,
            )

    return tuple(results_by_name[name] for name in catalog.models)


__all__ = [
    "AvailabilityResult",
    "CATALOG",
    "CATALOG_PATH",
    "CatalogValidationError",
    "JUDGE_MODEL_NAME",
    "LongContextPricing",
    "ModelCatalog",
    "ModelSpec",
    "Pricing",
    "TARGET_MODEL_NAMES",
    "estimate_call_upper_bound_usd",
    "load_model_catalog",
    "preflight_model_availability",
]
