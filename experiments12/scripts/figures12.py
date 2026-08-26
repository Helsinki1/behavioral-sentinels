"""Dependency-free, publication-oriented SVG figures for Experiment 12.

Each writer emits a self-contained SVG plus a deterministic ``*.data.json``
sidecar containing the exact plotted values and axis limits.  No result files
are read here; callers pass already validated metrics or explicit plot rows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import html
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Sequence

from .metrics12 import PairedEffect, PairedMetricEffect, PredictionMetrics


SCHEMA_VERSION = 1
_INK = "#222222"
_MUTED = "#666666"
_GRID = "#D9D9D9"
_PALETTE = (
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#000000",
)
_DASHES = ("", "8 4", "3 3", "10 3 2 3", "2 2")


class FigureInputError(ValueError):
    """Figure rows are empty, duplicated, non-finite, or internally invalid."""


@dataclass(frozen=True, slots=True)
class FigureArtifact:
    svg_path: Path
    data_path: Path
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class DeploymentBar:
    observation_class: str
    operator: str
    method: str
    value: float
    n_tasks: int
    ci_low: float | None = None
    ci_high: float | None = None

    def __post_init__(self) -> None:
        for name in ("observation_class", "operator", "method"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise FigureInputError(f"{name} must be a non-empty string")
        _finite("value", self.value)
        if isinstance(self.n_tasks, bool) or not isinstance(self.n_tasks, int) or self.n_tasks < 1:
            raise FigureInputError("n_tasks must be a positive integer")
        if (self.ci_low is None) != (self.ci_high is None):
            raise FigureInputError("ci_low and ci_high must be supplied together")
        if self.ci_low is not None and self.ci_high is not None:
            _finite("ci_low", self.ci_low)
            _finite("ci_high", self.ci_high)
            if not self.ci_low <= self.value <= self.ci_high:
                raise FigureInputError("bar confidence interval must contain value")


@dataclass(frozen=True, slots=True)
class AdvantageCell:
    method: str
    trace_length: str
    context: str
    difficulty: str
    advantage: float
    n_tasks: int

    def __post_init__(self) -> None:
        for name in ("method", "trace_length", "context", "difficulty"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise FigureInputError(f"{name} must be a non-empty string")
        _finite("advantage", self.advantage)
        if isinstance(self.n_tasks, bool) or not isinstance(self.n_tasks, int) or self.n_tasks < 1:
            raise FigureInputError("n_tasks must be a positive integer")


def _finite(name: str, value: float | int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise FigureInputError(f"{name} must be finite")
    return float(value)


def _x(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _fmt(value: float, *, signed: bool = False) -> str:
    if abs(value) < 0.0005:
        value = 0.0
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def _paths(svg_path: str | os.PathLike[str], sidecar_path: str | os.PathLike[str] | None):
    svg = Path(svg_path)
    if svg.suffix.lower() != ".svg":
        raise FigureInputError("svg_path must end in .svg")
    data = Path(sidecar_path) if sidecar_path is not None else svg.with_suffix(".data.json")
    return svg, data


def _write_artifact(
    svg_path: Path,
    data_path: Path,
    svg: str,
    data: dict[str, Any],
    width: int,
    height: int,
) -> FigureArtifact:
    encoded = json.dumps(
        data,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ) + "\n"
    _atomic_write(data_path, encoded)
    _atomic_write(svg_path, svg)
    return FigureArtifact(svg_path=svg_path, data_path=data_path, width=width, height=height)


def _document(
    width: int,
    height: int,
    title: str,
    description: str,
    body: Sequence[str],
    *,
    definitions: str = "",
    attributes: str = "",
) -> str:
    style = f"""
      text {{ font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif; fill: {_INK}; }}
      .title {{ font-size: 20px; font-weight: 700; }}
      .subtitle {{ font-size: 12px; fill: {_MUTED}; }}
      .panel-title {{ font-size: 14px; font-weight: 700; }}
      .label {{ font-size: 11px; }}
      .small {{ font-size: 10px; fill: {_MUTED}; }}
      .axis {{ stroke: {_INK}; stroke-width: 1; fill: none; }}
      .grid {{ stroke: {_GRID}; stroke-width: 1; fill: none; }}
      .zero-line {{ stroke: {_INK}; stroke-width: 1.6; stroke-dasharray: 5 4; fill: none; }}
    """
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="figure-title figure-desc" '
        f'{attributes}>\n'
        f"<title id=\"figure-title\">{_x(title)}</title>\n"
        f"<desc id=\"figure-desc\">{_x(description)}</desc>\n"
        f"<defs><style>{style}</style>{definitions}</defs>\n"
        + "\n".join(body)
        + "\n</svg>\n"
    )


def _nice_limit(maximum: float) -> float:
    maximum = max(float(maximum), 1e-6)
    exponent = math.floor(math.log10(maximum))
    scale = 10.0**exponent
    fraction = maximum / scale
    nice = 1.0 if fraction <= 1 else 2.0 if fraction <= 2 else 5.0 if fraction <= 5 else 10.0
    return nice * scale


def _effect_dict(effect: PairedEffect | PairedMetricEffect) -> dict[str, Any]:
    return asdict(effect)


def write_observer_effect_forest(
    effects: Iterable[PairedEffect | PairedMetricEffect],
    svg_path: str | os.PathLike[str],
    *,
    sidecar_path: str | os.PathLike[str] | None = None,
    title: str = "Observer effect of carried active probes",
) -> FigureArtifact:
    """Draw benchmark small multiples of paired active-minus-clean effects."""

    rows = tuple(effects)
    if not rows:
        raise FigureInputError("observer-effect figure requires at least one row")
    metric_rows = all(isinstance(row, PairedMetricEffect) for row in rows)
    if metric_rows:
        metric_keys = {(row.metric, row.unit, row.favorable_direction) for row in rows}
        arms = {row.active_arm for row in rows}
        if len(metric_keys) != 1 or len(arms) != 1:
            raise FigureInputError(
                "metric observer forest requires one metric/unit/direction and active arm"
            )
        metric, unit, favorable_direction = next(iter(metric_keys))
        axis_label = (
            f"{metric.replace('_', ' ').title()} difference "
            f"(active carried − clean), {unit}; "
            f"{'positive is better' if favorable_direction == 'higher' else 'positive is extra burden'}"
        )
        figure_type = "observer_metric_effect_forest"
    else:
        if any(not isinstance(row, PairedEffect) for row in rows):
            raise FigureInputError("observer-effect rows may not mix metric schemas")
        metric = "success"
        unit = "proportion"
        favorable_direction = "higher"
        axis_label = (
            "Task accuracy difference (active carried − clean); negative means degradation"
        )
        figure_type = "observer_effect_forest"
    effect_decimals = (
        6
        if metric == "actual_cost_usd"
        else 1
        if metric in {"task_tokens", "observer_tokens", "total_tokens", "latency_ms"}
        else 3
    )

    def format_effect(value: float, *, signed: bool = True) -> str:
        if abs(value) < 0.5 * 10 ** (-effect_decimals):
            value = 0.0
        template = f"{{:{'+' if signed else ''}.{effect_decimals}f}}"
        return template.format(value)

    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row.model, row.benchmark)
        if key in seen:
            raise FigureInputError(f"duplicate model/benchmark effect: {key!r}")
        seen.add(key)
        for field in (row.effect, row.ci_low, row.ci_high):
            _finite("effect/CI", field)
        if row.ci_low > row.ci_high or row.n_tasks < 1:
            raise FigureInputError("invalid effect confidence interval or task count")

    benchmarks = sorted({row.benchmark for row in rows})
    by_benchmark = {
        benchmark: sorted(
            (row for row in rows if row.benchmark == benchmark), key=lambda row: row.model
        )
        for benchmark in benchmarks
    }
    panel_width = 360
    left = 24
    top = 78
    row_height = 34
    max_rows = max(len(group) for group in by_benchmark.values())
    width = left * 2 + panel_width * len(benchmarks)
    height = top + max_rows * row_height + 92
    limit = _nice_limit(
        1.12 * max(abs(value) for row in rows for value in (row.ci_low, row.ci_high, row.effect))
    )
    ticks = (-limit, -limit / 2, 0.0, limit / 2, limit)
    body = [f'<text x="24" y="32" class="title">{_x(title)}</text>']
    body.append(
        '<text x="24" y="52" class="subtitle">Paired task bootstrap; points are active minus clean, bars are confidence intervals.</text>'
    )

    for panel_index, benchmark in enumerate(benchmarks):
        x0 = left + panel_index * panel_width
        chart_left = x0 + 112
        chart_right = x0 + panel_width - 22
        chart_width = chart_right - chart_left
        body.append(
            f'<text x="{x0 + panel_width / 2:.1f}" y="{top - 18}" text-anchor="middle" '
            f'class="panel-title">{_x(benchmark)}</text>'
        )
        y_bottom = top + max_rows * row_height
        for tick in ticks:
            x_tick = chart_left + (tick + limit) / (2 * limit) * chart_width
            klass = "zero-line" if tick == 0 else "grid"
            body.append(
                f'<line x1="{x_tick:.2f}" y1="{top - 4}" x2="{x_tick:.2f}" '
                f'y2="{y_bottom}" class="{klass}"/>'
            )
            body.append(
                f'<text x="{x_tick:.2f}" y="{y_bottom + 18}" text-anchor="middle" '
                f'class="small">{format_effect(tick)}</text>'
            )
        for row_index, row in enumerate(by_benchmark[benchmark]):
            y = top + row_index * row_height + 12
            map_x = lambda value: chart_left + (value + limit) / (2 * limit) * chart_width
            low_x, high_x, point_x = map_x(row.ci_low), map_x(row.ci_high), map_x(row.effect)
            body.append(
                f'<text x="{chart_left - 8}" y="{y + 4}" text-anchor="end" class="label">'
                f'{_x(row.model)} <tspan class="small">n={row.n_tasks}</tspan></text>'
            )
            body.append(
                f'<line x1="{low_x:.2f}" y1="{y}" x2="{high_x:.2f}" y2="{y}" '
                f'stroke="{_PALETTE[0]}" stroke-width="3"/>'
            )
            body.append(
                f'<line x1="{low_x:.2f}" y1="{y - 5}" x2="{low_x:.2f}" y2="{y + 5}" '
                f'stroke="{_PALETTE[0]}"/><line x1="{high_x:.2f}" y1="{y - 5}" '
                f'x2="{high_x:.2f}" y2="{y + 5}" stroke="{_PALETTE[0]}"/>'
            )
            body.append(
                f'<circle cx="{point_x:.2f}" cy="{y}" r="5" fill="{_PALETTE[0]}" '
                f'stroke="white" stroke-width="1.5"><title>{_x(row.model)}: '
                f'{format_effect(row.effect)} [{format_effect(row.ci_low)}, '
                f'{format_effect(row.ci_high)}]</title></circle>'
            )
        body.append(
            f'<line x1="{chart_left}" y1="{y_bottom}" x2="{chart_right}" y2="{y_bottom}" class="axis"/>'
        )
    body.append(
        f'<text x="{width / 2:.1f}" y="{height - 22}" text-anchor="middle" class="label">'
        f"{_x(axis_label)}</text>"
    )
    svg = _document(
        width,
        height,
        title,
        "Small multiples show paired task-level active-minus-clean effects by model and benchmark. A vertical zero line marks no observer effect.",
        body,
        attributes=f'data-axis-min="{-limit}" data-axis-max="{limit}"',
    )
    svg_output, data_output = _paths(svg_path, sidecar_path)
    data = {
        "schema_version": SCHEMA_VERSION,
        "figure_type": figure_type,
        "statistical_unit": "task",
        "metric": metric,
        "unit": unit,
        "favorable_direction": favorable_direction,
        "axis": {"minimum": -limit, "maximum": limit, "zero_included": True},
        "rows": [_effect_dict(row) for row in sorted(rows, key=lambda row: (row.benchmark, row.model))],
    }
    return _write_artifact(svg_output, data_output, svg, data, width, height)


def write_observer_metric_effect_forest(
    effects: Iterable[PairedMetricEffect],
    svg_path: str | os.PathLike[str],
    *,
    sidecar_path: str | os.PathLike[str] | None = None,
    title: str = "Resource observer effect of carried active probes",
) -> FigureArtifact:
    """Draw one resource metric's paired active-minus-clean effects."""

    rows = tuple(effects)
    if any(not isinstance(row, PairedMetricEffect) for row in rows):
        raise FigureInputError("effects must contain PairedMetricEffect records")
    return write_observer_effect_forest(
        rows,
        svg_path,
        sidecar_path=sidecar_path,
        title=title,
    )


def _summary_dict(summary: PredictionMetrics) -> dict[str, Any]:
    return asdict(summary)


def write_pr_curves(
    summaries: Iterable[PredictionMetrics],
    svg_path: str | os.PathLike[str],
    *,
    sidecar_path: str | os.PathLike[str] | None = None,
    title: str = "Task-level precision–recall curves",
) -> FigureArtifact:
    """Draw task-level PR curves and mark each locked operating point."""

    rows = tuple(summaries)
    if not rows:
        raise FigureInputError("PR figure requires at least one summary")
    seen: set[tuple[str, str, str, str]] = set()
    for row in rows:
        if not isinstance(row, PredictionMetrics):
            raise FigureInputError("summaries must contain PredictionMetrics")
        key = (row.model, row.benchmark, row.method, row.split)
        if key in seen:
            raise FigureInputError(f"duplicate PR series: {key!r}")
        seen.add(key)
        if row.statistical_unit != "task" or not row.pr_curve:
            raise FigureInputError("PR summaries must be task-level and non-empty")
        prior = -1.0
        for point in row.pr_curve:
            _finite("PR precision", point.precision)
            _finite("PR recall", point.recall)
            if not (0 <= point.precision <= 1 and 0 <= point.recall <= 1):
                raise FigureInputError("PR coordinates must lie in [0, 1]")
            if point.recall < prior:
                raise FigureInputError("PR recall must be non-decreasing")
            prior = point.recall

    width, height = 940, 590
    plot_left, plot_top = 76, 72
    plot_size = 460
    legend_left = 570
    body = [f'<text x="24" y="32" class="title">{_x(title)}</text>']
    body.append(
        '<text x="24" y="52" class="subtitle">Every point aggregates complete task trajectories; locked operating points are diamonds.</text>'
    )
    for index in range(6):
        value = index / 5
        x_pos = plot_left + value * plot_size
        y_pos = plot_top + (1 - value) * plot_size
        klass = "zero-line" if value == 0 else "grid"
        body.append(
            f'<line x1="{x_pos:.1f}" y1="{plot_top}" x2="{x_pos:.1f}" '
            f'y2="{plot_top + plot_size}" class="{klass}"/>'
        )
        body.append(
            f'<line x1="{plot_left}" y1="{y_pos:.1f}" x2="{plot_left + plot_size}" '
            f'y2="{y_pos:.1f}" class="{klass}"/>'
        )
        body.append(
            f'<text x="{x_pos:.1f}" y="{plot_top + plot_size + 18}" text-anchor="middle" '
            f'class="small">{value:.1f}</text>'
        )
        body.append(
            f'<text x="{plot_left - 10}" y="{y_pos + 4:.1f}" text-anchor="end" '
            f'class="small">{value:.1f}</text>'
        )

    for series_index, row in enumerate(sorted(rows, key=lambda item: (item.method, item.model, item.benchmark))):
        color = _PALETTE[series_index % len(_PALETTE)]
        dash = _DASHES[series_index % len(_DASHES)]
        points = list(row.pr_curve)
        path_parts: list[str] = []
        first = points[0]
        x_first = plot_left + first.recall * plot_size
        y_first = plot_top + (1 - first.precision) * plot_size
        path_parts.append(f"M {x_first:.2f} {y_first:.2f}")
        previous = first
        for point in points[1:]:
            x_new = plot_left + point.recall * plot_size
            y_new = plot_top + (1 - point.precision) * plot_size
            x_previous = plot_left + previous.recall * plot_size
            path_parts.append(f"L {x_previous:.2f} {y_new:.2f} L {x_new:.2f} {y_new:.2f}")
            previous = point
        dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
        label = f"{row.method} · {row.model} · {row.benchmark}"
        body.append(
            f'<path d="{" ".join(path_parts)}" fill="none" stroke="{color}" '
            f'stroke-width="2.5"{dash_attribute}><title>{_x(label)}; AUPRC '
            f'{row.auprc:.3f}</title></path>'
        )
        if row.precision is not None:
            operating_x = plot_left + row.recall * plot_size
            operating_y = plot_top + (1 - row.precision) * plot_size
            points_attr = (
                f"{operating_x:.2f},{operating_y - 6:.2f} "
                f"{operating_x + 6:.2f},{operating_y:.2f} "
                f"{operating_x:.2f},{operating_y + 6:.2f} "
                f"{operating_x - 6:.2f},{operating_y:.2f}"
            )
            body.append(
                f'<polygon points="{points_attr}" fill="{color}" stroke="white" stroke-width="1">'
                f'<title>Locked threshold {row.locked_threshold:.4g}: precision '
                f'{row.precision:.3f}, recall {row.recall:.3f}</title></polygon>'
            )
        legend_y = 92 + series_index * 55
        body.append(
            f'<line x1="{legend_left}" y1="{legend_y}" x2="{legend_left + 32}" '
            f'y2="{legend_y}" stroke="{color}" stroke-width="3"{dash_attribute}/>'
        )
        body.append(
            f'<text x="{legend_left + 42}" y="{legend_y - 2}" class="label">{_x(label)}</text>'
        )
        body.append(
            f'<text x="{legend_left + 42}" y="{legend_y + 15}" class="small">'
            f'AUPRC {row.auprc:.3f}; prevalence {row.n_positive_tasks / row.n_tasks:.3f}; '
            f'n={row.n_tasks}</text>'
        )
    body.append(
        f'<line x1="{plot_left}" y1="{plot_top + plot_size}" x2="{plot_left + plot_size}" '
        f'y2="{plot_top + plot_size}" class="axis"/>'
    )
    body.append(
        f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" '
        f'y2="{plot_top + plot_size}" class="axis"/>'
    )
    body.append(
        f'<text x="{plot_left + plot_size / 2}" y="{height - 25}" text-anchor="middle" '
        'class="label">Recall across tasks</text>'
    )
    body.append(
        f'<text x="22" y="{plot_top + plot_size / 2}" text-anchor="middle" class="label" '
        f'transform="rotate(-90 22 {plot_top + plot_size / 2})">Precision across fired tasks</text>'
    )
    svg = _document(
        width,
        height,
        title,
        "Task-level precision-recall curves with axes fixed from zero to one. Diamonds identify locked-threshold operating points.",
        body,
        attributes='data-axis-min="0" data-axis-max="1"',
    )
    svg_output, data_output = _paths(svg_path, sidecar_path)
    data = {
        "schema_version": SCHEMA_VERSION,
        "figure_type": "precision_recall",
        "statistical_unit": "task",
        "axes": {
            "recall": {"minimum": 0.0, "maximum": 1.0},
            "precision": {"minimum": 0.0, "maximum": 1.0},
        },
        "series": [_summary_dict(row) for row in sorted(rows, key=lambda item: (item.method, item.model, item.benchmark))],
    }
    return _write_artifact(svg_output, data_output, svg, data, width, height)


def _class_order(value: str) -> tuple[int, str]:
    preferred = {
        "baseline": 0,
        "active": 1,
        "passive-behavioral": 2,
        "passive-behavioural": 2,
        "passive-observational": 3,
    }
    return preferred.get(value.lower(), 10), value


def _bar_bounds(rows: Sequence[DeploymentBar]) -> tuple[float, float]:
    values = [0.0]
    for row in rows:
        values.extend(
            value
            for value in (row.value, row.ci_low, row.ci_high)
            if value is not None
        )
    low, high = min(values), max(values)
    if low == high == 0:
        return 0.0, 1.0
    span = high - low
    padded_low = min(0.0, low - 0.08 * span)
    padded_high = max(0.0, high + 0.08 * span)
    if padded_high == padded_low:
        padded_high = padded_low + 1.0
    return padded_low, padded_high


def write_deployment_grouped_bars(
    bars: Iterable[DeploymentBar],
    svg_path: str | os.PathLike[str],
    *,
    sidecar_path: str | os.PathLike[str] | None = None,
    title: str = "Deployment performance by observation class and operator",
    y_label: str = "Task performance",
) -> FigureArtifact:
    """Draw grouped deployment bars; the quantitative axis always includes zero."""

    rows = tuple(bars)
    if not rows:
        raise FigureInputError("deployment figure requires at least one bar")
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        if not isinstance(row, DeploymentBar):
            raise FigureInputError("bars must contain DeploymentBar records")
        key = (row.observation_class, row.operator, row.method)
        if key in seen:
            raise FigureInputError(f"duplicate deployment bar: {key!r}")
        seen.add(key)
    classes = sorted({row.observation_class for row in rows}, key=_class_order)
    operators = sorted({row.operator for row in rows})
    grouped = {
        observation_class: sorted(
            (row for row in rows if row.observation_class == observation_class),
            key=lambda row: (row.operator, row.method),
        )
        for observation_class in classes
    }
    width = max(760, 160 + len(classes) * 190)
    height = 600
    plot_left, plot_top = 82, 72
    plot_right, plot_bottom = width - 40, 480
    plot_height = plot_bottom - plot_top
    y_min, y_max = _bar_bounds(rows)
    body = [f'<text x="24" y="32" class="title">{_x(title)}</text>']
    body.append(
        '<text x="24" y="52" class="subtitle">Bars are task-level outcomes; error bars show supplied task-level intervals.</text>'
    )
    for tick_index in range(6):
        value = y_min + (y_max - y_min) * tick_index / 5
        y = plot_bottom - (value - y_min) / (y_max - y_min) * plot_height
        body.append(
            f'<line x1="{plot_left}" y1="{y:.2f}" x2="{plot_right}" y2="{y:.2f}" '
            'class="grid"/>'
        )
        body.append(
            f'<text x="{plot_left - 10}" y="{y + 4:.2f}" text-anchor="end" class="small">'
            f'{_fmt(value)}</text>'
        )
    group_width = (plot_right - plot_left) / len(classes)
    zero_y = plot_bottom - (0.0 - y_min) / (y_max - y_min) * plot_height
    body.append(
        f'<line x1="{plot_left}" y1="{zero_y:.2f}" x2="{plot_right}" y2="{zero_y:.2f}" '
        'class="zero-line"/>'
    )
    body.append(
        f'<text x="{plot_left - 10}" y="{zero_y + 4:.2f}" text-anchor="end" class="small">0</text>'
    )
    operator_colors = {operator: _PALETTE[index % len(_PALETTE)] for index, operator in enumerate(operators)}
    for group_index, observation_class in enumerate(classes):
        group_x = plot_left + group_index * group_width
        group_rows = grouped[observation_class]
        gap = 6
        available = group_width * 0.78
        bar_width = min(44.0, (available - gap * (len(group_rows) - 1)) / len(group_rows))
        total_width = bar_width * len(group_rows) + gap * (len(group_rows) - 1)
        start_x = group_x + (group_width - total_width) / 2
        for row_index, row in enumerate(group_rows):
            x = start_x + row_index * (bar_width + gap)
            value_y = plot_bottom - (row.value - y_min) / (y_max - y_min) * plot_height
            rect_y = min(value_y, zero_y)
            rect_height = max(abs(value_y - zero_y), 0.8)
            color = operator_colors[row.operator]
            body.append(
                f'<rect x="{x:.2f}" y="{rect_y:.2f}" width="{bar_width:.2f}" '
                f'height="{rect_height:.2f}" fill="{color}" stroke="{_INK}" stroke-width="0.6">'
                f'<title>{_x(row.observation_class)} / {_x(row.operator)} / {_x(row.method)}: '
                f'{row.value:.4g}, n={row.n_tasks}</title></rect>'
            )
            if row.ci_low is not None and row.ci_high is not None:
                low_y = plot_bottom - (row.ci_low - y_min) / (y_max - y_min) * plot_height
                high_y = plot_bottom - (row.ci_high - y_min) / (y_max - y_min) * plot_height
                center = x + bar_width / 2
                body.extend(
                    [
                        f'<line x1="{center:.2f}" y1="{high_y:.2f}" x2="{center:.2f}" y2="{low_y:.2f}" stroke="{_INK}"/>',
                        f'<line x1="{center - 5:.2f}" y1="{high_y:.2f}" x2="{center + 5:.2f}" y2="{high_y:.2f}" stroke="{_INK}"/>',
                        f'<line x1="{center - 5:.2f}" y1="{low_y:.2f}" x2="{center + 5:.2f}" y2="{low_y:.2f}" stroke="{_INK}"/>',
                    ]
                )
            label_y = value_y - 7 if row.value >= 0 else value_y + 14
            body.append(
                f'<text x="{x + bar_width / 2:.2f}" y="{label_y:.2f}" text-anchor="middle" '
                f'class="small">{_fmt(row.value)}</text>'
            )
            body.append(
                f'<text x="{x + bar_width / 2:.2f}" y="{plot_bottom + 16}" text-anchor="end" '
                f'class="small" transform="rotate(-35 {x + bar_width / 2:.2f} {plot_bottom + 16})">'
                f'{_x(row.method)}</text>'
            )
        body.append(
            f'<text x="{group_x + group_width / 2:.2f}" y="{height - 48}" text-anchor="middle" '
            f'class="panel-title">{_x(observation_class)}</text>'
        )
    body.append(
        f'<text x="20" y="{(plot_top + plot_bottom) / 2}" text-anchor="middle" class="label" '
        f'transform="rotate(-90 20 {(plot_top + plot_bottom) / 2})">{_x(y_label)}</text>'
    )
    legend_x = plot_left
    for operator_index, operator in enumerate(operators):
        x = legend_x + operator_index * 150
        body.append(
            f'<rect x="{x}" y="{height - 24}" width="13" height="13" '
            f'fill="{operator_colors[operator]}" stroke="{_INK}" stroke-width="0.5"/>'
        )
        body.append(
            f'<text x="{x + 19}" y="{height - 13}" class="small">operator: {_x(operator)}</text>'
        )
    svg = _document(
        width,
        height,
        title,
        "Grouped task-level deployment bars by observation class and recovery operator. The value axis includes a marked zero line.",
        body,
        attributes=f'data-axis-min="{y_min}" data-axis-max="{y_max}"',
    )
    svg_output, data_output = _paths(svg_path, sidecar_path)
    data = {
        "schema_version": SCHEMA_VERSION,
        "figure_type": "deployment_grouped_bars",
        "statistical_unit": "task",
        "axis": {"minimum": y_min, "maximum": y_max, "zero_included": True, "label": y_label},
        "rows": [asdict(row) for row in sorted(rows, key=lambda row: (_class_order(row.observation_class), row.operator, row.method))],
    }
    return _write_artifact(svg_output, data_output, svg, data, width, height)


def _category_order(value: str) -> tuple[int, str]:
    preferred = {
        "short": 0,
        "medium": 1,
        "long": 2,
        "low": 0,
        "easy": 0,
        "moderate": 1,
        "hard": 2,
        "high": 2,
    }
    return preferred.get(value.lower(), 10), value


def _hex_rgb(color: str) -> tuple[int, int, int]:
    return tuple(int(color[index : index + 2], 16) for index in (1, 3, 5))


def _mix(left: str, right: str, weight: float) -> str:
    weight = min(max(weight, 0.0), 1.0)
    a, b = _hex_rgb(left), _hex_rgb(right)
    rgb = tuple(round(a[index] * (1 - weight) + b[index] * weight) for index in range(3))
    return "#" + "".join(f"{value:02X}" for value in rgb)


def _diverging(value: float, limit: float) -> str:
    intensity = min(abs(value) / limit, 1.0) if limit else 0.0
    endpoint = "#0072B2" if value < 0 else "#D55E00"
    return _mix("#FFFFFF", endpoint, intensity)


def write_method_advantage_heatmap(
    cells: Iterable[AdvantageCell],
    svg_path: str | os.PathLike[str],
    *,
    sidecar_path: str | os.PathLike[str] | None = None,
    title: str = "Method advantage by trace length, context, and difficulty",
) -> FigureArtifact:
    """Draw method panels with a symmetric, zero-centered advantage scale."""

    rows = tuple(cells)
    if not rows:
        raise FigureInputError("advantage heatmap requires at least one cell")
    seen: set[tuple[str, str, str, str]] = set()
    for row in rows:
        if not isinstance(row, AdvantageCell):
            raise FigureInputError("cells must contain AdvantageCell records")
        key = (row.method, row.trace_length, row.context, row.difficulty)
        if key in seen:
            raise FigureInputError(f"duplicate heatmap cell: {key!r}")
        seen.add(key)
    methods = sorted({row.method for row in rows})
    traces = sorted({row.trace_length for row in rows}, key=_category_order)
    row_categories = sorted(
        {(row.context, row.difficulty) for row in rows},
        key=lambda pair: (_category_order(pair[0]), _category_order(pair[1])),
    )
    cell_width, cell_height = 86, 38
    label_width = 180
    panel_width = label_width + len(traces) * cell_width + 32
    panel_height = 58 + len(row_categories) * cell_height
    width = max(700, panel_width + 52)
    height = 92 + len(methods) * panel_height + 90
    limit = _nice_limit(max(abs(row.advantage) for row in rows) * 1.05)
    lookup = {
        (row.method, row.trace_length, row.context, row.difficulty): row for row in rows
    }
    body = [f'<text x="24" y="32" class="title">{_x(title)}</text>']
    body.append(
        '<text x="24" y="52" class="subtitle">Blue is negative, white is zero, vermillion is positive; all panels share one symmetric scale.</text>'
    )
    for method_index, method in enumerate(methods):
        panel_x = 34
        panel_y = 76 + method_index * panel_height
        body.append(
            f'<text x="{panel_x}" y="{panel_y + 16}" class="panel-title">{_x(method)}</text>'
        )
        grid_x = panel_x + label_width
        grid_y = panel_y + 42
        for trace_index, trace in enumerate(traces):
            body.append(
                f'<text x="{grid_x + trace_index * cell_width + cell_width / 2:.1f}" '
                f'y="{grid_y - 9}" text-anchor="middle" class="label">{_x(trace)}</text>'
            )
        for category_index, (context, difficulty) in enumerate(row_categories):
            y = grid_y + category_index * cell_height
            body.append(
                f'<text x="{grid_x - 8}" y="{y + cell_height / 2 + 4:.1f}" '
                f'text-anchor="end" class="label">{_x(context)} · {_x(difficulty)}</text>'
            )
            for trace_index, trace in enumerate(traces):
                x = grid_x + trace_index * cell_width
                row = lookup.get((method, trace, context, difficulty))
                if row is None:
                    fill, text_value, text_color = "#EFEFEF", "—", _MUTED
                    tooltip = f"{method}; {trace}; {context}; {difficulty}: missing"
                else:
                    fill = _diverging(row.advantage, limit)
                    text_value = _fmt(row.advantage, signed=True)
                    text_color = "#FFFFFF" if abs(row.advantage) / limit > 0.58 else _INK
                    tooltip = (
                        f"{method}; trace {trace}; context {context}; difficulty {difficulty}: "
                        f"{row.advantage:+.4g}, n={row.n_tasks}"
                    )
                body.append(
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_width}" height="{cell_height}" '
                    f'fill="{fill}" stroke="#B7B7B7" stroke-width="0.8"><title>{_x(tooltip)}</title></rect>'
                )
                body.append(
                    f'<text x="{x + cell_width / 2:.1f}" y="{y + cell_height / 2 + 4:.1f}" '
                    f'text-anchor="middle" class="label" fill="{text_color}" '
                    f'style="fill:{text_color}">{text_value}</text>'
                )

    gradient = (
        '<linearGradient id="advantage-gradient" x1="0%" x2="100%">'
        '<stop offset="0%" stop-color="#0072B2"/><stop offset="50%" stop-color="#FFFFFF"/>'
        '<stop offset="100%" stop-color="#D55E00"/></linearGradient>'
    )
    legend_y = height - 54
    legend_x = width / 2 - 150
    body.append(
        f'<rect x="{legend_x:.1f}" y="{legend_y}" width="300" height="15" '
        'fill="url(#advantage-gradient)" stroke="#777" stroke-width="0.7"/>'
    )
    body.append(
        f'<line x1="{width / 2:.1f}" y1="{legend_y - 3}" x2="{width / 2:.1f}" '
        f'y2="{legend_y + 18}" class="zero-line"/>'
    )
    body.extend(
        [
            f'<text x="{legend_x:.1f}" y="{legend_y + 32}" text-anchor="middle" class="small">{-limit:g}</text>',
            f'<text x="{width / 2:.1f}" y="{legend_y + 32}" text-anchor="middle" class="small">0</text>',
            f'<text x="{legend_x + 300:.1f}" y="{legend_y + 32}" text-anchor="middle" class="small">+{limit:g}</text>',
        ]
    )
    svg = _document(
        width,
        height,
        title,
        "Heatmap panels show method advantage across trace length columns and context-by-difficulty rows. The shared color scale is symmetric around zero.",
        body,
        definitions=gradient,
        attributes=f'data-axis-min="{-limit}" data-axis-max="{limit}" data-zero-centered="true"',
    )
    svg_output, data_output = _paths(svg_path, sidecar_path)
    data = {
        "schema_version": SCHEMA_VERSION,
        "figure_type": "method_advantage_heatmap",
        "statistical_unit": "task",
        "color_axis": {"minimum": -limit, "maximum": limit, "center": 0.0},
        "dimensions": {
            "methods": methods,
            "trace_lengths": traces,
            "context_difficulty_rows": [list(pair) for pair in row_categories],
        },
        "rows": [asdict(row) for row in sorted(rows, key=lambda row: (row.method, _category_order(row.trace_length), _category_order(row.context), _category_order(row.difficulty)))],
    }
    return _write_artifact(svg_output, data_output, svg, data, width, height)


__all__ = [
    "AdvantageCell",
    "DeploymentBar",
    "FigureArtifact",
    "FigureInputError",
    "write_deployment_grouped_bars",
    "write_method_advantage_heatmap",
    "write_observer_effect_forest",
    "write_observer_metric_effect_forest",
    "write_pr_curves",
]
