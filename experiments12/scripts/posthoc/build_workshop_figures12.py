#!/usr/bin/env python3
"""Build readable NewInML figures from frozen Experiment 12 summaries only.

This provider-free script reads already-collected, machine-readable analysis
artifacts.  It does not import the experiment runner, contact a model provider,
or rerun any trajectory.  SVG is the reproducible source format; pass the
optional path to a ``resvg`` executable to also render paper-ready PNG files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import subprocess
from typing import Callable, Iterable, Mapping, Sequence


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
DERIVED_ROOT = EXPERIMENT_ROOT / "data_results" / "derived"
SOURCE_ROOT = DERIVED_ROOT / "workshop-figures12"
GRAPH_ROOT = EXPERIMENT_ROOT / "graphs"

OPERATOR_EFFECTS = (
    DERIVED_ROOT
    / "deployment-paper-post-analysis-v1"
    / "online-operator-effects.csv"
)
ONLINE_PERFORMANCE = (
    DERIVED_ROOT
    / "deployment-paper-post-analysis-v1"
    / "online-performance.csv"
)
PAPER_MATERIALS = DERIVED_ROOT / "PAPER_MATERIALS12.json"
CONTROLLED_ORACLE = SOURCE_ROOT / "controlled-oracle-summary.json"

METHOD_ORDER = (
    "active_recompute",
    "frozen_probe:recompute",
    "frozen_probe:current_copy",
    "frozen_quiz",
    "trace_judge",
    "trace_rules",
    "turn_clock",
    "context_use",
)
METHOD_LABELS = {
    "active_recompute": "Active recompute",
    "frozen_probe:recompute": "Frozen recompute",
    "frozen_probe:current_copy": "Frozen current-copy",
    "frozen_quiz": "Frozen quiz",
    "trace_judge": "Trace judge",
    "trace_rules": "Trace rules",
    "turn_clock": "Turn clock",
    "context_use": "Context use",
}
CLASS_ORDER = ("active", "passive", "baseline")
CLASS_LABELS = {"active": "ACTIVE", "passive": "PASSIVE", "baseline": "BASELINE"}
CLASS_COLORS = {"active": "#d55e00", "passive": "#0072b2", "baseline": "#67788a"}
CLASS_BACKGROUNDS = {"active": "#fff3e8", "passive": "#eef7fc", "baseline": "#f1f4f7"}
METHOD_CLASSES = {
    "active_recompute": "active",
    "frozen_probe:recompute": "passive",
    "frozen_probe:current_copy": "passive",
    "frozen_quiz": "passive",
    "trace_judge": "passive",
    "trace_rules": "passive",
    "turn_clock": "baseline",
    "context_use": "baseline",
}
DISPLAY_METHODS = tuple(method for method in METHOD_ORDER if method != "frozen_quiz")
MODEL_LABELS = {
    "deepseek-v4-flash-0731": "DeepSeek V4",
    "gpt-5.6-luna": "GPT-5.6 Luna",
    "gpt-5.6-terra": "GPT-5.6 Terra",
    "gpt-oss-120b": "GPT-OSS 120B",
}


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def text(
    x: float,
    y: float,
    value: object,
    *,
    size: int = 24,
    weight: int = 400,
    anchor: str = "start",
    fill: str = "#17324d",
    rotate: int | None = None,
) -> str:
    transform = f' transform="rotate({rotate} {x:g} {y:g})"' if rotate else ""
    return (
        f'<text x="{x:g}" y="{y:g}" font-size="{size}" font-weight="{weight}" '
        f'text-anchor="{anchor}" fill="{fill}"{transform}>{esc(value)}</text>'
    )


def line(x1: float, y1: float, x2: float, y2: float, **attrs: object) -> str:
    rendered = " ".join(f'{key.replace("_", "-")}="{esc(value)}"' for key, value in attrs.items())
    return f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" {rendered}/>'


def rect(x: float, y: float, width: float, height: float, **attrs: object) -> str:
    rendered = " ".join(f'{key.replace("_", "-")}="{esc(value)}"' for key, value in attrs.items())
    return f'<rect x="{x:g}" y="{y:g}" width="{width:g}" height="{height:g}" {rendered}/>'


def circle(x: float, y: float, radius: float, **attrs: object) -> str:
    rendered = " ".join(f'{key.replace("_", "-")}="{esc(value)}"' for key, value in attrs.items())
    return f'<circle cx="{x:g}" cy="{y:g}" r="{radius:g}" {rendered}/>'


def svg(width: int, height: int, body: Iterable[str], *, description: str) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            f"<title>{esc(description)}</title>",
            "<style>text { font-family: 'Liberation Sans', Arial, sans-serif; }</style>",
            rect(0, 0, width, height, fill="#ffffff"),
            *body,
            "</svg>",
            "",
        ]
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_materials() -> Mapping[str, object]:
    with PAPER_MATERIALS.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("PAPER_MATERIALS12.json must contain an object")
    return value


def read_controlled_oracle() -> Mapping[str, object]:
    with CONTROLLED_ORACLE.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("controlled-oracle-summary.json must contain an object")
    return value


def recovery_figure() -> str:
    rows = read_csv(OPERATOR_EFFECTS)
    selected = {
        (row["method"], row["operator"]): row
        for row in rows
        if row["operator"] in {"lossy_compaction", "public_state_reground"}
    }
    # Current-copy is not part of the deployment grid; all other methods are.
    expected = {(method, operator) for method in METHOD_ORDER if method != "frozen_probe:current_copy" for operator in ("lossy_compaction", "public_state_reground")}
    if set(selected) != expected:
        raise ValueError("online operator-effect treatment grid changed")

    deployed_methods = [
        method
        for method in DISPLAY_METHODS
        if method != "frozen_probe:current_copy"
    ]
    mean_effect = {
        method: sum(
            float(selected[(method, operator)]["effect"])
            for operator in ("lossy_compaction", "public_state_reground")
        )
        / 2
        for method in deployed_methods
    }
    methods = [
        method
        for class_name in CLASS_ORDER
        for method in sorted(
            (candidate for candidate in deployed_methods if METHOD_CLASSES[candidate] == class_name),
            key=lambda candidate: (-mean_effect[candidate], METHOD_LABELS[candidate]),
        )
    ]
    width, height = 1500, 930
    left, right, top, bottom = 330, 80, 215, 120
    plot_width = width - left - right
    x_min, x_max = -0.80, 0.30
    row_step = (height - top - bottom) / len(methods)

    def xmap(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    body: list[str] = [
        text(55, 62, "Recovery changes performance—and depends on the monitor", size=38, weight=700),
        text(55, 104, "Natural-policy deployment: operator effect relative to monitored no-action", size=25, fill="#536679"),
        text(55, 142, "GPT-5.6 Luna × Evolving Intent; n = 40 paired source tasks; within class, methods are sorted by mean displayed effect", size=21, fill="#536679"),
    ]

    for tick in (-0.75, -0.50, -0.25, 0.00, 0.25):
        x = xmap(tick)
        body.append(line(x, top - 15, x, height - bottom + 5, stroke="#dbe3ea", stroke_width=2))
        body.append(text(x, height - 74, f"{tick * 100:+.0f}", size=21, anchor="middle", fill="#536679"))
    body.append(line(xmap(0), top - 22, xmap(0), height - bottom + 8, stroke="#17324d", stroke_width=3))
    body.append(text(left + plot_width / 2, height - 28, "change in task success (percentage points)", size=23, anchor="middle"))

    colors = {"lossy_compaction": "#d55e00", "public_state_reground": "#0072b2"}
    labels = {"lossy_compaction": "Lossy compaction", "public_state_reground": "Public-state reground"}
    legend_x = 720
    for index, operator in enumerate(("lossy_compaction", "public_state_reground")):
        x = legend_x + index * 330
        body.append(circle(x, 181, 9, fill=colors[operator]))
        body.append(text(x + 20, 189, labels[operator], size=22, fill="#334e68"))

    for index, method in enumerate(methods):
        y = top + row_step * (index + 0.5)
        class_name = METHOD_CLASSES[method]
        body.append(rect(45, y - row_step / 2, width - 90, row_step, fill=CLASS_BACKGROUNDS[class_name]))
        if index % 2:
            body.append(rect(45, y - row_step / 2, width - 90, row_step, fill="#ffffff", fill_opacity="0.42"))
        if index == 0 or METHOD_CLASSES[methods[index - 1]] != class_name:
            body.append(line(45, y - row_step / 2, width - 45, y - row_step / 2, stroke=CLASS_COLORS[class_name], stroke_width=4))
            class_indices = [position for position, candidate in enumerate(methods) if METHOD_CLASSES[candidate] == class_name]
            class_center = top + row_step * ((min(class_indices) + max(class_indices) + 1) / 2)
            body.append(text(28, class_center, CLASS_LABELS[class_name], size=17, weight=700, anchor="middle", fill=CLASS_COLORS[class_name], rotate=-90))
        body.append(text(left - 28, y + 8, METHOD_LABELS[method], size=23, anchor="end", weight=600 if method == "active_recompute" else 400))
        for offset, operator in ((-11, "lossy_compaction"), (11, "public_state_reground")):
            row = selected[(method, operator)]
            effect = float(row["effect"])
            low = float(row["ci_low"])
            high = float(row["ci_high"])
            color = colors[operator]
            yy = y + offset
            body.append(line(xmap(low), yy, xmap(high), yy, stroke=color, stroke_width=5, stroke_linecap="round"))
            body.append(line(xmap(low), yy - 7, xmap(low), yy + 7, stroke=color, stroke_width=3))
            body.append(line(xmap(high), yy - 7, xmap(high), yy + 7, stroke=color, stroke_width=3))
            body.append(circle(xmap(effect), yy, 9, fill=color, stroke="#ffffff", stroke_width=2))

    return svg(width, height, body, description="Paired recovery-operator effects relative to monitored no-action")


def heat_color(value: float, minimum: float = 0.10, maximum: float = 0.95) -> str:
    # ColorBrewer Blues, linearly interpolated from a near-white low end.
    low = (239, 246, 255)
    high = (8, 81, 156)
    ratio = max(0.0, min(1.0, (value - minimum) / (maximum - minimum)))
    rgb = tuple(round(a + (b - a) * ratio) for a, b in zip(low, high))
    return "#" + "".join(f"{channel:02x}" for channel in rgb)


def signal_figure(materials: Mapping[str, object]) -> str:
    signal = materials["signal_quality"]
    assert isinstance(signal, dict)
    rows = signal["all_primary_metrics"]
    assert isinstance(rows, list)
    by_cell = {(row["benchmark"], row["model"], row["method"]): row for row in rows}
    unsorted_strata = [
        ("evolving_intent_gsm8k", "deepseek-v4-flash-0731"),
        ("evolving_intent_gsm8k", "gpt-5.6-luna"),
        ("evolving_intent_gsm8k", "gpt-5.6-terra"),
        ("evolving_intent_gsm8k", "gpt-oss-120b"),
        ("bfcl_multi_turn", "gpt-5.6-luna"),
        ("bfcl_multi_turn", "gpt-5.6-terra"),
        ("bfcl_multi_turn", "gpt-oss-120b"),
    ]
    expected = {(benchmark, model, method) for benchmark, model in unsorted_strata for method in METHOD_ORDER}
    if set(by_cell) != expected:
        raise ValueError("primary signal-quality treatment grid changed")

    mean_auprc = {
        method: sum(float(by_cell[(benchmark, model, method)]["auprc"]) for benchmark, model in unsorted_strata)
        / len(unsorted_strata)
        for method in DISPLAY_METHODS
    }
    methods = [
        method
        for class_name in CLASS_ORDER
        for method in sorted(
            (candidate for candidate in DISPLAY_METHODS if METHOD_CLASSES[candidate] == class_name),
            key=lambda candidate: (-mean_auprc[candidate], METHOD_LABELS[candidate]),
        )
    ]
    strata = [
        stratum
        for benchmark in ("evolving_intent_gsm8k", "bfcl_multi_turn")
        for stratum in sorted(
            (candidate for candidate in unsorted_strata if candidate[0] == benchmark),
            key=lambda candidate: (
                -max(float(by_cell[(candidate[0], candidate[1], method)]["auprc"]) for method in methods),
                MODEL_LABELS[candidate[1]],
            ),
        )
    ]

    width, height = 1700, 1030
    left, top = 370, 285
    cell_width, cell_height = 175, 90
    body: list[str] = [
        text(55, 62, "No monitoring method dominates across settings", size=38, weight=700),
        text(55, 104, "Each cell reports AUPRC, then precision (P) and recall (R) at the matched firing rate", size=25, fill="#536679"),
        text(55, 142, "Methods are grouped by taxonomy and sorted by mean AUPRC; Frozen quiz is excluded", size=21, fill="#536679"),
        rect(55, 163, 285, 66, fill="#f6f8fa", stroke="#9aa8b5", stroke_width=2, rx=6),
        text(197, 190, "Perfect-label ceiling = 1.00", size=18, weight=700, anchor="middle"),
        text(197, 216, "best = 0.94; gap = 0.06 (not an arm)", size=15, anchor="middle", fill="#536679"),
    ]

    for class_name in CLASS_ORDER:
        class_columns = [column for column, method in enumerate(methods) if METHOD_CLASSES[method] == class_name]
        x1 = left + min(class_columns) * cell_width + 8
        x2 = left + (max(class_columns) + 1) * cell_width - 8
        body.append(text((x1 + x2) / 2, top - 99, CLASS_LABELS[class_name], size=18, weight=700, anchor="middle", fill=CLASS_COLORS[class_name]))
        body.append(line(x1, top - 86, x2, top - 86, stroke=CLASS_COLORS[class_name], stroke_width=5))

    for column, method in enumerate(methods):
        x = left + column * cell_width + cell_width / 2
        label = METHOD_LABELS[method].replace(" ", "\n", 1)
        parts = label.split("\n")
        body.append(text(x, top - 54, parts[0], size=19, weight=600, anchor="middle"))
        if len(parts) > 1:
            body.append(text(x, top - 29, parts[1], size=19, weight=600, anchor="middle"))

    for row_index, (benchmark, model) in enumerate(strata):
        y = top + row_index * cell_height
        benchmark_label = "Evolving" if benchmark == "evolving_intent_gsm8k" else "BFCL"
        body.append(text(left - 25, y + 49, f"{benchmark_label} · {MODEL_LABELS[model]}", size=22, anchor="end"))
        values = [float(by_cell[(benchmark, model, method)]["auprc"]) for method in methods]
        maximum = max(values)
        for column, (method, value) in enumerate(zip(methods, values)):
            x = left + column * cell_width
            fill = heat_color(value)
            body.append(rect(x + 3, y + 3, cell_width - 6, cell_height - 6, fill=fill, stroke="#ffffff", stroke_width=2))
            if abs(value - maximum) < 1e-12:
                body.append(rect(x + 5, y + 5, cell_width - 10, cell_height - 10, fill="none", stroke="#102a43", stroke_width=5, rx=5))
            row = by_cell[(benchmark, model, method)]
            foreground = "#ffffff" if value >= 0.48 else "#17324d"
            body.append(text(x + cell_width / 2, y + 41, f"{value:.2f}", size=22, weight=700 if abs(value - maximum) < 1e-12 else 400, anchor="middle", fill=foreground))
            body.append(text(x + cell_width / 2, y + 68, f"P {float(row['precision']):.2f}  R {float(row['recall']):.2f}", size=15, anchor="middle", fill=foreground))
        if row_index == 3:
            body.append(line(55, y + cell_height, width - 55, y + cell_height, stroke="#7b8794", stroke_width=3, stroke_dasharray="7 7"))

    legend_y = height - 73
    body.append(text(55, legend_y + 8, "AUPRC", size=21, weight=600))
    for index in range(11):
        value = 0.10 + index * 0.085
        x = 150 + index * 38
        body.append(rect(x, legend_y - 16, 38, 28, fill=heat_color(value)))
    body.append(text(150, legend_y + 42, "0.10", size=18, anchor="middle", fill="#536679"))
    body.append(text(150 + 11 * 38, legend_y + 42, "0.95", size=18, anchor="middle", fill="#536679"))
    body.append(text(width - 55, legend_y + 8, "Active wins 3/7 slices; passive or baseline methods win 4/7", size=22, anchor="end", weight=600, fill="#334e68"))
    return svg(width, height, body, description="AUPRC heatmap for all primary model-by-task slices and monitoring methods")


def observer_effect_figure(materials: Mapping[str, object]) -> str:
    observer = materials["observer_effect"]
    assert isinstance(observer, dict)
    rows = observer["effects"]
    assert isinstance(rows, list)
    if len(rows) != 7:
        raise ValueError("powered observer-effect grid changed")
    ordered: list[Mapping[str, object]] = []
    for benchmark in ("evolving_intent_gsm8k", "bfcl_multi_turn"):
        group = [row for row in rows if row["benchmark"] == benchmark]
        ordered.extend(sorted(group, key=lambda row: (float(row["effect"]), MODEL_LABELS[str(row["model"])])))

    width, height = 1500, 850
    left, right, top, bottom = 385, 80, 205, 115
    plot_width = width - left - right
    x_min, x_max = -0.42, 0.17
    row_step = (height - top - bottom) / len(ordered)

    def xmap(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    body: list[str] = [
        text(55, 60, "Active observation usually reduces task success", size=38, weight=700),
        text(55, 103, "Paired active-recompute minus clean-arm success; whiskers are task-bootstrap 95% intervals", size=24, fill="#536679"),
        text(55, 143, "Within each benchmark, models are sorted from the largest decrease to the largest increase", size=21, fill="#536679"),
    ]
    for index, row in enumerate(ordered):
        y = top + row_step * (index + 0.5)
        background = "#f5f9fc" if row["benchmark"] == "evolving_intent_gsm8k" else "#fff7ef"
        body.append(rect(48, y - row_step / 2, width - 96, row_step, fill=background))
    for tick in (-0.40, -0.30, -0.20, -0.10, 0.00, 0.10):
        x = xmap(tick)
        body.append(line(x, top - 10, x, height - bottom + 4, stroke="#dbe3ea", stroke_width=2))
        body.append(text(x, height - 68, f"{tick * 100:+.0f}", size=20, anchor="middle", fill="#536679"))
    body.append(line(xmap(0), top - 17, xmap(0), height - bottom + 6, stroke="#17324d", stroke_width=4))
    body.append(text(left + plot_width / 2, height - 25, "change in task success (percentage points)", size=22, anchor="middle"))

    for index, row in enumerate(ordered):
        y = top + row_step * (index + 0.5)
        benchmark = str(row["benchmark"])
        if index and ordered[index - 1]["benchmark"] != benchmark:
            body.append(line(48, y - row_step / 2, width - 48, y - row_step / 2, stroke="#7b8794", stroke_width=3, stroke_dasharray="7 7"))
        benchmark_label = "Evolving" if benchmark == "evolving_intent_gsm8k" else "BFCL"
        body.append(text(left - 25, y + 8, f"{benchmark_label} · {MODEL_LABELS[str(row['model'])]}", size=22, anchor="end"))
        effect = float(row["effect"])
        low = float(row["ci_low"])
        high = float(row["ci_high"])
        color = "#d55e00" if effect < 0 else "#0072b2"
        body.append(line(xmap(low), y, xmap(high), y, stroke=color, stroke_width=6, stroke_linecap="round"))
        body.append(line(xmap(low), y - 8, xmap(low), y + 8, stroke=color, stroke_width=3))
        body.append(line(xmap(high), y - 8, xmap(high), y + 8, stroke=color, stroke_width=3))
        body.append(circle(xmap(effect), y, 11, fill=color, stroke="#ffffff", stroke_width=3))
        body.append(text(width - 55, y + 8, f"{effect * 100:+.1f} pp", size=21, anchor="end", weight=700 if bool(row["ci_excludes_zero"]) else 400, fill=color))

    body.append(text(width - 55, 180, "6/7 estimates negative; 3 intervals exclude zero below", size=20, anchor="end", weight=600, fill="#334e68"))
    return svg(width, height, body, description="Powered paired observer effects for active recomputation")


def overhead_figure(materials: Mapping[str, object]) -> str:
    overhead = materials["observer_overhead"]
    assert isinstance(overhead, dict)
    rows = overhead["method_aggregates"]
    assert isinstance(rows, list)
    by_method = {row["method"]: row for row in rows}
    if set(by_method) != set(METHOD_ORDER):
        raise ValueError("observer-overhead method grid changed")

    methods = [
        method
        for class_name in CLASS_ORDER
        for method in sorted(
            (candidate for candidate in DISPLAY_METHODS if METHOD_CLASSES[candidate] == class_name),
            key=lambda candidate: (-int(by_method[candidate]["total_tokens"]), METHOD_LABELS[candidate]),
        )
    ]

    width, height = 1500, 960
    left, right, top, bottom = 340, 250, 220, 115
    plot_width = width - left - right
    maximum = 1_800_000
    row_step = (height - top - bottom) / len(methods)

    def xmap(value: float) -> float:
        return left + value / maximum * plot_width

    body: list[str] = [
        text(55, 62, "Observation itself consumes compute", size=38, weight=700),
        text(55, 104, "Total observer tokens across 392 model–task cases; annotations show recorded observer cost", size=25, fill="#536679"),
        text(55, 142, "Grouped by taxonomy and sorted by decreasing tokens; provider-backed methods each made 1,764 calls", size=21, fill="#536679"),
    ]
    for tick in (0, 500_000, 1_000_000, 1_500_000):
        x = xmap(tick)
        body.append(line(x, top - 15, x, height - bottom + 5, stroke="#dbe3ea", stroke_width=2))
        body.append(text(x, height - 66, "0" if tick == 0 else f"{tick / 1_000_000:.1f}M", size=20, anchor="middle", fill="#536679"))
    body.append(text(left + plot_width / 2, height - 25, "total observer tokens", size=23, anchor="middle"))
    body.append(text(width - right + 55, top - 30, "recorded cost", size=20, anchor="middle", weight=600, fill="#536679"))

    for index, method in enumerate(methods):
        row = by_method[method]
        y = top + row_step * (index + 0.5)
        class_name = METHOD_CLASSES[method]
        body.append(rect(45, y - row_step / 2, width - 90, row_step, fill=CLASS_BACKGROUNDS[class_name]))
        if index % 2:
            body.append(rect(45, y - row_step / 2, width - 90, row_step, fill="#ffffff", fill_opacity="0.42"))
        if index == 0 or METHOD_CLASSES[methods[index - 1]] != class_name:
            body.append(line(45, y - row_step / 2, width - 45, y - row_step / 2, stroke=CLASS_COLORS[class_name], stroke_width=4))
            class_indices = [position for position, candidate in enumerate(methods) if METHOD_CLASSES[candidate] == class_name]
            class_center = top + row_step * ((min(class_indices) + max(class_indices) + 1) / 2)
            body.append(text(28, class_center, CLASS_LABELS[class_name], size=17, weight=700, anchor="middle", fill=CLASS_COLORS[class_name], rotate=-90))
        body.append(text(left - 28, y + 8, METHOD_LABELS[method], size=23, anchor="end", weight=600 if method == "active_recompute" else 400))
        tokens = int(row["total_tokens"])
        color = CLASS_COLORS[class_name] if tokens else "#b8c2cc"
        if tokens:
            body.append(rect(left, y - 17, xmap(tokens) - left, 34, fill=color, rx=4))
            body.append(text(xmap(tokens) - 12, y + 8, f"{tokens / 1_000_000:.2f}M", size=20, anchor="end", weight=600, fill="#ffffff"))
        else:
            body.append(line(left, y, left + 38, y, stroke=color, stroke_width=7, stroke_linecap="round"))
            body.append(text(left + 52, y + 8, "0", size=20, fill="#536679"))
        cost = float(row["cost_usd"])
        body.append(text(width - right + 55, y + 8, f"${cost:.2f}", size=22, anchor="middle", weight=600 if cost else 400, fill="#17324d"))

    return svg(width, height, body, description="Observer token and recorded-cost overhead by monitoring method")


def deployment_success_rows() -> dict[tuple[str, str], dict[str, str]]:
    rows = read_csv(ONLINE_PERFORMANCE)
    selected = {
        (row["method"], row["operator"]): row
        for row in rows
        if row["metric"] == "success"
    }
    expected_methods = {
        "active_recompute",
        "frozen_probe:recompute",
        "frozen_quiz",
        "trace_judge",
        "trace_rules",
        "turn_clock",
        "context_use",
    }
    expected_operators = {
        "none",
        "lossy_compaction",
        "public_state_reground",
        "good_bad_watch_feedback",
    }
    if set(selected) != {
        (method, operator)
        for method in expected_methods
        for operator in expected_operators
    }:
        raise ValueError("online deployment success grid changed")
    return selected


def draw_success_axis(
    body: list[str],
    *,
    left: float,
    right: float,
    top: float,
    bottom: float,
    minimum: float,
    maximum: float,
    ticks: Sequence[float],
    axis_label: str = "task success (%)",
) -> Callable[[float], float]:
    def ymap(value: float) -> float:
        return bottom - (value - minimum) / (maximum - minimum) * (bottom - top)

    for tick in ticks:
        y = ymap(tick)
        body.append(line(left, y, right, y, stroke="#dbe3ea", stroke_width=2))
        body.append(text(left - 18, y + 7, f"{tick * 100:.0f}", size=20, anchor="end", fill="#536679"))
    body.append(line(left, top, left, bottom, stroke="#7b8794", stroke_width=2))
    body.append(line(left, bottom, right, bottom, stroke="#7b8794", stroke_width=2))
    body.append(text(40, (top + bottom) / 2, axis_label, size=22, anchor="middle", rotate=-90))
    # Mark the intentionally truncated success-rate axis.
    body.append(line(left - 8, bottom - 4, left + 8, bottom - 14, stroke="#536679", stroke_width=3))
    body.append(line(left - 8, bottom + 8, left + 8, bottom - 2, stroke="#536679", stroke_width=3))
    return ymap


def overall_recovery_figure() -> str:
    rows = deployment_success_rows()
    methods = (
        "active_recompute",
        "frozen_probe:recompute",
        "trace_judge",
        "trace_rules",
        "turn_clock",
        "context_use",
    )
    operators = ("none", "lossy_compaction", "public_state_reground", "good_bad_watch_feedback")
    means = {
        operator: sum(float(rows[(method, operator)]["mean"]) for method in methods)
        / len(methods)
        for operator in operators
    }
    order = sorted(operators, key=lambda operator: -means[operator])
    labels = {
        "none": ("Monitored", "no state action"),
        "lossy_compaction": ("Lossy", "compaction"),
        "public_state_reground": ("Deterministic", "reconstruction"),
        "good_bad_watch_feedback": ("Quote-only", "WATCH note"),
    }
    colors = {
        "none": "#67788a",
        "lossy_compaction": "#d55e00",
        "public_state_reground": "#0072b2",
        "good_bad_watch_feedback": "#5b8c5a",
    }
    width, height = 1400, 820
    left, right, top, bottom = 150, 75, 225, 680
    body: list[str] = [
        text(55, 58, "Overall recovery success across monitoring methods", size=38, weight=700),
        text(55, 101, "Equal-weighted descriptive mean across six monitor-method cells; Frozen quiz excluded", size=24, fill="#536679"),
        text(55, 141, "GPT-5.6 Luna × Evolving Intent; n = 40 paired source tasks per cell", size=21, fill="#536679"),
        text(width - 55, 183, "BFCL recovery interventions were not evaluated", size=20, anchor="end", weight=600, fill="#a14300"),
    ]
    ymap = draw_success_axis(
        body,
        left=left,
        right=width - right,
        top=top,
        bottom=bottom,
        minimum=0.50,
        maximum=0.78,
        ticks=(0.50, 0.60, 0.70),
    )
    available = width - left - right
    centers = [left + available * fraction for fraction in (0.125, 0.375, 0.625, 0.875)]
    bar_width = 205
    for center, operator in zip(centers, order):
        value = means[operator]
        body.append(rect(center - bar_width / 2, ymap(value), bar_width, bottom - ymap(value), fill=colors[operator], rx=6))
        body.append(text(center, ymap(value) - 15, f"{value * 100:.1f}%", size=25, weight=700, anchor="middle"))
        first, second = labels[operator]
        body.append(text(center, bottom + 43, first, size=22, weight=600, anchor="middle"))
        body.append(text(center, bottom + 72, second, size=22, weight=600, anchor="middle"))
    return svg(width, height, body, description="Equal-weighted Evolving Intent deployment success by recovery operator")


def active_recovery_figure() -> str:
    rows = deployment_success_rows()
    operators = ("lossy_compaction", "public_state_reground")
    order = sorted(operators, key=lambda operator: -float(rows[("active_recompute", operator)]["mean"]))
    labels = {
        "lossy_compaction": ("Lossy", "compaction"),
        "public_state_reground": ("Deterministic", "reconstruction"),
    }
    colors = {"lossy_compaction": "#d55e00", "public_state_reground": "#0072b2"}
    width, height = 1200, 820
    left, right, top, bottom = 150, 70, 225, 680
    body: list[str] = [
        text(55, 58, "Active monitoring: recovery choice changes success", size=38, weight=700),
        text(55, 101, "Active recomputation paired with deterministic reconstruction or lossy compaction", size=24, fill="#536679"),
        text(55, 141, "GPT-5.6 Luna × Evolving Intent; n = 40 paired source tasks per cell", size=21, fill="#536679"),
        text(width - 55, 183, "BFCL recovery interventions were not evaluated", size=20, anchor="end", weight=600, fill="#a14300"),
    ]
    ymap = draw_success_axis(
        body,
        left=left,
        right=width - right,
        top=top,
        bottom=bottom,
        minimum=0.55,
        maximum=0.96,
        ticks=(0.60, 0.70, 0.80, 0.90),
    )
    centers = (430, 820)
    bar_width = 250
    for center, operator in zip(centers, order):
        row = rows[("active_recompute", operator)]
        value = float(row["mean"])
        body.append(rect(center - bar_width / 2, ymap(value), bar_width, bottom - ymap(value), fill=colors[operator], rx=6))
        low = float(row["ci_low"])
        high = float(row["ci_high"])
        body.append(line(center, ymap(low), center, ymap(high), stroke="#17324d", stroke_width=4))
        body.append(line(center - 18, ymap(low), center + 18, ymap(low), stroke="#17324d", stroke_width=4))
        body.append(line(center - 18, ymap(high), center + 18, ymap(high), stroke="#17324d", stroke_width=4))
        body.append(text(center, ymap(value) - 15, f"{value * 100:.1f}%", size=26, weight=700, anchor="middle"))
        first, second = labels[operator]
        body.append(text(center, bottom + 43, first, size=23, weight=600, anchor="middle"))
        body.append(text(center, bottom + 73, second, size=23, weight=600, anchor="middle"))
    body.append(text(width - 55, height - 30, "Descriptive difference: +12.5 percentage points for reconstruction", size=21, anchor="end", weight=600, fill="#334e68"))
    return svg(width, height, body, description="Active-monitor deployment success under two recovery operators")


def passive_recovery_figure() -> str:
    rows = deployment_success_rows()
    methods = ("frozen_probe:recompute", "trace_rules", "trace_judge")
    difference = {
        method: float(rows[(method, "lossy_compaction")]["mean"])
        - float(rows[(method, "public_state_reground")]["mean"])
        for method in methods
    }
    order = sorted(methods, key=lambda method: (-difference[method], METHOD_LABELS[method]))
    width, height = 1500, 850
    left, right, top, bottom = 155, 70, 230, 690
    body: list[str] = [
        text(55, 58, "Selected passive monitors show a different descriptive pattern", size=38, weight=700),
        text(55, 101, "Lossy compaction matches or exceeds deterministic reconstruction for these methods", size=24, fill="#536679"),
        text(55, 141, "GPT-5.6 Luna × Evolving Intent; n = 40 paired source tasks per cell; sorted by compaction − reconstruction", size=21, fill="#536679"),
        text(width - 55, 183, "BFCL recovery interventions were not evaluated", size=20, anchor="end", weight=600, fill="#a14300"),
    ]
    ymap = draw_success_axis(
        body,
        left=left,
        right=width - right,
        top=top,
        bottom=bottom,
        minimum=0.65,
        maximum=0.825,
        ticks=(0.65, 0.70, 0.75, 0.80),
    )
    available = width - left - right
    centers = [left + available * fraction for fraction in (0.20, 0.50, 0.80)]
    bar_width = 135
    gap = 10
    for center, method in zip(centers, order):
        for offset, operator, color in (
            (-bar_width / 2 - gap / 2, "lossy_compaction", "#d55e00"),
            (bar_width / 2 + gap / 2, "public_state_reground", "#0072b2"),
        ):
            value = float(rows[(method, operator)]["mean"])
            body.append(rect(center + offset - bar_width / 2, ymap(value), bar_width, bottom - ymap(value), fill=color, rx=5))
            body.append(text(center + offset, ymap(value) - 12, f"{value * 100:.1f}", size=20, weight=700, anchor="middle"))
        body.append(text(center, bottom + 48, METHOD_LABELS[method], size=22, weight=600, anchor="middle"))
        body.append(text(center, bottom + 78, f"compaction − reconstruction: {difference[method] * 100:+.1f} pp", size=18, anchor="middle", fill="#536679"))
    legend_y = height - 28
    body.append(rect(55, legend_y - 17, 28, 20, fill="#d55e00", rx=3))
    body.append(text(95, legend_y, "Lossy compaction", size=19, fill="#334e68"))
    body.append(rect(285, legend_y - 17, 28, 20, fill="#0072b2", rx=3))
    body.append(text(325, legend_y, "Deterministic reconstruction", size=19, fill="#334e68"))
    body.append(text(width - 55, legend_y, "Point estimates are descriptive; Trace judge ties", size=19, anchor="end", fill="#536679"))
    return svg(width, height, body, description="Selected passive-monitor deployment success under two recovery operators")


def controlled_oracle_figure(source: Mapping[str, object]) -> str:
    cells = source["cells"]
    assert isinstance(cells, dict)
    no_probe = cells["no_probe"]
    carried = cells["carried_probe"]
    assert isinstance(no_probe, dict) and isinstance(carried, dict)
    groups = (("No carried probe", no_probe), ("Carried active probe", carried))
    operators = ("no_intervention", "oracle_compaction", "oracle_regrounding")
    labels = {
        "no_intervention": ("No", "intervention"),
        "oracle_compaction": ("Oracle +", "compaction"),
        "oracle_regrounding": ("Oracle +", "reconstruction"),
    }
    colors = {
        "no_intervention": "#67788a",
        "oracle_compaction": "#d55e00",
        "oracle_regrounding": "#0072b2",
    }
    width, height = 1500, 860
    left, right, top, bottom = 145, 65, 250, 690
    body: list[str] = [
        text(55, 58, "Perfect timing helps, but recovery determines how much", size=38, weight=700),
        text(55, 101, "Controlled coding factorial: GPT-OSS-20B; n = 100 tasks", size=24, fill="#536679"),
        text(55, 141, "Oracle resets immediately before the first hallucinated turn in the matched no-intervention trajectory", size=21, fill="#536679"),
        text(width - 55, 181, "This oracle was not run on Evolving Intent or BFCL", size=20, anchor="end", weight=600, fill="#a14300"),
    ]
    ymap = draw_success_axis(
        body,
        left=left,
        right=width - right,
        top=top,
        bottom=bottom,
        minimum=0.80,
        maximum=0.93,
        ticks=(0.80, 0.84, 0.88, 0.92),
        axis_label="clean-turn accuracy (%)",
    )
    plot_width = width - left - right
    group_centers = (left + plot_width * 0.27, left + plot_width * 0.73)
    bar_width = 115
    spacing = 145
    for group_index, (group_label, values) in enumerate(groups):
        center = group_centers[group_index]
        baseline = float(values["no_intervention"])
        for operator_index, operator in enumerate(operators):
            x = center + (operator_index - 1) * spacing
            value = float(values[operator])
            body.append(rect(x - bar_width / 2, ymap(value), bar_width, bottom - ymap(value), fill=colors[operator], rx=5))
            body.append(text(x, ymap(value) - 13, f"{value * 100:.1f}%", size=22, weight=700, anchor="middle"))
            first, second = labels[operator]
            body.append(text(x, bottom + 34, first, size=18, weight=600, anchor="middle"))
            body.append(text(x, bottom + 58, second, size=18, weight=600, anchor="middle"))
            if operator != "no_intervention":
                body.append(text(x, bottom + 84, f"Δ {100 * (value - baseline):+.1f} pp", size=17, anchor="middle", fill=colors[operator]))
        body.append(text(center, top - 22, group_label, size=24, weight=700, anchor="middle"))
        if group_index == 0:
            body.append(line((group_centers[0] + group_centers[1]) / 2, top - 5, (group_centers[0] + group_centers[1]) / 2, bottom + 92, stroke="#c8d1da", stroke_width=2))
    body.append(text(55, height - 27, "monitor × recovery interaction: compaction −3.2 pp; reconstruction +0.1 pp", size=20, weight=600, fill="#334e68"))
    return svg(width, height, body, description="Controlled oracle-timing success with and without a carried probe under two recovery mechanisms")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(renderer: Path | None = None) -> tuple[Path, ...]:
    materials = read_materials()
    controlled_oracle = read_controlled_oracle()
    figures = (
        ("02-active-probe-observer-effect", observer_effect_figure(materials)),
        ("03-signal-quality-auprc", signal_figure(materials)),
        ("05-overall-recovery-success", overall_recovery_figure()),
        ("06-active-recovery-success", active_recovery_figure()),
        ("07-passive-recovery-success", passive_recovery_figure()),
        ("08-controlled-oracle-timing", controlled_oracle_figure(controlled_oracle)),
    )
    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    GRAPH_ROOT.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for stem, content in figures:
        source = SOURCE_ROOT / f"{stem}.svg"
        source.write_text(content, encoding="utf-8")
        outputs.append(source)
        if renderer is not None:
            destination = GRAPH_ROOT / f"{stem}.png"
            subprocess.run(
                [
                    str(renderer.resolve(strict=True)),
                    "--background",
                    "white",
                    "--zoom",
                    "2",
                    str(source),
                    str(destination),
                ],
                check=True,
            )
            outputs.append(destination)

    receipt = {
        "artifact": "newinml_workshop_figures",
        "provider_calls_made": 0,
        "sources": {
            str(ONLINE_PERFORMANCE.relative_to(EXPERIMENT_ROOT)): sha256(ONLINE_PERFORMANCE),
            str(PAPER_MATERIALS.relative_to(EXPERIMENT_ROOT)): sha256(PAPER_MATERIALS),
            str(CONTROLLED_ORACLE.relative_to(EXPERIMENT_ROOT)): sha256(CONTROLLED_ORACLE),
        },
        "outputs": {
            str(path.relative_to(EXPERIMENT_ROOT)): sha256(path)
            for path in outputs
        },
    }
    receipt_path = SOURCE_ROOT / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs.append(receipt_path)
    return tuple(outputs)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renderer", type=Path, help="optional path to the resvg executable")
    args = parser.parse_args(argv)
    for output in build(args.renderer):
        print(output.relative_to(EXPERIMENT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
