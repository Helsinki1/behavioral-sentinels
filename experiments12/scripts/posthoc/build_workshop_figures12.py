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
from typing import Iterable, Mapping, Sequence


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
DERIVED_ROOT = EXPERIMENT_ROOT / "data_results" / "derived"
SOURCE_ROOT = DERIVED_ROOT / "workshop-figures12"
GRAPH_ROOT = EXPERIMENT_ROOT / "graphs"

OPERATOR_EFFECTS = (
    DERIVED_ROOT
    / "deployment-paper-post-analysis-v1"
    / "online-operator-effects.csv"
)
PAPER_MATERIALS = DERIVED_ROOT / "PAPER_MATERIALS12.json"

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

    methods = [method for method in METHOD_ORDER if method != "frozen_probe:current_copy"]
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
        text(55, 142, "GPT-5.6 Luna × Evolving Intent; n = 40 paired source tasks; bars are paired-bootstrap 95% CIs", size=21, fill="#536679"),
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
        if index % 2 == 0:
            body.append(rect(45, y - row_step / 2, width - 90, row_step, fill="#f7f9fb"))
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


def heat_color(value: float) -> str:
    # ColorBrewer Blues, linearly interpolated from a near-white low end.
    low = (239, 246, 255)
    high = (8, 81, 156)
    ratio = max(0.0, min(1.0, value))
    rgb = tuple(round(a + (b - a) * ratio) for a, b in zip(low, high))
    return "#" + "".join(f"{channel:02x}" for channel in rgb)


def signal_figure(materials: Mapping[str, object]) -> str:
    signal = materials["signal_quality"]
    assert isinstance(signal, dict)
    rows = signal["all_primary_metrics"]
    assert isinstance(rows, list)
    by_cell = {(row["benchmark"], row["model"], row["method"]): row for row in rows}
    strata = [
        ("evolving_intent_gsm8k", "deepseek-v4-flash-0731"),
        ("evolving_intent_gsm8k", "gpt-5.6-luna"),
        ("evolving_intent_gsm8k", "gpt-5.6-terra"),
        ("evolving_intent_gsm8k", "gpt-oss-120b"),
        ("bfcl_multi_turn", "gpt-5.6-luna"),
        ("bfcl_multi_turn", "gpt-5.6-terra"),
        ("bfcl_multi_turn", "gpt-oss-120b"),
    ]
    expected = {(benchmark, model, method) for benchmark, model in strata for method in METHOD_ORDER}
    if set(by_cell) != expected:
        raise ValueError("primary signal-quality treatment grid changed")

    width, height = 1700, 940
    left, top = 370, 240
    cell_width, cell_height = 155, 80
    body: list[str] = [
        text(55, 62, "No monitoring method dominates across settings", size=38, weight=700),
        text(55, 104, "Area under the precision–recall curve (AUPRC); darker cells indicate better signal quality", size=25, fill="#536679"),
        text(55, 142, "Outlined cells are the best method within each model × task slice", size=21, fill="#536679"),
    ]

    for column, method in enumerate(METHOD_ORDER):
        x = left + column * cell_width + cell_width / 2
        label = METHOD_LABELS[method].replace(" ", "\n", 1)
        parts = label.split("\n")
        body.append(text(x, top - 54, parts[0], size=19, weight=600, anchor="middle"))
        if len(parts) > 1:
            body.append(text(x, top - 29, parts[1], size=19, weight=600, anchor="middle"))
        if method == "active_recompute":
            body.append(line(left + column * cell_width + 5, top - 16, left + (column + 1) * cell_width - 5, top - 16, stroke="#d55e00", stroke_width=5))

    for row_index, (benchmark, model) in enumerate(strata):
        y = top + row_index * cell_height
        benchmark_label = "Evolving" if benchmark == "evolving_intent_gsm8k" else "BFCL"
        body.append(text(left - 25, y + 49, f"{benchmark_label} · {MODEL_LABELS[model]}", size=22, anchor="end"))
        values = [float(by_cell[(benchmark, model, method)]["auprc"]) for method in METHOD_ORDER]
        maximum = max(values)
        for column, (method, value) in enumerate(zip(METHOD_ORDER, values)):
            x = left + column * cell_width
            fill = heat_color(value)
            body.append(rect(x + 3, y + 3, cell_width - 6, cell_height - 6, fill=fill, stroke="#ffffff", stroke_width=2))
            if abs(value - maximum) < 1e-12:
                body.append(rect(x + 5, y + 5, cell_width - 10, cell_height - 10, fill="none", stroke="#102a43", stroke_width=5, rx=5))
            body.append(text(x + cell_width / 2, y + 50, f"{value:.2f}", size=22, weight=700 if abs(value - maximum) < 1e-12 else 400, anchor="middle", fill="#ffffff" if value >= 0.48 else "#17324d"))
        if row_index == 3:
            body.append(line(55, y + cell_height, width - 55, y + cell_height, stroke="#7b8794", stroke_width=3, stroke_dasharray="7 7"))

    legend_y = height - 78
    body.append(text(55, legend_y + 8, "AUPRC", size=21, weight=600))
    for index in range(11):
        value = index / 10
        x = 150 + index * 38
        body.append(rect(x, legend_y - 16, 38, 28, fill=heat_color(value)))
    body.append(text(150, legend_y + 42, "0", size=18, anchor="middle", fill="#536679"))
    body.append(text(150 + 11 * 38, legend_y + 42, "1", size=18, anchor="middle", fill="#536679"))
    body.append(text(width - 55, legend_y + 8, "Active wins 3/7 slices; passive or baseline methods win 4/7", size=22, anchor="end", weight=600, fill="#334e68"))
    return svg(width, height, body, description="AUPRC heatmap for all primary model-by-task slices and monitoring methods")


def overhead_figure(materials: Mapping[str, object]) -> str:
    overhead = materials["observer_overhead"]
    assert isinstance(overhead, dict)
    rows = overhead["method_aggregates"]
    assert isinstance(rows, list)
    by_method = {row["method"]: row for row in rows}
    if set(by_method) != set(METHOD_ORDER):
        raise ValueError("observer-overhead method grid changed")

    width, height = 1500, 960
    left, right, top, bottom = 340, 250, 220, 115
    plot_width = width - left - right
    maximum = 1_800_000
    row_step = (height - top - bottom) / len(METHOD_ORDER)

    def xmap(value: float) -> float:
        return left + value / maximum * plot_width

    body: list[str] = [
        text(55, 62, "Observation itself consumes compute", size=38, weight=700),
        text(55, 104, "Total observer tokens across 392 model–task cases; annotations show recorded observer cost", size=25, fill="#536679"),
        text(55, 142, "Provider-backed methods each made 1,764 calls; deterministic rules, clock, and context-use made none", size=21, fill="#536679"),
    ]
    for tick in (0, 500_000, 1_000_000, 1_500_000):
        x = xmap(tick)
        body.append(line(x, top - 15, x, height - bottom + 5, stroke="#dbe3ea", stroke_width=2))
        body.append(text(x, height - 66, "0" if tick == 0 else f"{tick / 1_000_000:.1f}M", size=20, anchor="middle", fill="#536679"))
    body.append(text(left + plot_width / 2, height - 25, "total observer tokens", size=23, anchor="middle"))
    body.append(text(width - right + 55, top - 30, "recorded cost", size=20, anchor="middle", weight=600, fill="#536679"))

    for index, method in enumerate(METHOD_ORDER):
        row = by_method[method]
        y = top + row_step * (index + 0.5)
        if index % 2 == 0:
            body.append(rect(45, y - row_step / 2, width - 90, row_step, fill="#f7f9fb"))
        body.append(text(left - 28, y + 8, METHOD_LABELS[method], size=23, anchor="end", weight=600 if method == "active_recompute" else 400))
        tokens = int(row["total_tokens"])
        observation_class = str(row["observation_class"])
        color = "#d55e00" if observation_class == "active" else "#0072b2" if tokens else "#b8c2cc"
        if tokens:
            body.append(rect(left, y - 17, xmap(tokens) - left, 34, fill=color, rx=4))
            body.append(text(xmap(tokens) - 12, y + 8, f"{tokens / 1_000_000:.2f}M", size=20, anchor="end", weight=600, fill="#ffffff"))
        else:
            body.append(line(left, y, left + 38, y, stroke=color, stroke_width=7, stroke_linecap="round"))
            body.append(text(left + 52, y + 8, "0", size=20, fill="#536679"))
        cost = float(row["cost_usd"])
        body.append(text(width - right + 55, y + 8, f"${cost:.2f}", size=22, anchor="middle", weight=600 if cost else 400, fill="#17324d"))

    return svg(width, height, body, description="Observer token and recorded-cost overhead by monitoring method")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(renderer: Path | None = None) -> tuple[Path, ...]:
    materials = read_materials()
    figures = (
        ("01-recovery-operator-effects", recovery_figure()),
        ("03-signal-quality-auprc", signal_figure(materials)),
        ("04-observation-overhead", overhead_figure(materials)),
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
            str(OPERATOR_EFFECTS.relative_to(EXPERIMENT_ROOT)): sha256(OPERATOR_EFFECTS),
            str(PAPER_MATERIALS.relative_to(EXPERIMENT_ROOT)): sha256(PAPER_MATERIALS),
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
