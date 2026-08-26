"""Export the 11 final Experiment 12 paper figures as high-resolution PNGs."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import struct
import subprocess
from typing import Sequence

from experiments12.paths12 import DERIVED_ROOT, EXPERIMENT_ROOT, GRAPHS_ROOT, RUNS_ROOT


FIGURES: tuple[tuple[str, str, Path], ...] = (
    (
        "01-deployment-interactions.png",
        "Main: deployment interactions",
        DERIVED_ROOT / "deployment-interaction-confirmatory-v1.svg",
    ),
    (
        "02-active-probe-observer-effect.png",
        "Main: carried active probes and observer effect",
        DERIVED_ROOT / "active-probe-ladder-confirmatory-v1.svg",
    ),
    (
        "03-observation-overhead.png",
        "Main: observation cost",
        DERIVED_ROOT / "observer-overhead-confirmatory-v1.svg",
    ),
    (
        "04-signal-pr-evolving-deepseek.png",
        "Signal PR: Evolving Intent / DeepSeek V4 Flash",
        RUNS_ROOT
        / "e12-confirmatory-evolving-core-v2/results/signal-figures/"
        "signal-pr-evolving_intent_gsm8k-deepseek-v4-flash-0731.svg",
    ),
    (
        "05-signal-pr-evolving-luna.png",
        "Signal PR: Evolving Intent / GPT-5.6 Luna",
        RUNS_ROOT
        / "e12-confirmatory-evolving-core-v2/results/signal-figures/"
        "signal-pr-evolving_intent_gsm8k-gpt-5.6-luna.svg",
    ),
    (
        "06-signal-pr-evolving-terra.png",
        "Signal PR: Evolving Intent / GPT-5.6 Terra",
        RUNS_ROOT
        / "e12-confirmatory-evolving-core-v2/results/signal-figures/"
        "signal-pr-evolving_intent_gsm8k-gpt-5.6-terra.svg",
    ),
    (
        "07-signal-pr-evolving-gpt-oss.png",
        "Signal PR: Evolving Intent / GPT-OSS-120B",
        RUNS_ROOT
        / "e12-confirmatory-evolving-core-v2/results/signal-figures/"
        "signal-pr-evolving_intent_gsm8k-gpt-oss-120b.svg",
    ),
    (
        "08-signal-pr-bfcl-luna.png",
        "Signal PR: BFCL / GPT-5.6 Luna",
        RUNS_ROOT
        / "e12-confirmatory-bfcl-core-v3/results/signal-figures/"
        "signal-pr-bfcl_multi_turn-gpt-5.6-luna.svg",
    ),
    (
        "09-signal-pr-bfcl-terra.png",
        "Signal PR: BFCL / GPT-5.6 Terra",
        RUNS_ROOT
        / "e12-confirmatory-bfcl-core-v3/results/signal-figures/"
        "signal-pr-bfcl_multi_turn-gpt-5.6-terra.svg",
    ),
    (
        "10-signal-pr-bfcl-gpt-oss.png",
        "Signal PR: BFCL / GPT-OSS-120B",
        RUNS_ROOT
        / "e12-confirmatory-bfcl-core-v3/results/signal-figures/"
        "signal-pr-bfcl_multi_turn-gpt-oss-120b.svg",
    ),
    (
        "11-online-deployment-success.png",
        "Appendix: absolute online deployment success",
        DERIVED_ROOT
        / "adaptive-analysis-staging-v1/analysis/figures/"
        "deployment-evolving_intent_gsm8k-gpt-5.6-luna.svg",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG file: {path}")
    return struct.unpack(">II", header[16:24])


def export(
    *,
    scale: float = 2.0,
    renderer: Path | None = None,
    fonts_dir: Path | None = None,
) -> tuple[Path, ...]:
    if scale <= 0:
        raise ValueError("scale must be positive")

    cairosvg = None
    renderer_label: str
    if renderer is None:
        try:
            import cairosvg as imported_cairosvg
        except (ImportError, OSError) as exc:  # pragma: no cover - local tooling
            raise RuntimeError(
                "install cairosvg or pass --renderer /path/to/resvg"
            ) from exc
        cairosvg = imported_cairosvg
        renderer_label = f"CairoSVG {cairosvg.__version__}"
    else:
        renderer = renderer.expanduser().resolve(strict=True)
        if fonts_dir is None:
            candidate = Path("/usr/share/fonts/truetype/liberation")
            fonts_dir = candidate if candidate.is_dir() else None
        elif not fonts_dir.expanduser().is_dir():
            raise FileNotFoundError(f"font directory is unavailable: {fonts_dir}")
        version = subprocess.run(
            [str(renderer), "--version"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        renderer_label = f"resvg {version}"
        if fonts_dir is not None:
            fonts_dir = fonts_dir.expanduser().resolve()
            renderer_label += " with Liberation Sans"

    missing = [source for _name, _label, source in FIGURES if not source.is_file()]
    if missing:
        raise FileNotFoundError("missing source SVGs: " + ", ".join(map(str, missing)))

    GRAPHS_ROOT.mkdir(parents=True, exist_ok=True)
    exported: list[Path] = []
    rows: list[tuple[str, str, str, str, int, int]] = []
    for filename, label, source in FIGURES:
        destination = GRAPHS_ROOT / filename
        if renderer is not None:
            command = [
                str(renderer),
                "--quiet",
                "--background",
                "white",
                "--zoom",
                f"{scale:g}",
                "--font-family",
                "Liberation Sans",
                "--sans-serif-family",
                "Liberation Sans",
            ]
            if fonts_dir is not None:
                command.extend(["--use-fonts-dir", str(fonts_dir)])
            command.extend([str(source), str(destination)])
            subprocess.run(command, check=True)
        else:
            assert cairosvg is not None
            cairosvg.svg2png(
                bytestring=source.read_bytes(),
                write_to=str(destination),
                scale=scale,
                background_color="white",
            )
        width, height = _png_dimensions(destination)
        rows.append(
            (
                filename,
                label,
                source.relative_to(EXPERIMENT_ROOT).as_posix(),
                _sha256(source),
                width,
                height,
            )
        )
        exported.append(destination)

    lines = [
        "# Figure index",
        "",
        f"PNG exports generated from frozen SVGs at {scale:g}× source scale with {renderer_label}.",
        "",
        "| PNG | role | source SVG | source SHA256 | pixels |",
        "|---|---|---|---|---:|",
    ]
    for filename, label, source, source_sha256, width, height in rows:
        lines.append(
            f"| `{filename}` | {label} | `{source}` | `{source_sha256}` | {width}×{height} |"
        )
    lines.extend(["", "## PNG integrity", ""])
    for path in exported:
        lines.append(f"- `{path.name}`: `{_sha256(path)}`")
    lines.append("")
    (GRAPHS_ROOT / "FIGURE_INDEX.md").write_text("\n".join(lines), encoding="utf-8")
    return tuple(exported)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument(
        "--renderer",
        type=Path,
        help="optional path to the self-contained resvg executable",
    )
    parser.add_argument(
        "--fonts-dir",
        type=Path,
        help="font directory for resvg (defaults to system Liberation Sans)",
    )
    args = parser.parse_args(argv)
    for path in export(
        scale=args.scale,
        renderer=args.renderer,
        fonts_dir=args.fonts_dir,
    ):
        print(path.relative_to(EXPERIMENT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
