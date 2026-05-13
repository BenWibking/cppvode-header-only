#!/usr/bin/env python3
"""Plot runtime versus SGPR/VGPR spill counts from the register-usage CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


DEFAULT_INPUT = Path("kernel_register_usage_by_version.txt")
DEFAULT_OUTPUT = Path("runtime_vs_spills.png")


def parse_number(row: dict[str, str], field: str) -> float | None:
    value = row.get(field, "").strip()
    if not value or value == "FAILED":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def plot_runtime_vs_spills(input_path: Path, output_path: Path, *, show: bool) -> int:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for plotting; install it and rerun this script", file=sys.stderr)
        return 1

    rows = read_rows(input_path)
    points: list[dict[str, float | str]] = []
    skipped: list[str] = []

    for row in rows:
        rocm_dir = row.get("rocm_dir", "")
        runtime_s = parse_number(row, "runtime_s")
        sgpr_spills = parse_number(row, "sgpr_spill_count")
        vgpr_spills = parse_number(row, "vgpr_spill_count")
        if runtime_s is None or sgpr_spills is None or vgpr_spills is None:
            skipped.append(rocm_dir or "<unknown>")
            continue
        points.append(
            {
                "rocm_dir": rocm_dir,
                "runtime_s": runtime_s,
                "sgpr_spill_count": sgpr_spills,
                "vgpr_spill_count": vgpr_spills,
            }
        )

    if not points:
        print(f"no plottable rows found in {input_path}", file=sys.stderr)
        return 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, constrained_layout=True)
    series = [
        ("sgpr_spill_count", "SGPR spills"),
        ("vgpr_spill_count", "VGPR spills"),
    ]

    for ax, (field, xlabel) in zip(axes, series):
        x_values = [float(point[field]) for point in points]
        y_values = [float(point["runtime_s"]) for point in points]
        ax.scatter(x_values, y_values, color="#1f77b4")
        for point, x_value, y_value in zip(points, x_values, y_values):
            ax.annotate(
                str(point["rocm_dir"]).removeprefix("rocm_"),
                (x_value, y_value),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Runtime (s)")
    fig.suptitle("Runtime vs register spills")
    fig.savefig(output_path, dpi=180)
    if show:
        plt.show()

    print(f"wrote {output_path}")
    if skipped:
        print("skipped rows without numeric runtime/spill data: " + ", ".join(skipped), file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot runtime against SGPR and VGPR spill counts from kernel_register_usage_by_version.txt."
    )
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT, help=f"CSV input path (default: {DEFAULT_INPUT})")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT, help=f"plot output path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--show", action="store_true", help="also display the plot interactively")
    args = parser.parse_args()

    return plot_runtime_vs_spills(args.input, args.output, show=args.show)


if __name__ == "__main__":
    raise SystemExit(main())
