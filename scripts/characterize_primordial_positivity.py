#!/usr/bin/env python3
"""Run CPU primordial collapse positivity diagnostics for Rosenbrock methods."""

from __future__ import annotations

import argparse
import csv
import subprocess
from collections import defaultdict
from pathlib import Path


METHODS = ["ros2s", "sandu-a", "sandu-b", "sandu-d"]


def extract_csv_block(text: str) -> list[dict[str, str]]:
    lines = text.splitlines()
    try:
        begin = lines.index("positivity_csv_begin")
        end = lines.index("positivity_csv_end")
    except ValueError as exc:
        raise RuntimeError("missing positivity CSV block") from exc
    reader = csv.DictReader(lines[begin + 1:end])
    return list(reader)


def run_method(exe: Path, method: str, args: argparse.Namespace) -> tuple[list[dict[str, str]], int]:
    cmd = [
        str(exe),
        "--integrator",
        method,
        "--grid",
        str(args.grid),
        "--rtol",
        f"{args.rtol:.17e}",
        "--atol",
        f"{args.atol:.17e}",
        "--energy-atol",
        f"{args.energy_atol:.17e}",
        "--positivity-report",
    ]
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    rows = extract_csv_block(proc.stdout + proc.stderr)
    for row in rows:
        row["cli_method"] = method
    return rows, proc.returncode


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["cli_method", "method", "step", "component", "nonpositive_count", "cumulative_count"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    totals: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        key = (row["method"], row["component"])
        step = int(row["step"])
        count = int(row["nonpositive_count"])
        cumulative = int(row["cumulative_count"])
        item = totals.setdefault(
            key,
            {
                "method": row["method"],
                "component": row["component"],
                "first_step": step,
                "last_step": step,
                "steps_with_nonpositive": 0,
                "final_cumulative": 0,
            },
        )
        item["first_step"] = min(int(item["first_step"]), step)
        item["last_step"] = max(int(item["last_step"]), step)
        if count > 0:
            item["steps_with_nonpositive"] = int(item["steps_with_nonpositive"]) + 1
        item["final_cumulative"] = max(int(item["final_cumulative"]), cumulative)
    return sorted(totals.values(), key=lambda x: (str(x["component"]), str(x["method"])))


def write_markdown(path: Path, summary: list[dict[str, object]], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_method: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in summary:
        by_method[str(row["method"])].append(row)

    with path.open("w") as handle:
        handle.write("# Primordial Rosenbrock Positivity Diagnostic\n\n")
        handle.write("Counts are sampled after each successful collapse burn, before cleanup floors and normalization.\n\n")
        handle.write("## Components Ever Nonpositive\n\n")
        handle.write("| method | component | first step | last reported step | steps with count > 0 | final cumulative count |\n")
        handle.write("|---|---:|---:|---:|---:|---:|\n")
        for row in summary:
            handle.write(
                f"| {row['method']} | {row['component']} | {row['first_step']} | "
                f"{row['last_step']} | {row['steps_with_nonpositive']} | {row['final_cumulative']} |\n"
            )

        handle.write("\n## Per-Step Counts\n\n")
        for method in sorted({row["method"] for row in rows}):
            handle.write(f"### {method}\n\n")
            handle.write("| step | component | nonpositive count | cumulative count |\n")
            handle.write("|---:|---:|---:|---:|\n")
            for row in rows:
                if row["method"] != method:
                    continue
                handle.write(
                    f"| {row['step']} | {row['component']} | "
                    f"{row['nonpositive_count']} | {row['cumulative_count']} |\n"
                )
            handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, default=Path("build-codex-check/examples/primordial_chem"))
    parser.add_argument("--grid", type=int, default=1)
    parser.add_argument("--rtol", type=float, default=1.0e-4)
    parser.add_argument("--atol", type=float, default=1.0e-4)
    parser.add_argument("--energy-atol", type=float, default=1.0e-6)
    parser.add_argument("--output-csv", type=Path, default=Path("logs/primordial_rosenbrock_positivity.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("logs/primordial_rosenbrock_positivity.md"))
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    statuses: dict[str, int] = {}
    for method in METHODS:
        method_rows, status = run_method(args.exe, method, args)
        rows.extend(method_rows)
        statuses[method] = status
        print(f"{method}: rows={len(method_rows)} exit={status}")

    write_csv(args.output_csv, rows)
    summary = summarize(rows)
    write_markdown(args.output_md, summary, rows)
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.output_md}")

    if summary:
        print("\nComponents ever nonpositive:")
        for row in summary:
            print(
                f"{row['method']:18s} {row['component']:>5s} "
                f"first={row['first_step']:>4} final_count={row['final_cumulative']}"
            )
    else:
        print("\nNo nonpositive species or energy were observed.")

    return 0 if all(status in (0, 1) for status in statuses.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
