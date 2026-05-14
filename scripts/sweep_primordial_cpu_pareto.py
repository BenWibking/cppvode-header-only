#!/usr/bin/env python3
"""Sweep primordial chemistry tolerances and plot timing/accuracy curves."""

from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
from pathlib import Path


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def logspace(start: float, stop: float, count: int) -> list[float]:
    if count == 1:
        return [start]
    log_start = math.log10(start)
    log_stop = math.log10(stop)
    return [10.0 ** (log_start + i * (log_stop - log_start) / (count - 1)) for i in range(count)]


def parse_metric(pattern: str, text: str) -> float:
    match = re.search(pattern, text)
    if match is None:
        raise RuntimeError(f"missing metric matching {pattern!r}")
    return float(match.group(1))


DEFAULT_ROSENBROCK_INTEGRATORS = ["ros2s", "sandu-a", "sandu-b", "sandu-d"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def display_name(integrator: str) -> str:
    return {
        "ros2s": "ROS2S",
        "sandu-a": "Sandu A",
        "sandu-b": "Sandu B",
        "sandu-d": "Sandu D",
    }.get(integrator, integrator.upper())


def run_case(exe: Path, integrator: str, rtol: float, atol: float | None,
             energy_atol: float | None, grid: int, extra_args: list[str]) -> dict[str, object]:
    cmd = [str(exe), "--integrator", integrator, "--rtol", f"{rtol:.17e}"]
    if grid != 1:
        cmd.extend(["--grid", str(grid)])
    if atol is not None:
        cmd.extend(["--atol", f"{atol:.17e}"])
    if energy_atol is not None:
        cmd.extend(["--energy-atol", f"{energy_atol:.17e}"])
    cmd.extend(extra_args)
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    text = proc.stdout + proc.stderr
    try:
        time_sec = parse_metric(r"collapse loop walltime: (" + FLOAT + r") s", text)
        species_err = parse_metric(
            r"max non-deuterium species relative error vs Microphysics reference: (" + FLOAT + r")",
            text,
        )
        thermo_err = parse_metric(
            r"max T/Eint relative error vs Microphysics reference: (" + FLOAT + r")",
            text,
        )
    except RuntimeError:
        return {
            "integrator": integrator,
            "rtol": rtol,
            "atol": math.nan if atol is None else atol,
            "energy_rtol": math.nan,
            "energy_atol": math.nan if energy_atol is None else energy_atol,
            "status": "INTERNAL_FAIL",
            "reference": "FAIL",
            "time_sec": math.nan,
            "species_rel_error": math.nan,
            "thermo_rel_error": math.nan,
            "max_rel_error": math.nan,
        }

    validity = "PASS" if "state validity: PASS" in text else "FAIL"
    reference = "PASS" if "reference comparison: PASS" in text else "FAIL"
    status = "PASS" if validity == "PASS" else "STATE_FAIL"
    return {
        "integrator": integrator,
        "rtol": rtol,
        "atol": math.nan if atol is None else atol,
        "energy_rtol": math.nan,
        "energy_atol": math.nan if energy_atol is None else energy_atol,
        "status": status,
        "reference": reference,
        "time_sec": time_sec,
        "species_rel_error": species_err,
        "thermo_rel_error": thermo_err,
        "max_rel_error": max(species_err, thermo_err),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "integrator",
                "rtol",
                "atol",
                "energy_rtol",
                "energy_atol",
                "status",
                "reference",
                "time_sec",
                "species_rel_error",
                "thermo_rel_error",
                "max_rel_error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_plot(path: Path, rows: list[dict[str, object]], x_key: str, x_label: str,
               timing_label: str, title: str, show_failures: bool = False,
               annotate_points: bool = True) -> None:
    selected = [row for row in rows if row["status"] == "PASS"]
    if not selected:
        raise RuntimeError("no valid points to plot")

    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    integrators = list(dict.fromkeys(str(row["integrator"]) for row in rows))
    for index, integrator in enumerate(integrators):
        marker = MARKERS[index % len(MARKERS)]
        rows_i = [
            row for row in selected
            if row["integrator"] == integrator
        ]
        if not rows_i:
            continue
        rows_i.sort(key=lambda row: float(row[x_key]), reverse=True)
        ax.plot(
            [float(row[x_key]) for row in rows_i],
            [float(row["time_sec"]) for row in rows_i],
            marker=marker,
            linewidth=1.8,
            label=display_name(integrator),
        )
        if annotate_points:
            for row in rows_i:
                ax.annotate(
                    f"{float(row['rtol']):.0e}",
                    (float(row[x_key]), float(row["time_sec"])),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                )

    ymax = max(float(row["time_sec"]) for row in selected)
    if show_failures:
        fail_y = ymax * 1.6
        for index, integrator in enumerate(integrators):
            y_scale = 1.0 + 0.12 * index
            failed = [
                row for row in rows
                if row["integrator"] == integrator and row["status"] != "PASS"
            ]
            if not failed:
                continue
            ax.scatter(
                [float(row[x_key]) for row in failed],
                [fail_y * y_scale for row in failed],
                marker="x",
                s=70,
                color="red",
                linewidths=2.0,
                label=f"{display_name(integrator)} failure",
            )
        ymax = fail_y * 1.35

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(
        min(float(row["time_sec"]) for row in selected) * 0.75,
        ymax * 1.25,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(timing_label)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=180)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, default=Path("build-cpu/examples/primordial_chem"))
    parser.add_argument("--output-csv", type=Path, default=Path("logs/primordial_cpu_pareto.csv"))
    parser.add_argument("--output-plot", type=Path, default=Path("logs/primordial_cpu_pareto_error.png"))
    parser.add_argument("--output-rtol-plot", type=Path, default=Path("logs/primordial_cpu_pareto_rtol.png"))
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--count", type=int, default=13)
    parser.add_argument("--rtol-min", type=float, default=1.0e-6)
    parser.add_argument("--rtol-max", type=float, default=2.0e-2)
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--energy-atol", type=float, default=None)
    parser.add_argument("--scale-atol", action="store_true",
                        help="Scale species atol with requested rtol, relative to --base-rtol/--base-atol.")
    parser.add_argument("--scale-energy-atol", action="store_true",
                        help="Scale energy atol with requested rtol, relative to --base-rtol/--base-energy-atol.")
    parser.add_argument("--base-rtol", type=float, default=1.0e-4)
    parser.add_argument("--base-atol", type=float, default=1.0e-4)
    parser.add_argument("--base-energy-atol", type=float, default=1.0e-6)
    parser.add_argument("--grid", type=int, default=1)
    parser.add_argument("--timing-label", default="single-cell CPU time [s]")
    parser.add_argument("--title", default="Primordial chemistry CPU Pareto curve")
    parser.add_argument("--integrator", action="append", choices=["vode", *DEFAULT_ROSENBROCK_INTEGRATORS],
                        help="Integrator to sweep. May be repeated. Defaults to all Rosenbrock methods.")
    parser.add_argument("--extra-arg", action="append", default=[])
    args = parser.parse_args()

    if args.input_csv is None:
        rtols = logspace(args.rtol_max, args.rtol_min, args.count)
        rows: list[dict[str, object]] = []
        integrators = args.integrator if args.integrator is not None else DEFAULT_ROSENBROCK_INTEGRATORS
        for rtol in rtols:
            scale = rtol / args.base_rtol
            atol = args.base_atol * scale if args.scale_atol else args.atol
            energy_atol = args.base_energy_atol * scale if args.scale_energy_atol else args.energy_atol
            for integrator in integrators:
                row = run_case(args.exe, integrator, rtol, atol, energy_atol,
                               args.grid, args.extra_arg)
                row["energy_rtol"] = rtol * 1.0e-2
                rows.append(row)
                print(
                    f"{integrator:5s} rtol={rtol:.4e} status={row['status']} "
                    f"time={row['time_sec']} max_err={row['max_rel_error']}"
                )
        write_csv(args.output_csv, rows)
    else:
        rows = read_csv(args.input_csv)
    if not args.skip_plots:
        write_plot(args.output_plot, rows, "max_rel_error", "max relative error vs reference",
                   args.timing_label, args.title)
        write_plot(args.output_rtol_plot, rows, "rtol", "requested species relative tolerance",
                   args.timing_label, args.title, show_failures=True, annotate_points=False)
    print(f"wrote {args.output_csv}")
    if not args.skip_plots:
        print(f"wrote {args.output_plot}")
        print(f"wrote {args.output_rtol_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
