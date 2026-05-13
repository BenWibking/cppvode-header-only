#!/usr/bin/env python3
"""Dump AMDGPU kernel register usage and spill counts from HIP binaries."""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


FIELDS = [
    "rocm_dir",
    "binary",
    "target",
    "kernel",
    "sgpr_count",
    "vgpr_count",
    "sgpr_spill_count",
    "vgpr_spill_count",
    "agpr_count",
    "private_segment_fixed_size",
    "wavefront_size",
]


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def find_rocm_tool(binary: Path, name: str) -> str:
    version_match = re.search(r"rocm_(\d+\.\d+\.\d+)", str(binary))
    candidates: list[Path] = []
    if version_match:
        candidates.append(Path(f"/opt/rocm-{version_match.group(1)}") / "llvm" / "bin" / name)
    candidates.append(Path("/opt/rocm") / "llvm" / "bin" / name)

    for root in sorted(Path("/opt").glob("rocm-*"), reverse=True):
        candidates.append(root / "llvm" / "bin" / name)

    path_tool = shutil.which(name)
    if path_tool:
        candidates.append(Path(path_tool))

    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)

    raise FileNotFoundError(f"could not find {name}; pass --rocm-bin /path/to/rocm/llvm/bin")


def is_probable_binary(path: Path) -> bool:
    if not path.is_file() or not os.access(path, os.X_OK):
        return False
    try:
        with path.open("rb") as f:
            return f.read(4) == b"\x7fELF"
    except OSError:
        return False


def discover_binaries(paths: list[Path]) -> list[Path]:
    binaries: list[Path] = []
    for path in paths:
        if path.is_dir():
            binaries.extend(p for p in sorted(path.iterdir()) if is_probable_binary(p))
        elif is_probable_binary(path):
            binaries.append(path)
    return binaries


def clean_value(value: str) -> str:
    value = value.strip()
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    return value


def parse_kernel_metadata(text: str) -> list[dict[str, str]]:
    kernels: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    in_kernels = False
    in_args = False

    for line in text.splitlines():
        if line.strip() == "amdhsa.kernels:":
            in_kernels = True
            continue
        if not in_kernels:
            continue
        if line.startswith("amdhsa.") and not line.startswith("amdhsa.kernels"):
            break

        kernel_start = re.match(r"^  - \.([A-Za-z0-9_]+):\s*(.*)$", line)
        if kernel_start:
            if current:
                kernels.append(current)
            current = {kernel_start.group(1): clean_value(kernel_start.group(2))}
            in_args = False
            continue

        if current is None:
            continue

        top_key = re.match(r"^    \.([A-Za-z0-9_]+):\s*(.*)$", line)
        if top_key:
            key, value = top_key.groups()
            in_args = key == "args"
            if not in_args:
                current[key] = clean_value(value)
            continue

        if in_args:
            continue

    if current:
        kernels.append(current)
    return kernels


def demangle_names(rows: list[dict[str, str]]) -> None:
    names = [row["kernel"] for row in rows if row.get("kernel")]
    cxxfilt = shutil.which("c++filt")
    if not names or not cxxfilt:
        return
    proc = subprocess.run(
        [cxxfilt],
        input="\n".join(names) + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if proc.returncode != 0:
        return
    demangled = proc.stdout.splitlines()
    for row, name in zip((r for r in rows if r.get("kernel")), demangled):
        row["kernel"] = name


def dump_binary(binary: Path, rocm_bin: Path | None) -> list[dict[str, str]]:
    if rocm_bin:
        objcopy = str(rocm_bin / "llvm-objcopy")
        bundler = str(rocm_bin / "clang-offload-bundler")
        readobj = str(rocm_bin / "llvm-readobj")
    else:
        objcopy = find_rocm_tool(binary, "llvm-objcopy")
        bundler = find_rocm_tool(binary, "clang-offload-bundler")
        readobj = find_rocm_tool(binary, "llvm-readobj")

    rows: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="kernel-register-usage.") as tmp:
        tmpdir = Path(tmp)
        fatbin = tmpdir / "hip_fatbin"
        dumped = run([objcopy, "--dump-section", f".hip_fatbin={fatbin}", str(binary)], check=False)
        if dumped.returncode != 0:
            return rows

        listed = run([bundler, "--list", "--type=o", f"--input={fatbin}"])
        targets = [line.strip() for line in listed.stdout.splitlines() if line.strip()]
        for index, target in enumerate(targets):
            if "amdgcn" not in target:
                continue
            code_object = tmpdir / f"code-object-{index}.o"
            run(
                [
                    bundler,
                    "--unbundle",
                    "--type=o",
                    f"--targets={target}",
                    f"--input={fatbin}",
                    f"--output={code_object}",
                ]
            )
            notes = run([readobj, "--notes", str(code_object)]).stdout
            for kernel in parse_kernel_metadata(notes):
                row = {field: "" for field in FIELDS}
                row["rocm_dir"] = binary.parent.name
                row["binary"] = str(binary)
                row["target"] = target
                row["kernel"] = kernel.get("name", "")
                for field in FIELDS:
                    if field in kernel:
                        row[field] = kernel[field]
                rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dump SGPR/VGPR usage and spill counts for kernels in HIP binaries."
    )
    parser.add_argument("paths", nargs="*", type=Path, default=sorted(Path(".").glob("rocm_*")))
    parser.add_argument("--rocm-bin", type=Path, help="Directory containing llvm-objcopy, clang-offload-bundler, and llvm-readobj")
    parser.add_argument("--format", choices=("tsv", "csv"), default="tsv")
    parser.add_argument("--no-demangle", action="store_true", help="Keep mangled kernel names")
    args = parser.parse_args()

    binaries = discover_binaries(args.paths)
    if not binaries:
        print("no ELF executables found", file=sys.stderr)
        return 1

    rows: list[dict[str, str]] = []
    for binary in binaries:
        try:
            rows.extend(dump_binary(binary, args.rocm_bin))
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            print(f"warning: {binary}: {exc}", file=sys.stderr)

    if not args.no_demangle:
        demangle_names(rows)

    delimiter = "\t" if args.format == "tsv" else ","
    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS, delimiter=delimiter, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
