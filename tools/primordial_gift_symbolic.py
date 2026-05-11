#!/usr/bin/env python3
"""Find a GIFT-style diagonal-pivot symbolic LU ordering for primordial chemistry."""

from __future__ import annotations

import re
from pathlib import Path


N = 15
NAMES = [
    "E",
    "Hp",
    "H",
    "Hm",
    "Dp",
    "D",
    "H2p",
    "Dm",
    "H2",
    "HDp",
    "HD",
    "HEpp",
    "HEp",
    "HE",
    "Eint",
]


def read_pattern(path: Path) -> int:
    text = path.read_text()
    pattern = 0
    for match in re.finditer(r"jac\((\d+),(\d+)\)\s*=\s*([^;]+);", text):
        row = int(match.group(1)) - 1
        col = int(match.group(2)) - 1
        expr = match.group(3).strip()
        if expr not in {"0", "0.0", "0."}:
            pattern |= 1 << (row * N + col)

    # ROS2S factors fac*I - J, so the diagonal is always structurally present.
    for i in range(N):
        pattern |= 1 << (i * N + i)
    return pattern


def immediate_fill(pattern: int, remaining: int, pivot: int) -> tuple[int, int, int]:
    rows = [
        i
        for i in range(N)
        if (remaining >> i) & 1
        and i != pivot
        and (pattern >> (i * N + pivot)) & 1
    ]
    cols = [
        j
        for j in range(N)
        if (remaining >> j) & 1
        and j != pivot
        and (pattern >> (pivot * N + j)) & 1
    ]
    added = 0
    fill = 0
    for i in rows:
        for j in cols:
            bit = 1 << (i * N + j)
            if not pattern & bit:
                added |= bit
                fill += 1
    return added, fill, len(rows) * len(cols)


def factor(pattern: int, order: list[int]) -> tuple[int, int, int, list[tuple[int, int, int, int]]]:
    remaining = (1 << N) - 1
    fill = 0
    ops = 0
    steps = []
    for pivot in order:
        added, step_fill, step_ops = immediate_fill(pattern, remaining, pivot)
        rows = sum(
            1
            for i in range(N)
            if (remaining >> i) & 1
            and i != pivot
            and (pattern >> (i * N + pivot)) & 1
        )
        cols = sum(
            1
            for j in range(N)
            if (remaining >> j) & 1
            and j != pivot
            and (pattern >> (pivot * N + j)) & 1
        )
        pattern |= added
        remaining &= ~(1 << pivot)
        fill += step_fill
        ops += step_ops
        steps.append((pivot, rows, cols, step_fill))
    return pattern.bit_count(), fill, ops, steps


def find_order_with_budget(pattern: int, budget: int) -> tuple[bool, list[int], int]:
    full = (1 << N) - 1
    memo: set[tuple[int, int, int]] = set()
    nodes = 0

    def search(current: int, remaining: int, fill_budget: int, order: list[int]) -> list[int] | None:
        nonlocal nodes
        nodes += 1
        if remaining == 0:
            return order
        key = (current, remaining, fill_budget)
        if key in memo:
            return None

        candidates = []
        scan = remaining
        while scan:
            bit = scan & -scan
            pivot = bit.bit_length() - 1
            scan -= bit
            added, fill, ops = immediate_fill(current, remaining, pivot)
            if fill <= fill_budget:
                candidates.append((fill, ops, pivot, added))

        candidates.sort()
        for fill, _, pivot, added in candidates:
            found = search(
                current | added,
                remaining & ~(1 << pivot),
                fill_budget - fill,
                order + [pivot],
            )
            if found is not None:
                return found

        memo.add(key)
        return None

    order = search(pattern, full, budget, [])
    return order is not None, order or [], nodes


def main() -> None:
    source = Path("include/integrators/primordial_chem.hpp")
    pattern = read_pattern(source)
    natural = list(range(N))
    print(f"initial nonzeros: {pattern.bit_count()} / {N * N}")
    print(f"natural order: final/fill/update_ops = {factor(pattern, natural)[:3]}")

    for budget in range(N * N):
        ok, order, nodes = find_order_with_budget(pattern, budget)
        print(f"fill budget {budget}: {'yes' if ok else 'no'} ({nodes} search nodes)")
        if ok:
            total, fill, ops, steps = factor(pattern, order)
            print(f"best order: {[NAMES[i] for i in order]}")
            print(f"best final/fill/update_ops: {total}, {fill}, {ops}")
            print("pivot rows cols fill")
            for pivot, rows, cols, step_fill in steps:
                print(f"{NAMES[pivot]:>4} {rows:2d} {cols:2d} {step_fill:2d}")
            return


if __name__ == "__main__":
    main()
