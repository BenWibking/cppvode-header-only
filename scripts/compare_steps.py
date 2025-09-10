#!/usr/bin/env python3

import argparse
import math
import re
from typing import List, Tuple, Optional


Entry = Tuple[float, float, int]  # (t, hu, nq)


def parse_dvode(path: str) -> List[Entry]:
    # Lines like: t =   0.0001234567 hu =  1.2346E-04 nq = 1
    rx = re.compile(r"t\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?)\s+hu\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?)\s+nq\s*=\s*(\d+)")
    out: List[Entry] = []
    with open(path, 'r') as f:
        for line in f:
            m = rx.search(line)
            if not m:
                continue
            t = float(m.group(1).replace('D', 'E').replace('d', 'e'))
            hu = float(m.group(2).replace('D', 'E').replace('d', 'e'))
            nq = int(m.group(3))
            out.append((t, hu, nq))
    return out


def parse_cpp(path: str) -> List[Entry]:
    # Lines like: [VODE] ACCEPT t=0.000123456 hu=1.23456e-04 nq=1
    rx = re.compile(r"ACCEPT\s+t=([^\s]+)\s+hu=([^\s]+)\s+nq=(\d+)")
    out: List[Entry] = []
    with open(path, 'r') as f:
        for line in f:
            m = rx.search(line)
            if not m:
                continue
            t = float(m.group(1))
            hu = float(m.group(2))
            nq = int(m.group(3))
            out.append((t, hu, nq))
    return out


def align(a: List[Entry], b: List[Entry], tol: float) -> Tuple[List[Tuple[Entry, Entry]], List[Entry], List[Entry]]:
    # Two-pointer alignment by time
    i = j = 0
    matched: List[Tuple[Entry, Entry]] = []
    extra_a: List[Entry] = []
    extra_b: List[Entry] = []

    def close(x: float, y: float) -> bool:
        if x == y:
            return True
        absv = abs(x - y)
        rel = absv / max(1.0, abs(x), abs(y))
        return absv <= tol or rel <= tol

    while i < len(a) and j < len(b):
        ta = a[i][0]
        tb = b[j][0]
        if close(ta, tb):
            matched.append((a[i], b[j]))
            i += 1
            j += 1
        elif ta < tb:
            extra_a.append(a[i])
            i += 1
        else:
            extra_b.append(b[j])
            j += 1

    while i < len(a):
        extra_a.append(a[i]); i += 1
    while j < len(b):
        extra_b.append(b[j]); j += 1

    return matched, extra_a, extra_b


def main() -> None:
    p = argparse.ArgumentParser(description="Compare per-step sizes between DVODE (Fortran) and C++ VODE logs.")
    p.add_argument('dvode_log', help='Path to dvode_steps.txt')
    p.add_argument('vode_cpp_log', help='Path to vode_cpp_debug.txt (will parse ACCEPT lines)')
    p.add_argument('--tol', type=float, default=1e-10, help='Time alignment tolerance (absolute or relative). Default 1e-10')
    p.add_argument('--limit', type=int, default=50, help='Max rows to print in detailed comparison (default 50)')
    args = p.parse_args()

    a = parse_dvode(args.dvode_log)
    b = parse_cpp(args.vode_cpp_log)

    if not a:
        print(f"No DVODE steps parsed from {args.dvode_log}")
        return
    if not b:
        print(f"No C++ VODE steps parsed from {args.vode_cpp_log}")
        return

    matched, extra_a, extra_b = align(a, b, args.tol)

    print(f"Matched steps: {len(matched)}  |  DVODE-only: {len(extra_a)}  |  C++-only: {len(extra_b)}")

    # Summaries
    max_abs_dh = 0.0
    max_rel_dh = 0.0
    max_dt = 0.0
    nq_mismatch = 0
    sum_abs_dh = 0.0
    for (ta, ha, nqa), (tb, hb, nqb) in matched:
        dt = abs(tb - ta)
        dh = hb - ha
        denom = max(abs(ha), abs(hb), 1e-300)
        rd = abs(dh) / denom
        max_abs_dh = max(max_abs_dh, abs(dh))
        max_rel_dh = max(max_rel_dh, rd)
        max_dt = max(max_dt, dt)
        sum_abs_dh += abs(dh)
        if nqa != nqb:
            nq_mismatch += 1

    mean_abs_dh = sum_abs_dh / max(1, len(matched))
    print(f"Max |dt|: {max_dt:.3e}")
    print(f"Max |Δhu|: {max_abs_dh:.3e}  |  Mean |Δhu|: {mean_abs_dh:.3e}  |  Max rel |Δhu|: {max_rel_dh:.3e}")
    print(f"Order mismatches: {nq_mismatch}")

    # Detailed rows
    print("\nFirst {} matched steps: t_dvode  t_cpp  dt  hu_dvode  hu_cpp  d_hu  rel_d_hu  nq_d  nq_c".format(min(args.limit, len(matched))))
    count = 0
    for (ta, ha, nqa), (tb, hb, nqb) in matched:
        if count >= args.limit:
            break
        dt = tb - ta
        d_hu = hb - ha
        denom = max(abs(ha), abs(hb), 1e-300)
        rel = abs(d_hu) / denom
        print(f"{ta:.10e}  {tb:.10e}  {dt:.3e}  {ha:.6e}  {hb:.6e}  {d_hu:.3e}  {rel:.3e}  {nqa}  {nqb}")
        count += 1

    if extra_a:
        print(f"\nDVODE-only steps (first {min(5, len(extra_a))}):")
        for t, h, nq in extra_a[:5]:
            print(f"t={t:.10e} hu={h:.6e} nq={nq}")
    if extra_b:
        print(f"\nC++-only steps (first {min(5, len(extra_b))}):")
        for t, h, nq in extra_b[:5]:
            print(f"t={t:.10e} hu={h:.6e} nq={nq}")


if __name__ == '__main__':
    main()

