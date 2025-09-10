#!/usr/bin/env python3
import argparse
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class StepInfo:
    # Values captured for an accepted step
    pre: Dict[str, float] = field(default_factory=dict)
    post: Dict[str, float] = field(default_factory=dict)
    decide: Dict[str, float] = field(default_factory=dict)
    apply: Dict[str, float] = field(default_factory=dict)
    accept: Dict[str, float] = field(default_factory=dict)
    rejections: int = 0


def parse_dvode(path: str, limit: int = 50) -> List[StepInfo]:
    pre1 = re.compile(r"\[DVODE\] PRE tn=\s*([\dEeDd+\-.]+)\s*H=\s*([\dEeDd+\-.]+)\s*NQ=\s*(\d+)\s*L=\s*(\d+)")
    pre2 = re.compile(r"RL1=\s*([\dEeDd+\-.]+)\s*RC=\s*([\dEeDd+\-.]+)")
    pre3 = re.compile(r"NQWAIT=\s*(\d+)\s*ETA=\s*([\dEeDd+\-.]+)\s*TQ2=\s*([\dEeDd+\-.]+)")
    post1 = re.compile(r"\[DVODE\] POST ACNRM=\s*([\dEeDd+\-.]+)\s*DSM=\s*([\dEeDd+\-.]+)")
    post2 = re.compile(r"tq2=\s*([\dEeDd+\-.]+)\s*JCUR=\s*(\d+)\s*ICF=\s*(\d+)")
    post3 = re.compile(r"CRATE=\s*([\dEeDd+\-.]+)\s*RC=\s*([\dEeDd+\-.]+)")
    decide1 = re.compile(r"\[DVODE\] ORDER_DECIDE DSM=\s*([\dEeDd+\-.]+)\s*ETAQ_eff=\s*([\dEeDd+\-.]+)")
    decide2 = re.compile(r"ETAQM1=\s*([\dEeDd+\-.]+)\s*ETAQP1=\s*([\dEeDd+\-.]+)")
    apply = re.compile(r"\[DVODE\] ORDER_APPLY ETA=\s*([\dEeDd+\-.]+)\s*NEWQ=\s*(\d+)")
    accept = re.compile(r"\[DVODE\] ACCEPT t=\s*([\dEeDd+\-.]+)\s*hu=\s*([\dEeDd+\-.]+)\s*nq=\s*(\d+)")
    reject1 = re.compile(r"\[DVODE\] REJECT ")

    steps: List[StepInfo] = []
    cur: Optional[StepInfo] = None
    # track whether we're on the current in-progress step before acceptance
    for line in open(path, 'r'):
        if limit and len(steps) >= limit:
            break
        m = pre1.search(line)
        if m:
            cur = StepInfo()
            cur.pre.update({
                'tn': float(m.group(1).replace('D', 'E').replace('d', 'e')),
                'H': float(m.group(2).replace('D', 'E').replace('d', 'e')),
                'NQ': float(m.group(3)),
                'L': float(m.group(4)),
            })
            continue
        if cur is None:
            # not inside a step context; skip
            continue
        m = pre2.search(line)
        if m:
            cur.pre['RL1'] = float(m.group(1).replace('D', 'E').replace('d', 'e'))
            cur.pre['RC'] = float(m.group(2).replace('D', 'E').replace('d', 'e'))
            continue
        m = pre3.search(line)
        if m:
            cur.pre['NQWAIT'] = float(m.group(1))
            cur.pre['ETA'] = float(m.group(2).replace('D', 'E').replace('d', 'e'))
            cur.pre['TQ2'] = float(m.group(3).replace('D', 'E').replace('d', 'e'))
            continue
        m = post1.search(line)
        if m:
            cur.post['ACNRM'] = float(m.group(1).replace('D', 'E').replace('d', 'e'))
            cur.post['DSM'] = float(m.group(2).replace('D', 'E').replace('d', 'e'))
            continue
        m = post2.search(line)
        if m:
            cur.post['tq2'] = float(m.group(1).replace('D', 'E').replace('d', 'e'))
            cur.post['JCUR'] = float(m.group(2))
            cur.post['ICF'] = float(m.group(3))
            continue
        m = post3.search(line)
        if m:
            cur.post['CRATE'] = float(m.group(1).replace('D', 'E').replace('d', 'e'))
            cur.post['RC'] = float(m.group(2).replace('D', 'E').replace('d', 'e'))
            continue
        m = decide1.search(line)
        if m:
            cur.decide['DSM'] = float(m.group(1).replace('D', 'E').replace('d', 'e'))
            cur.decide['ETAQ_eff'] = float(m.group(2).replace('D', 'E').replace('d', 'e'))
            continue
        m = decide2.search(line)
        if m:
            cur.decide['ETAQM1'] = float(m.group(1).replace('D', 'E').replace('d', 'e'))
            cur.decide['ETAQP1'] = float(m.group(2).replace('D', 'E').replace('d', 'e'))
            continue
        m = apply.search(line)
        if m:
            cur.apply['ETA'] = float(m.group(1).replace('D', 'E').replace('d', 'e'))
            cur.apply['NEWQ'] = float(m.group(2))
            continue
        if reject1.search(line):
            cur.rejections += 1
            continue
        m = accept.search(line)
        if m:
            cur.accept['t'] = float(m.group(1).replace('D', 'E').replace('d', 'e'))
            cur.accept['hu'] = float(m.group(2).replace('D', 'E').replace('d', 'e'))
            cur.accept['nq'] = float(m.group(3))
            steps.append(cur)
            cur = None
            continue
    return steps[:limit]


def parse_cpp(path: str, limit: int = 50) -> List[StepInfo]:
    pre = re.compile(r"PRE tn=([^\s]+)\s+H=([^\s]+)\s+NQ=(\d+)\s+L=(\d+).*RL1=([^\s]+).*RC=([^\s]+).*NQWAIT=(\d+).*ETA=([^\s]+).*TQ2=([^\s]+).*TQ3=([^\s]+).*TQ4=([^\s]+).*TQ5=([^\s]+)")
    post = re.compile(r"POST ACNRM=([^\s]+)\s+DSM=([^\s]+)\s+tq2=([^\s]+)\s+JCUR=(\d+)\s+ICF=(\d+)\s+CRATE=([^\s]+)\s+RC=([^\s]+)")
    decide = re.compile(r"ORDER_DECIDE DSM=([^\s]+)\s+ETAQ_eff=([^\s]+)\s+ETAQM1=([^\s]+)\s+ETAQP1=([^\s]+)")
    apply = re.compile(r"ORDER_APPLY ETA=([^\s]+)\s+NEWQ=(\d+)")
    accept = re.compile(r"ACCEPT\s+t=([^\s]+)\s+hu=([^\s]+)\s+nq=(\d+)")
    reject = re.compile(r"REJECT ")

    steps: List[StepInfo] = []
    cur: Optional[StepInfo] = None
    for line in open(path, 'r'):
        if limit and len(steps) >= limit:
            break
        m = pre.search(line)
        if m:
            cur = StepInfo()
            cur.pre.update({
                'tn': float(m.group(1)),
                'H': float(m.group(2)),
                'NQ': float(m.group(3)),
                'L': float(m.group(4)),
                'RL1': float(m.group(5)),
                'RC': float(m.group(6)),
                'NQWAIT': float(m.group(7)),
                'ETA': float(m.group(8)),
                'TQ2': float(m.group(9)),
                # we collect TQ3/TQ4/TQ5 but won't compare to DVODE
            })
            continue
        if cur is None:
            continue
        m = post.search(line)
        if m:
            cur.post.update({
                'ACNRM': float(m.group(1)),
                'DSM': float(m.group(2)),
                'tq2': float(m.group(3)),
                'JCUR': float(m.group(4)),
                'ICF': float(m.group(5)),
                'CRATE': float(m.group(6)),
                'RC': float(m.group(7)),
            })
            continue
        m = decide.search(line)
        if m:
            cur.decide.update({
                'DSM': float(m.group(1)),
                'ETAQ_eff': float(m.group(2)),
                'ETAQM1': float(m.group(3)),
                'ETAQP1': float(m.group(4)),
            })
            continue
        m = apply.search(line)
        if m:
            cur.apply['ETA'] = float(m.group(1))
            cur.apply['NEWQ'] = float(m.group(2))
            continue
        if reject.search(line):
            cur.rejections += 1
            continue
        m = accept.search(line)
        if m:
            cur.accept.update({
                't': float(m.group(1)),
                'hu': float(m.group(2)),
                'nq': float(m.group(3)),
            })
            steps.append(cur)
            cur = None
            continue
    return steps[:limit]


def close(a: float, b: float, atol: float, rtol: float) -> bool:
    if a == b:
        return True
    if math.isfinite(a) and math.isfinite(b):
        if abs(a - b) <= atol:
            return True
        denom = max(abs(a), abs(b), 1.0)
        return abs(a - b) / denom <= rtol
    return False


def compare_sequences(a: List[StepInfo], b: List[StepInfo], atol: float, rtol: float) -> Optional[Tuple[int, str, float, float]]:
    # Compare per accepted step i, check fields in this order
    fields = [
        ('pre', 'tn'), ('pre', 'H'), ('pre', 'NQ'), ('pre', 'L'), ('pre', 'RL1'), ('pre', 'RC'), ('pre', 'NQWAIT'), ('pre', 'ETA'), ('pre', 'TQ2'),
        ('post', 'ACNRM'), ('post', 'DSM'), ('post', 'tq2'), ('post', 'JCUR'), ('post', 'ICF'), ('post', 'CRATE'), ('post', 'RC'),
        ('decide', 'DSM'), ('decide', 'ETAQ_eff'), ('decide', 'ETAQM1'), ('decide', 'ETAQP1'),
        ('apply', 'ETA'), ('apply', 'NEWQ'),
        ('accept', 't'), ('accept', 'hu'), ('accept', 'nq'),
    ]
    count = min(len(a), len(b))
    for i in range(count):
        ai = a[i]; bi = b[i]
        # quick check rejections count (only warn; not used to define divergence)
        for sect, key in fields:
            if key not in ai.__getattribute__(sect) or key not in bi.__getattribute__(sect):
                # skip if one side missing (e.g., DVODE lacks TQ3/TQ4/TQ5 which we don't compare anyway)
                continue
            va = ai.__getattribute__(sect)[key]
            vb = bi.__getattribute__(sect)[key]
            if not close(va, vb, atol=atol, rtol=rtol):
                return (i+1, f"{sect}.{key}", va, vb)
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Compare decision variables per step between DVODE (Fortran) and C++ VODE for the first steps.")
    p.add_argument('dvode_log')
    p.add_argument('vode_cpp_log')
    p.add_argument('--limit', type=int, default=50, help='Number of accepted steps to compare (default 50)')
    p.add_argument('--atol', type=float, default=1e-14, help='Absolute tolerance (default 1e-14)')
    p.add_argument('--rtol', type=float, default=1e-10, help='Relative tolerance (default 1e-10)')
    args = p.parse_args()

    a = parse_dvode(args.dvode_log, args.limit)
    b = parse_cpp(args.vode_cpp_log, args.limit)

    print(f"DVODE accepted steps parsed: {len(a)}  |  C++ VODE accepted steps parsed: {len(b)}")
    if not a or not b:
        print("Missing data; ensure both logs contain debug lines.")
        return

    div = compare_sequences(a, b, atol=args.atol, rtol=args.rtol)
    if div is None:
        print(f"No divergence found within first {min(len(a), len(b))} accepted steps at tolerances atol={args.atol}, rtol={args.rtol}.")
        return
    step_idx, field, va, vb = div
    print(f"Divergence at accepted step {step_idx} on {field}:")
    print(f"  DVODE: {va:.16e}")
    print(f"  C++  : {vb:.16e}")

    # show a short per-step summary around the divergence
    start = max(1, step_idx - 2)
    end = min(args.limit, step_idx + 2)
    print("\nContext (accepted steps):")
    for i in range(start, end + 1):
        if i > len(a) or i > len(b):
            break
        ai = a[i-1]; bi = b[i-1]
        print(f"  Step {i}: t_dv={ai.accept.get('t', float('nan')):.10e}  t_cpp={bi.accept.get('t', float('nan')):.10e}  hu_dv={ai.accept.get('hu', float('nan')):.6e}  hu_cpp={bi.accept.get('hu', float('nan')):.6e}  nq_dv={ai.accept.get('nq', float('nan')):.0f}  nq_cpp={bi.accept.get('nq', float('nan')):.0f}")


if __name__ == '__main__':
    main()

