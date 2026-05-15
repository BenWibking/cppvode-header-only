# Primordial Chemistry GPU Performance Tuning Notes

This note summarizes the ROS2S vs VODE GPU tuning experiments for the primordial
chemistry benchmark. The main target was the CUDA `collapse_step_kernel` path in
`examples/primordial_chem.cpp`, using H200-class GPUs and CUDA event timings.

## Current Best Configuration

The best valid ROS2S configuration found so far is the specialized ROS2S path
with dense pivoted factorization, compact RHS scratch, static tolerances, stats
specialization, and active kernel instantiations only.

Relevant build options:

```text
-DPRIMORDIAL_CHEM_ROS2S_STATIC_TOLERANCES=1
-DPRIMORDIAL_CHEM_ROS2S_SPECIALIZE_STATS=1
-DPRIMORDIAL_CHEM_ACTIVE_KERNELS_ONLY=1
```

Compact RHS scratch is now always selected when a problem declares that its RHS
supports input/output aliasing.

In this configuration, the resource usage was:

```text
ROS2S analytic/no-stats: REG 255  STACK 5840
VODE analytic/stats:     REG 255  STACK 7120
```

Representative event timings:

```text
32^3: VODE  ~6.36 s, ROS2S ~7.04-7.06 s
64^3: VODE  ~51.70 s, ROS2S ~55.79 s
```

With the current optimized VODE path, ROS2S is still slower, but the gap is much
smaller than it was before the ROS2S specialization work.

## Main Lessons

Stack size is highly predictive of performance for this kernel. Changes that
increase stack or local-memory traffic have usually made performance worse, even
when they reduce floating-point work.

Kernel count is also highly predictive. Splitting work into separate kernels was
bad for this benchmark and should not be revisited without a very different
problem structure. The successful direction is a single specialized kernel per
integrator, with runtime dispatch to the chosen kernel.

VODE must be compared against its tuned version. Earlier results showed ROS2S
slightly faster than VODE, but those measurements were before VODE received the
same specialization and tuning attention. After tuning VODE, ROS2S is behind
again.

The useful optimizations were the ones that reduced per-thread state or removed
inactive code paths without increasing stack pressure:

- Specializing the Rosenbrock implementation completely to ROS2S.
- Removing RODAS4 support from the active path.
- Specializing stats collection so the no-stats ROS2S kernel does not carry
  stats fields.
- Using static ROS2S tolerances so tolerance arrays do not need to live in the
  state.
- Using compact RHS scratch where the primordial RHS supports input/output
  aliasing.
- Building only the active kernel instantiations to reduce compile time and keep
  the generated code surface smaller.
- Assembling the shifted-negated Jacobian directly when an analytic Jacobian is
  available.

## Experiments That Did Not Help

### Sparse Factorization Prototype

A symbolic sparse factorization ordering was implemented to exploit the
Jacobian sparsity. It minimized fill reasonably well, but the generated
factorization was much worse for GPU performance. The implementation has since
been removed.

Representative result:

```text
32^3: VODE ~6.38 s, ROS2S sparse factorization ~34.26 s
ROS2S sparse factorization stack: ~7936 bytes
```

The sparse path increased stack pressure and introduced too much scalar
bookkeeping. For this small 15-variable system, dense pivoted factorization is
currently better on GPU.

### External Matrix Storage

Moving the ROS2S matrix out of the per-thread state reduced stack size, but it
forced far more global-memory traffic. The option has since been removed.

Representative result:

```text
ROS2S external matrix stack: ~4656 bytes
32^3 ROS2S external matrix: ~18.02 s
```

This is a useful example of why stack reduction alone is not sufficient. Moving
hot per-thread data to global memory was much worse.

### No-Pivot Factorization

Disabling pivoting did not produce a valid or faster result. The ROS2S
no-pivot option has since been removed.

Representative result:

```text
ROS2S no-pivot stack: ~6752 bytes
32^3: VODE ~6.37 s, ROS2S no-pivot ~7.20 s
Reference comparison: FAIL
```

No-pivot factorization should not be used for this problem.

### Removing `ynew`

Removing the stored `ynew` vector reduced reported stack size, but slowed the
kernel. The change likely made scheduling or live ranges worse enough to offset
the apparent stack improvement.

Representative result:

```text
ROS2S stack: 5840 -> 5720 bytes
32^3 ROS2S: ~7.07-7.11 s
64^3 ROS2S: ~56.81 s, compared with ~55.79 s before removal
```

The change was reverted. Do not remove `ynew` again unless there is new evidence
from a compiler-resource and timing experiment.

### Register Capping

Forcing a lower maximum register count caused spilling and slowed the kernels.

Example with `--maxrregcount=224`:

```text
ROS2S: REG 224  STACK 6032
VODE:  REG 224  STACK 7392

32^3: VODE ~6.87 s, ROS2S ~7.28 s
```

This was worse than the uncapped 255-register build for both integrators. A
240-register build was also checked for resource usage. It has not been timed
yet, but it still showed increased stack relative to baseline:

```text
ROS2S: REG 240  STACK 5920
VODE:  REG 240  STACK 7336
```

The 240-register build is less damaging than the 224-register build, but the
resource report points in the same direction: register caps are trading register
pressure for stack/local pressure, which has been a bad exchange for this
kernel.

The evidence so far says that blunt register caps are not a good tuning lever for
this kernel.

## Directions To Avoid

These directions either made performance much worse or were consistently worse
than the current dense pivoted single-kernel ROS2S path:

- Do not split ROS2S into multiple kernels.
- Do not move hot per-thread matrix storage to global memory.
- Do not reintroduce the removed sparse factorization prototype for this
  benchmark.
- Do not disable pivoting for primordial chemistry.
- Do not remove `ynew` solely to reduce stack.
- Do not pursue broad `--maxrregcount` caps unless a very narrow cap is tested
  with both resource reports and event timings.
- Do not optimize only for lower flop count if the change increases stack,
  register pressure, local-memory traffic, or scalar bookkeeping.
- Do not compare ROS2S against an untuned VODE baseline.

## Remaining Plausible Directions

The remaining credible options are narrow. They should be tested one at a time
with resource reports first, then dedicated-GPU event timings only when the
resource usage is plausible.

### Reduce ROS2S Live Ranges

Inspect the ROS2S step, `decompose`, `solve`, Jacobian assembly, and
error-control code for values that stay live across calls or loops
unnecessarily. This is the most plausible path because stack/register pressure
has been the strongest predictor of performance.

- Inspect long live ranges in the ROS2S step and try to shorten them without
  removing useful storage that improves scheduling.
- Move temporary calculations into narrower scopes where that reduces live
  overlap.
- Avoid keeping both old and transformed scalar forms live when one can be
  recomputed cheaply without increasing stack.
- Do not remove useful storage such as `ynew` unless a new resource and timing
  experiment proves that the compiler handles the replacement better.

### Restructure Dense LU Storage

Keep the dense pivoted LU algorithm, but look for ways to reduce temporary arrays
and lifetime overlap in the matrix factorization and solve path.

- Preserve pivoting and the dense representation.
- Avoid sparse/no-pivot variants; those have already failed for this problem.
- Avoid moving hot per-thread data to global memory.

### Specialize Remaining Runtime Constants

Some runtime branches or state fields may still be constant for the primordial
chemistry benchmark and could become template/static configuration.

- Continue using active kernel instantiations only for timing builds.
- Look for remaining branches in ROS2S setup, tolerances, Jacobian handling, and
  diagnostics that are fixed in the benchmark configuration.
- Specialization is only useful if it reduces resource usage or code size without
  increasing stack.

### Resource-Guided Micro-Edits

Make small edits, rebuild, and inspect `cuobjdump --dump-resource-usage` before
running longer benchmarks.

- Good candidates are small scope changes, scalar-temporary reductions, and
  simplifications that reduce lifetime overlap.
- A lower flop count is not enough; the edit must preserve or improve stack,
  register, and local-memory behavior.
- Keep resource reports next to every timing result. A lower stack number is only
  useful if event timing also improves.

### Launch Bounds

Dedicated-GPU Slurm measurements are pending for launch-bounds variants. Local
GPU timing was contaminated by other active jobs and should be ignored.

Resource reports so far suggest:

- `launch_bounds(256, 1)` leaves ROS2S resource usage unchanged and slightly
  worsens VODE stack.
- `launch_bounds(256, 2)` behaves like register capping: it forces fewer
  registers and increases stack substantially.
- `launch_bounds(128, 1)` and `launch_bounds(128, 2)` did not change the active
  kernel resource usage relative to the corresponding baseline build.

The plausible launch-bounds cases are therefore the `minBlocks=1` variants. The
`minBlocks=2` case should be treated as suspicious unless the dedicated-GPU
timing contradicts the resource report.

### Better Timing Decomposition

CUDA events around narrower regions could determine whether remaining ROS2S cost
comes from integration math, result handling, state setup, or memory movement.
This is diagnostic rather than an optimization, but it can guide the next edit.

## Suggested Benchmark Discipline

For future tuning runs:

1. Build one configuration at a time with active kernels only.
2. Record `cuobjdump --dump-resource-usage` for the active VODE and ROS2S
   kernels.
3. Run at least `32^3` and `64^3` GPU comparisons with CUDA event timings.
4. Avoid CPU multi-cell runs; use multi-cell grids only on GPU.
5. Treat any optimization that increases ROS2S stack as suspicious until timing
   proves otherwise.
