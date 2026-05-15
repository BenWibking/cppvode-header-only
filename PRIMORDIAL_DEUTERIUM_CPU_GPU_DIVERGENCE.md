# Primordial Deuterium CPU/GPU Divergence

This note records the investigation into why the primordial chemistry example
diverges in deuterium-bearing species between CPU and CUDA VODE runs.

## Reproduction

The isolated failing stream is a single-cell run using the same logical cell id
as the first failing 32^3 perturbed CUDA cell:

```sh
build-cpu/examples/primordial_chem --grid 1 --cell-id 1622 --integrator=vode --perturb
build/examples/primordial_chem --grid 1 --cell-id 1622 --integrator=vode --perturb
```

The CUDA run fails around collapse step 368. At the failure point the thermodynamic
state and non-deuterium species are close, but the deuterium-bearing species have
already separated by roughly 25-30%.

## Useful Diagnostics

Two diagnostic CLI paths were added to `examples/primordial_chem.cpp`:

```sh
--dump-history
--trace-vode --trace-cell-id N --trace-step S
```

`--dump-history` prints one CSV row per collapse step for the representative cell.
This showed that the initial deuterium divergence is already present after
collapse step 0, before any perturbation is applied. The same step-0 split occurs
in unperturbed runs, so perturbations are not the origin; they only expose and
amplify the existing CPU/GPU branch.

`--trace-vode` prints internal `vode_trace` rows for the selected burn. The trace
includes the initial RHS, chosen initial step size, nonlinear corrector RHS/correction
states, accepted internal VODE steps, and final interpolation.

## Step-0 Collapse State Difference

After collapse step 0, CPU and GPU thermodynamic values differ only at roundoff:

```text
T relative difference:    ~1.3e-16
Eint relative difference: ~1.7e-16
H/Hp relative difference: ~5e-15
```

The deuterium channels, however, already differ substantially:

```text
Dp:  CPU 1.6272283788908528e-24   GPU 1.9881552323587847e-24
D:   CPU 2.7422183834953766e-20   GPU 3.3394850690709068e-20
HDp: CPU 1.0e-100                 GPU 9.342488659938672e-34
HD:  CPU 1.0e-100                 GPU 1.2035257137339023e-20
```

## Exact Internal Mechanism

The step-0 VODE burn starts identically on CPU and GPU for the selected species.
The first measurable numerical difference is only a one-ulp-level RHS difference:

```text
init_rhs fHD relative difference: ~1.36e-16
```

The first accepted internal VODE step is identical. During the second accepted
internal step, tiny correction-level differences appear in the deuterium variables:

```text
Dp relative difference: ~1.6e-5
D  relative difference: ~1.6e-5
HD relative difference: ~2.2e-6
```

By the fourth accepted internal VODE step, `HD` has crossed zero on CPU but not
on GPU:

```text
CPU HD = -3.0916722267455132e-21
GPU HD =  2.8875073257878925e-21
```

The discontinuity comes from `VODE::rhs_state()` calling `clean_state_vector()`
before each chemistry RHS evaluation:

```cpp
value = std::max(value, s.component_floor);
```

Thus the next predicted RHS input is different by many orders of magnitude:

```text
CPU HD = 1.0e-100
GPU HD = 9.497767206261249e-21
```

That changes the `Hp * HD` destruction/source channel immediately and sends the
rare deuterium network down a different trajectory. The divergence is therefore
not caused by a gross GPU evaluation error in `exp`, `log`, or `sqrt` at identical
input states. It is caused by tiny floating-point differences being amplified by
a near-zero, positivity-constrained stiff solve.

## Dominant Downstream Chemistry Terms

Once the histories have diverged, the relevant generated terms in
`include/integrators/primordial_chem.hpp` are:

```text
x51  = Hp * HD channel
x112 = -x51
x101 = D * H2 * k(D + H2 -> HD + H)
x102 = HD * H * k(HD + H -> D + H2)
x103 = x101 - x102
x115 = x103 + secondary HD source terms
```

At collapse step 368, the near cancellation in `x101 - x102` has opposite signs
between the CPU-history and GPU-history states:

```text
CPU-history:
x101 = 39.224900234492424
x102 = 39.224901491031495
x103 = -1.2565390719e-6
ydot_D  = +1.2756987647e-6
ydot_HD = -1.2756985950e-6

GPU-history:
x101 = 28.529912943952446
x102 = 28.529912289770895
x103 = +6.5418155160e-7
ydot_D  = -6.4024423618e-7
ydot_HD = +6.4024436168e-7
```

## Conclusion

The CPU/GPU divergence is seeded inside the first VODE burn. It is a trajectory
branch caused by:

1. tiny CPU/GPU roundoff in the nonlinear solve,
2. amplification in floor-level deuterium species,
3. an `HD` sign difference,
4. discontinuous positivity cleanup before RHS evaluation, and
5. subsequent deuterium-network cancellation around `x101 - x102`.

The special functions themselves agree to printed precision when evaluated on
identical states. The instability is in the interaction between adaptive stiff
integration, near-floor abundances, and positivity enforcement.
