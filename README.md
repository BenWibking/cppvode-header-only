# Primordial ROS2S

This branch is a stripped-down primordial chemistry one-zone collapse example.
All source files are in this directory, and ROS2S is the only integrator path.

Build and run:

```sh
make
build/primordial_ros2s --grid 4 --perturb
build/primordial_ros2s_ref --grid 4 --perturb -N 8
build/primordial_ros2s --grid 4 --compare-final-state final_states_grid4_cpu.bin
```

CUDA and HIP builds use the same grid-step launcher:

```sh
make CUDA=1 CUDA_ARCHS=90
make HIP=1 HIP_ARCHS=gfx90a
```

The serial executable is `build/primordial_ros2s`. The threaded CPU reference
executable is `build/primordial_ros2s_ref`.

Options:

- `--grid N`: run `N^3` independent cells.
- `-N THREADS`: for `primordial_ros2s_ref`, run independent cells across this
  many CPU threads.
- `--perturb`: every 20 collapse steps, scale each cell's density and species by
  a deterministic random factor in `[0.9, 1.1]`.
- `--compare-final-state FILE`: after the run, read `FILE` as packed final-state
  records in the same format and print a PASS/FAIL comparison. The comparison
  uses the same network tolerances as the `cusolverdx` branch PASS test:
  per-species relative tolerances, `1e-4` relative tolerance for temperature and
  internal energy, `1e-4` relative tolerance for density, and the existing
  species/energy absolute tolerances.

For `--grid 1`, the final state is printed to stdout. For `--grid N` with
`N > 1`, stdout contains only the run summary and all cell final states are
written to `final_states_grid<N>_<backend>.bin` as consecutive packed records,
where `<backend>` is `cpu`, `cuda`, or `hip`:

If the output file already exists, it is first moved to a unique
`*.old.######` filename with randomly chosen digits, then the new output file is
written.

```c++
#pragma pack(push, 1)
struct PackedFinalState {
    int32_t cell;
    int32_t i, j, k;
    int32_t completed_steps;
    double time;
    double density_driver;
    double rho;
    double T;
    double e;
    double xn[14];
};
#pragma pack(pop)
```
