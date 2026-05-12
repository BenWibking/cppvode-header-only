# Primordial ROS2S

This branch is a stripped-down primordial chemistry one-zone collapse example.
All source files are in this directory, and ROS2S is the only integrator path.

Build and run:

```sh
make
build/primordial_ros2s --grid 4 --perturb
```

The executable is `build/primordial_ros2s`.

Options:

- `--grid N`: run `N^3` independent cells.
- `--perturb`: every 20 collapse steps, scale each cell's density and species by
  a deterministic random factor in `[0.9, 1.1]`.

For `--grid 1`, the final state is printed to stdout. For `--grid N` with
`N > 1`, stdout contains only the run summary and all cell final states are
written to `final_states_grid<N>_<backend>.bin` as consecutive packed records,
where `<backend>` is `cpu`, `cuda`, or `hip`:

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
