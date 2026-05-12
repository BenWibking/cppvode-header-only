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
