# Header-Only ODE Integrator Library

A modern C++20 header-only library containing ODE integrators extracted from the AMReX Microphysics framework. This library provides two robust integrators suitable for scientific computing applications:

- **VODE**: Variable-coefficient ODE solver using Backward Differentiation Formulas (BDF)
- **Backward Euler**: Simple implicit first-order method

## Features

- **Header-only**: No compilation required, just include the headers
- **Modern C++20**: Uses standard library containers and modern C++ features including concepts
- **Self-contained**: No external dependencies beyond standard library
- **Template-based**: Generic interfaces supporting different problem types

## Quick Start

```cpp
#include <integrators/integrators.hpp>

// Define your ODE problem
struct MyProblem {
    static constexpr integrators::size_type neqs = 2;
    using state_type = std::array<integrators::Real, neqs>;
    using rhs_type = std::array<integrators::Real, neqs>;
    
    // Define dy/dt = f(t, y)
    static void rhs(integrators::Real t, const state_type& y, rhs_type& dydt) {
        dydt[0] = -y[0] + y[1];
        dydt[1] = y[0] - y[1];
    }
};

// Use an integrator
auto integrator = integrators::BackwardEuler<MyProblem>{};
auto state = integrators::BackwardEulerState<2>{};

// Set up initial conditions
state.t = 0.0;
state.tout = 1.0;
state.y = {1.0, 0.0};
state.rtol = 1.e-6;
state.atol = 1.e-12;

MyProblem::state_type problem_state{};
auto result = integrator.integrate(problem_state, state);
```

## Build, Run, and Test

```bash
# Configure
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -j

# Run all tests
ctest --test-dir build --output-on-failure

# Run examples
build/examples/simple_ode
build/examples/robertson
```

Useful options:
- `-DBUILD_TESTS=ON` and `-DBUILD_EXAMPLES=ON` (both ON by default)
- `-DINTEGRATORS_VODE_DEBUG=ON` for verbose VODE internal logs
- Debug builds enable AddressSanitizer: `-DCMAKE_BUILD_TYPE=Debug`

## Development Tips

- Compiler selection: pass `-DCMAKE_CXX_COMPILER=/path/to/clang++` (or `g++`) to `cmake` or set `CXX` in the environment.
- CUDA tests: CUDA is enabled automatically if a CUDA compiler is detected. To force-enable, pass `-DCMAKE_CUDA_COMPILER=nvcc` (or a compatible Clang CUDA). The CUDA test target `test_vode_gpu` builds only when CUDA is available.
- LAPACK usage: system LAPACK is used when found (default `-DINTEGRATORS_USE_LAPACK=ON`); disable with `-DINTEGRATORS_USE_LAPACK=OFF` to use the built-in LU/solve.
- Reproducibility: record `-DCMAKE_CXX_COMPILER` and `-DCMAKE_BUILD_TYPE` with results. Use `-DINTEGRATORS_VODE_DEBUG=ON` for verbose VODE traces in Debug builds.

## Problem Interface Requirements

Your problem struct must provide:

```cpp
struct YourProblem {
    static constexpr integrators::size_type neqs = N;  // Number of equations
    using state_type = std::array<integrators::Real, neqs>;
    using rhs_type = std::array<integrators::Real, neqs>;
    
    // Required: RHS function dy/dt = f(t, y)
    static void rhs(integrators::Real t, const state_type& y, rhs_type& dydt);
    
    // Optional: Analytic Jacobian df/dy (recommended for stiff problems)
    static void jacobian(integrators::Real t, const state_type& y, jacobian_type& jac);
};
```

## Directory Structure

```
include/integrators/          # Header-only library (public API)
├── integrators.hpp           # Main umbrella header
├── integrator_types.hpp      # Core types, traits, states
├── linear_algebra.hpp        # Linear algebra utilities (LU/solve, helpers)
├── backward_euler.hpp        # Backward Euler integrator
└── vode.hpp                  # VODE (BDF) integrator

examples/                     # Example programs
├── simple_ode.cpp            # dy/dt = -y demo (BE + VODE)
├── robertson.cpp             # Robertson stiff kinetics (VODE)
└── robertson_dvode.f90       # Fortran driver to compare with DVODE

tests/                        # Executable tests (run via ctest)
├── test_linear_algebra.cpp   # Linear algebra unit tests
├── test_convergence.cpp      # BE and VODE convergence/error-control
├── test_vode_stiff_decay.cpp # Stiff decay regression
├── test_vode_hires.cpp       # HIRES stiff benchmark
├── test_vode_nelson.cpp      # Nelson astrochemistry check
├── test_vode_robertson_strict.cpp # Strict Robertson tolerances
└── test_vode_gpu.cu          # Optional CUDA test (if CUDA enabled)

extern/                       # External references for comparisons
└── dvode.f                   # Original DVODE Fortran source (reference)

scripts/                      # Utility scripts for comparisons/plots
├── compare_decisions.py
└── compare_steps.py

CMakeLists.txt                # Top-level CMake: builds examples/tests, installs headers
```

## Credits

This library is extracted and adapted from the [AMReX Microphysics](https://github.com/AMReX-Astro/Microphysics) framework, which provides physics modules for astrophysical simulations. The original integrators were developed for stellar evolution and explosive astrophysics applications.

## License

This library is licensed under the 3-clause BSD license.
