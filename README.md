# Header-Only ODE Integrator Library

A modern C++20 header-only library containing ODE integrators extracted from the AMReX Microphysics framework. This library provides three robust integrators suitable for scientific computing applications:

- **VODE**: Variable-coefficient ODE solver using Backward Differentiation Formulas (BDF)
- **RKC**: Runge-Kutta-Chebyshev method for problems with large spectral radius  
- **Backward Euler**: Simple implicit first-order method

## Features

- **Header-only**: No compilation required, just include the headers
- **Modern C++20**: Uses standard library containers and modern C++ features including concepts
- **Self-contained**: No external dependencies beyond standard library
- **Template-based**: Generic interfaces supporting different problem types
- **Extracted from production code**: Based on battle-tested AMReX Microphysics integrators

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

## Building Examples and Tests

```bash
mkdir build && cd build
cmake ..
make -j

# Run examples
./examples/simple_ode
./examples/robertson

# Run tests  
./tests/test_linear_algebra
./tests/test_convergence
```

## Integrator Selection Guide

### VODE (Recommended for most problems)
- **Best for**: General-purpose, stiff and non-stiff problems
- **Method**: Variable-order BDF with adaptive timestepping
- **Pros**: High accuracy, robust error control, efficient for stiff systems
- **Cons**: Most complex implementation

### Backward Euler
- **Best for**: Very stiff problems, simple implementation needs
- **Method**: First-order implicit method
- **Pros**: Unconditionally stable, simple, reliable for stiff problems  
- **Cons**: Only first-order accurate

### RKC (Runge-Kutta-Chebyshev)
- **Best for**: Problems with large spectral radius but not too stiff
- **Method**: Explicit method with extended stability region
- **Pros**: No linear solves needed, good for mildly stiff problems
- **Cons**: Explicit method limitations, requires spectral radius estimation

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
include/integrators/     # Header files
├── integrator_types.hpp # Base types and utilities  
├── linear_algebra.hpp   # Linear algebra utilities
├── backward_euler.hpp   # Backward Euler integrator
├── rkc.hpp             # RKC integrator
├── vode.hpp            # VODE integrator
└── integrators.hpp     # Main header (includes all)

examples/               # Example programs
├── simple_ode.cpp     # Simple exponential decay
└── robertson.cpp      # Robertson chemical kinetics

tests/                 # Unit tests
├── test_linear_algebra.cpp  # Linear algebra tests
└── test_convergence.cpp     # Convergence order tests
```

## Installation

As a header-only library, you can either:

1. **Copy headers**: Copy the `include/integrators/` directory to your project
2. **CMake install**: Use `make install` to install system-wide
3. **Package managers**: Add as git submodule or use with package managers

## Credits

This library is extracted and adapted from the [AMReX Microphysics](https://github.com/AMReX-Astro/Microphysics) framework, which provides physics modules for astrophysical simulations. The original integrators were developed for stellar evolution and explosive astrophysics applications.

## License

This library maintains compatibility with the original AMReX license terms.