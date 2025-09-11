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
├── vode.hpp            # VODE integrator
└── integrators.hpp     # Main header (includes all)

examples/               # Example programs
├── simple_ode.cpp     # Simple exponential decay
└── robertson.cpp      # Robertson chemical kinetics

tests/                 # Unit tests
├── test_linear_algebra.cpp  # Linear algebra tests
└── test_convergence.cpp     # Convergence order tests
```

## Credits

This library is extracted and adapted from the [AMReX Microphysics](https://github.com/AMReX-Astro/Microphysics) framework, which provides physics modules for astrophysical simulations. The original integrators were developed for stellar evolution and explosive astrophysics applications.

## License

This library is licensed under the 3-clause BSD license.
