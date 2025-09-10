# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a modern C++20 header-only library containing ODE integrators extracted from the AMReX Microphysics framework. The library provides two robust integrators:

- **VODE**: Variable-coefficient ODE solver using Backward Differentiation Formulas (BDF) - recommended for most problems
- **Backward Euler**: Simple implicit first-order method for very stiff problems

The library is completely self-contained with no external dependencies beyond the standard library (optional LAPACK integration available for improved numerical performance).

## Essential Commands

### Building and Testing
```bash
# Initial setup
mkdir build && cd build
cmake ..

# Build everything with parallelism
make -j

# Run all tests
ctest

# Run specific tests
./tests/test_linear_algebra
./tests/test_convergence
./examples/simple_ode
./examples/robertson
```

### CMake Configuration Options
```bash
# Enable LAPACK for improved numerical accuracy
cmake .. -DINTEGRATORS_USE_LAPACK=ON

# Enable verbose VODE debugging
cmake .. -DINTEGRATORS_VODE_DEBUG=ON

# Build without examples/tests
cmake .. -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF

# Debug build with sanitizers
cmake .. -DCMAKE_BUILD_TYPE=Debug
```

### Development Workflow
```bash
# Clean rebuild
rm -rf build && mkdir build && cd build && cmake .. && make -j

# Install headers system-wide
make install

# Run specific example with different parameters
./examples/robertson  # Robertson chemical kinetics problem
./examples/simple_ode # Simple exponential decay
```

## Project Architecture

### Header Organization
The library follows a modular header-only design:

- `integrator_types.hpp`: Core types, concepts, and mathematical utilities
- `linear_algebra.hpp`: Matrix operations and linear solvers (LU decomposition, etc.)
- `backward_euler.hpp`: First-order implicit integrator implementation
- `vode.hpp`: Variable-order BDF integrator (main production solver)
- `integrators.hpp`: Main include file with factory patterns and convenience aliases

### Problem Interface Requirements
User problems must implement this interface:
```cpp
struct YourProblem {
    static constexpr integrators::size_type neqs = N;
    using state_type = std::array<integrators::Real, neqs>;
    using rhs_type = std::array<integrators::Real, neqs>;
    using jacobian_type = std::array<std::array<integrators::Real, neqs>, neqs>;
    
    // Required: RHS function dy/dt = f(t, y)
    static void rhs(integrators::Real t, const state_type& y, rhs_type& dydt);
    
    // Optional but recommended: Analytic Jacobian df/dy
    static void jacobian(integrators::Real t, const state_type& y, jacobian_type& jac);
};
```

### Integrator State Management
Each integrator uses its own state structure:
- `BackwardEulerState<N>`: Simple state for Backward Euler
- `VODEState<N>`: Complex adaptive state with order/step control for VODE

### Modern C++ Features
The library extensively uses:
- C++20 concepts for compile-time interface validation
- Template metaprogramming for zero-cost abstractions
- `constexpr` mathematics utilities
- Standard library containers (`std::array`)

## Key Integration Points

### LAPACK Integration
When `INTEGRATORS_USE_LAPACK=ON` is enabled:
- Uses `dgetrf`/`dgetrs` for LU decomposition and solving
- Provides exact numerical compatibility with original DVODE Fortran code
- Falls back to built-in implementations when LAPACK unavailable

### VODE Debugging
Enable with `INTEGRATORS_VODE_DEBUG=ON` for verbose logging of:
- Step size selection and order changes
- Convergence behavior
- Error control decisions
- Internal state transitions

### Fortran Compatibility
The repository includes `dvode.f` and `robertson_dvode.f90` for:
- Numerical validation against original DVODE implementation
- Performance benchmarking
- Reference step size sequences

## Testing Strategy

### Unit Tests
- `test_linear_algebra.cpp`: Matrix operations and LU solver validation
- `test_convergence.cpp`: Convergence order verification for both integrators
- `test_vode_stiff_decay.cpp`: VODE-specific stiff problem tests
- `test_vode_robertson_strict.cpp`: Strict Robertson problem validation

### Example Programs
- `simple_ode.cpp`: Basic exponential decay problem
- `robertson.cpp`: Stiff chemical kinetics (Robertson problem)
- `robertson_dvode.f90`: Fortran reference implementation

### Continuous Integration
Uses CTest framework with automatic test discovery. All tests must pass for successful builds.

## Development Notes

1. **Header-Only Design**: No compilation required for users - just include headers
2. **Template-Heavy**: Extensive use of templates for performance and type safety
3. **AMReX Heritage**: Extracted from production astrophysics code, maintaining numerical accuracy
4. **Standards Compliance**: Requires C++20 with extensions disabled
5. **Cross-Platform**: Supports GCC, Clang, and MSVC compilers

## Performance Considerations

- Built-in linear algebra uses partial pivoting LU decomposition
- LAPACK integration available for maximum performance on large systems
- Template specialization eliminates runtime overhead
- Compile with `-O3` for production builds (handled automatically by CMake)