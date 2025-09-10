// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: Header-only integrator library - core type definitions and interfaces
// ABOUTME: Extracted from AMReX Microphysics with AMReX dependencies removed
#ifndef INTEGRATOR_TYPES_HPP
#define INTEGRATOR_TYPES_HPP

#include <cmath>
#include <limits>
#include <array>
#include <algorithm>

namespace integrators {

// Basic real type - can be changed to double/long double as needed
using Real = double;
using size_type = std::size_t;

// Integration error codes
enum class IntegratorResult : int {
    SUCCESS = 1,
    BAD_INPUTS = -1,
    DT_UNDERFLOW = -2,
    SPRAD_CONVERGENCE = -3,
    TOO_MANY_STEPS = -4,
    TOO_MUCH_ACCURACY_REQUESTED = -5,
    CORRECTOR_CONVERGENCE = -6,
    LU_DECOMPOSITION_ERROR = -7
};

// Base traits for problem definition
template<typename Problem>
struct ProblemTraits {
    static constexpr size_type neqs = Problem::neqs;
    using state_type = typename Problem::state_type;
    using rhs_type = typename Problem::rhs_type;
    using jacobian_type = typename Problem::jacobian_type;
};

// Base integrator state
template<size_type N>
struct IntegratorState {
    Real t{0.0};        // current time
    Real tout{0.0};     // target time
    Real dt{0.0};       // timestep
    
    // Tolerances
    Real rtol{1.e-6};
    Real atol{1.e-12};
    
    // Statistics
    int n_step{0};
    int n_rhs{0};
    int n_jac{0};
    
    // Solution vector
    std::array<Real, N> y{};
    
    bool jacobian_analytic{false};
};

// RHS function interface
template<typename Problem>
concept RHSFunction = requires(const typename Problem::state_type& state, 
                              typename Problem::rhs_type& rhs, 
                              Real t) {
    Problem::rhs(t, state, rhs);
};

// Jacobian function interface  
template<typename Problem>
concept JacobianFunction = requires(const typename Problem::state_type& state,
                                   typename Problem::jacobian_type& jac,
                                   Real t) {
    Problem::jacobian(t, state, jac);
};

// Math utilities
namespace math {

constexpr Real UROUND = std::numeric_limits<Real>::epsilon();

template<int N>
constexpr Real powi(Real x) {
    if constexpr (N == 0) return 1.0;
    if constexpr (N == 1) return x;
    if constexpr (N == 2) return x * x;
    if constexpr (N == 3) return x * x * x;
    if constexpr (N < 0) return 1.0 / powi<-N>(x);
    else {
        Real result = 1.0;
        Real base = x;
        int exp = N;
        while (exp > 0) {
            if (exp & 1) result *= base;
            base *= base;
            exp >>= 1;
        }
        return result;
    }
}

template<typename T>
constexpr T clamp(const T& value, const T& low, const T& high) {
    return std::min(std::max(value, low), high);
}

} // namespace math

} // namespace integrators

#endif // INTEGRATOR_TYPES_HPP
