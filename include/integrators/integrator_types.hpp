// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: Header-only integrator library - core type definitions and interfaces
// ABOUTME: Extracted from AMReX Microphysics with AMReX dependencies removed
#ifndef INTEGRATOR_TYPES_HPP
#define INTEGRATOR_TYPES_HPP

#include <cmath>
#include <limits>
#include <array>
#include <algorithm>
#include <type_traits>

#ifndef INTEGRATORS_HOST_DEVICE
#if defined(__CUDACC__) || defined(__HIPCC__)
#define INTEGRATORS_HOST_DEVICE __host__ __device__
#else
#define INTEGRATORS_HOST_DEVICE
#endif
#endif

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

// Identity preconditioner marker used when a problem does not provide one.
struct IdentityPreconditioner {};

enum class PreconditionerSide {
    NONE,
    LEFT,
    RIGHT,
    BOTH
};

namespace detail {

template<typename Problem, typename = void>
struct problem_preconditioner {
    using type = IdentityPreconditioner;
    static constexpr bool available = false;
};

template<typename Problem>
struct problem_preconditioner<Problem, std::void_t<typename Problem::preconditioner_type>> {
    using type = typename Problem::preconditioner_type;
    static constexpr bool available = true;
};

template<typename Problem, size_type N, typename = void>
struct problem_jacobian {
    using type = std::array<std::array<Real, N>, N>;
    static constexpr bool available = false;
};

template<typename Problem, size_type N>
struct problem_jacobian<Problem, N, std::void_t<typename Problem::jacobian_type>> {
    using type = typename Problem::jacobian_type;
    static constexpr bool available = true;
};

} // namespace detail

// Base traits for problem definition
template<typename Problem>
struct ProblemTraits {
    static constexpr size_type neqs = Problem::neqs;
    using state_type = typename Problem::state_type;
    using rhs_type = typename Problem::rhs_type;
    using jacobian_type = typename detail::problem_jacobian<Problem, neqs>::type;
    using preconditioner_type = typename detail::problem_preconditioner<Problem>::type;
    static constexpr bool has_custom_preconditioner = detail::problem_preconditioner<Problem>::available;
    static constexpr bool has_analytic_jacobian = detail::problem_jacobian<Problem, neqs>::available;
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
    bool use_vector_tolerances{false};
    std::array<Real, N> rtol_vec{};
    std::array<Real, N> atol_vec{};
    
    // Statistics
    int n_step{0};
    int n_rhs{0};
    int n_jac{0};
    
    // Solution vector
    std::array<Real, N> y{};
    
    bool jacobian_analytic{false};
    int constrained_components{0};
    Real species_failure_tolerance{1.e-2};
    Real reject_change_buffer{0.0};
    Real increase_change_factor{4.0};
    Real decrease_change_factor{0.25};
    bool enforce_component_ceiling{false};
    Real component_ceiling{1.0};
    bool clean_constrained_components{false};
    Real component_floor{0.0};
};

// RHS/Jacobian concepts (available when compiling with C++20 concepts support).
// Some CUDA toolchains in CI may not expose CUDA20 to CMake; guard for portability.
#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
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
                                    typename ProblemTraits<Problem>::jacobian_type& jac,
                                    Real t) {
    Problem::jacobian(t, state, jac);
};
#else
// Fallback placeholders when concepts are unavailable; not used in the library code paths.
template<typename Problem>
using RHSFunction = int;
template<typename Problem>
using JacobianFunction = int;
#endif

// Math utilities
namespace math {

constexpr Real UROUND = std::numeric_limits<Real>::epsilon();

template<int N>
constexpr Real powi(Real x) {
    if constexpr (N < 0) return 1.0 / powi<-N>(x);
    else if constexpr (N == 0) return 1.0;
    else if constexpr (N == 1) return x;
    else if constexpr (N == 2) return x * x;
    else if constexpr (N % 2 == 0) return powi<2>(powi<N / 2>(x));
    else return x * powi<N - 1>(x);
}

template<typename T>
constexpr T clamp(const T& value, const T& low, const T& high) {
    return std::min(std::max(value, low), high);
}

} // namespace math

} // namespace integrators

#endif // INTEGRATOR_TYPES_HPP
