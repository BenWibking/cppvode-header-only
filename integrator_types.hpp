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
#if defined(__CUDACC__)
#define INTEGRATORS_HOST_DEVICE __host__ __device__ __forceinline__
#elif defined(__HIPCC__)
#define INTEGRATORS_HOST_DEVICE __host__ __device__ __attribute__((always_inline)) inline
#else
#define INTEGRATORS_HOST_DEVICE inline
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
    TOO_MANY_STEPS = -4,
    TOO_MUCH_ACCURACY_REQUESTED = -5,
    LU_DECOMPOSITION_ERROR = -7
};

namespace detail {

template<typename Problem, size_type N, typename = void>
struct problem_jacobian {
    static constexpr bool available = false;
};

template<typename Problem, size_type N>
struct problem_jacobian<Problem, N, std::void_t<typename Problem::jacobian_type>> {
    static constexpr bool available = true;
};

} // namespace detail

// Base traits for problem definition
template<typename Problem>
struct ProblemTraits {
    static constexpr size_type neqs = Problem::neqs;
    using state_type = typename Problem::state_type;
    static constexpr bool has_analytic_jacobian = detail::problem_jacobian<Problem, neqs>::available;
};

// Math utilities
namespace math {

template<int N>
constexpr Real powi(Real x) {
    if constexpr (N < 0) return 1.0 / powi<-N>(x);
    else if constexpr (N == 0) return 1.0;
    else if constexpr (N == 1) return x;
    else if constexpr (N == 2) return x * x;
    else if constexpr (N % 2 == 0) return powi<2>(powi<N / 2>(x));
    else return x * powi<N - 1>(x);
}

} // namespace math

} // namespace integrators

#endif // INTEGRATOR_TYPES_HPP
