// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: Main header file for the header-only integrator library
// ABOUTME: Includes all available integrator implementations
#ifndef INTEGRATORS_HPP
#define INTEGRATORS_HPP

#include "integrator_types.hpp"
#include "linear_algebra.hpp"
#include "backward_euler.hpp"
#include "vode.hpp"

namespace integrators {

// Factory function to create integrator instances
template<typename Problem>
struct IntegratorFactory {
    
    enum class Type { BACKWARD_EULER, VODE };
    
    template<Type IntType>
    static auto create() {
        if constexpr (IntType == Type::BACKWARD_EULER) {
            return BackwardEuler<Problem>{};
        } else if constexpr (IntType == Type::VODE) {
            return VODE<Problem>{};
        }
    }
    
    template<Type IntType>
    using state_type = std::conditional_t<
        IntType == Type::BACKWARD_EULER,
        BackwardEulerState<ProblemTraits<Problem>::neqs>,
        VODEState<ProblemTraits<Problem>::neqs>>;
};

// Convenience aliases
template<typename Problem>
using BE = BackwardEuler<Problem>;

template<typename Problem>
using VODE_Integrator = VODE<Problem>;

} // namespace integrators

#endif // INTEGRATORS_HPP
