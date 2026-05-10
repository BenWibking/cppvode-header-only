// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: Main header file for the header-only integrator library
// ABOUTME: Includes all available integrator implementations
#ifndef INTEGRATORS_HPP
#define INTEGRATORS_HPP

#include "integrator_types.hpp"
#include "linear_algebra.hpp"
#include "backward_euler.hpp"
#include "yass.hpp"
#include "vode.hpp"
#include "rodas.hpp"

namespace integrators {

// Factory function to create integrator instances
template<typename Problem>
struct IntegratorFactory {
    
    enum class Type { BACKWARD_EULER, YASS, VODE, RODAS };
    
    template<Type IntType>
    static INTEGRATORS_HOST_DEVICE auto create() {
        if constexpr (IntType == Type::BACKWARD_EULER) {
            return BackwardEuler<Problem>{};
        } else if constexpr (IntType == Type::YASS) {
            return YASS<Problem>{};
        } else if constexpr (IntType == Type::RODAS) {
            return RODAS<Problem>{};
        } else {
            static_assert(IntType == Type::VODE, "Unsupported integrator type");
            return VODE<Problem>{};
        }
    }
    
private:
    template<typename State>
    struct state_identity {
        using type = State;
    };

    template<Type IntType>
    static consteval auto select_state() {
        if constexpr (IntType == Type::BACKWARD_EULER) {
            return state_identity<BackwardEulerState<ProblemTraits<Problem>::neqs>>{};
        } else if constexpr (IntType == Type::YASS) {
            return state_identity<YASSState<ProblemTraits<Problem>::neqs>>{};
        } else if constexpr (IntType == Type::RODAS) {
            return state_identity<RODASState<ProblemTraits<Problem>::neqs>>{};
        } else {
            static_assert(IntType == Type::VODE, "Unsupported integrator type");
            return state_identity<VODEState<ProblemTraits<Problem>::neqs>>{};
        }
    }

public:
    template<Type IntType>
    using state_type = typename decltype(select_state<IntType>())::type;
};

// Convenience aliases
template<typename Problem>
using BE = BackwardEuler<Problem>;

template<typename Problem>
using YASS_Integrator = YASS<Problem>;

template<typename Problem>
using VODE_Integrator = VODE<Problem>;

template<typename Problem>
using RODAS_Integrator = RODAS<Problem>;

} // namespace integrators

#endif // INTEGRATORS_HPP
