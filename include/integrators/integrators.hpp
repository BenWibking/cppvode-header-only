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

namespace integrators {

// Factory function to create integrator instances
template<typename Problem>
struct IntegratorFactory {
    
    enum class Type { BACKWARD_EULER, YASS, VODE };
    
    template<Type IntType>
    static auto create() {
        if constexpr (IntType == Type::BACKWARD_EULER) {
            return BackwardEuler<Problem>{};
        } else if constexpr (IntType == Type::YASS) {
            return YASS<Problem>{};
        } else {
            static_assert(IntType == Type::VODE, "Unsupported integrator type");
            return VODE<Problem>{};
        }
    }
    
private:
    template<Type>
    struct state_selector;

    template<>
    struct state_selector<Type::BACKWARD_EULER> {
        using type = BackwardEulerState<ProblemTraits<Problem>::neqs>;
    };

    template<>
    struct state_selector<Type::YASS> {
        using type = YASSState<ProblemTraits<Problem>::neqs>;
    };

    template<>
    struct state_selector<Type::VODE> {
        using type = VODEState<ProblemTraits<Problem>::neqs>;
    };

public:
    template<Type IntType>
    using state_type = typename state_selector<IntType>::type;
};

// Convenience aliases
template<typename Problem>
using BE = BackwardEuler<Problem>;

template<typename Problem>
using YASS_Integrator = YASS<Problem>;

template<typename Problem>
using VODE_Integrator = VODE<Problem>;

} // namespace integrators

#endif // INTEGRATORS_HPP
