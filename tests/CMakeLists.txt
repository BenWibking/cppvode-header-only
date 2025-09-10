# ABOUTME: CMake configuration for unit tests
# ABOUTME: Tests individual components of the integrator library
add_executable(test_linear_algebra test_linear_algebra.cpp)
target_link_libraries(test_linear_algebra PRIVATE integrators)

add_executable(test_convergence test_convergence.cpp)  
target_link_libraries(test_convergence PRIVATE integrators)

# Add as tests
add_test(NAME linear_algebra_test COMMAND test_linear_algebra)
add_test(NAME convergence_test COMMAND test_convergence)

add_executable(test_vode_stiff_decay test_vode_stiff_decay.cpp)
target_link_libraries(test_vode_stiff_decay PRIVATE integrators)
add_test(NAME vode_stiff_decay_test COMMAND test_vode_stiff_decay)
