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

# HIRES stiff benchmark (VODE)
add_executable(test_vode_hires test_vode_hires.cpp)
target_link_libraries(test_vode_hires PRIVATE integrators)
add_test(NAME vode_hires_test COMMAND test_vode_hires)

# Nelson astrochemistry L2 relative error test
add_executable(test_vode_nelson test_vode_nelson.cpp)
target_link_libraries(test_vode_nelson PRIVATE integrators)
add_test(NAME vode_nelson_test COMMAND test_vode_nelson)

# GPU test (requires CUDA)
if(CMAKE_CUDA_COMPILER)
    add_executable(test_vode_gpu test_vode_gpu.cu)
    # Ensure CUDA compiles with C++20 to match headers
    set_target_properties(test_vode_gpu PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 90)
    # Enable extended device lambda support for NVCC
    target_compile_options(test_vode_gpu PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>)
    target_link_libraries(test_vode_gpu PRIVATE integrators)
    add_test(NAME vode_gpu_test COMMAND test_vode_gpu)
    # Mark as GPU test so CI can identify/skip running if no GPU is present
    set_tests_properties(vode_gpu_test PROPERTIES LABELS "GPU")
endif()
