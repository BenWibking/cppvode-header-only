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

    # Choose the highest CUDA C++ standard supported; fall back to C++17 if CUDA20 isn't available.
    # Use several heuristics because older CMake versions may not advertise features for the CUDA toolchain.
    set(_cuda_std_msg "unknown")
    if(DEFINED CMAKE_CUDA20_STANDARD_COMPILE_OPTION OR "cuda_std_20" IN_LIST CMAKE_CUDA_COMPILE_FEATURES OR CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0)
        set_target_properties(test_vode_gpu PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED OFF)
        set(_cuda_std_msg "C++20")
    elseif("cuda_std_17" IN_LIST CMAKE_CUDA_COMPILE_FEATURES OR CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.0)
        set_target_properties(test_vode_gpu PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED OFF)
        set(_cuda_std_msg "C++17")
    else()
        # Last resort for very old CMake/compilers: try a manual flag.
        target_compile_options(test_vode_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-std=c++17>)
        set(_cuda_std_msg "C++17 (manual flag)")
    endif()
    message(STATUS "test_vode_gpu: using ${_cuda_std_msg} for CUDA (compiler ${CMAKE_CUDA_COMPILER_VERSION})")

    # Common CUDA target properties
    set_target_properties(test_vode_gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 90)

    # Enable extended device lambda support for NVCC
    target_compile_options(test_vode_gpu PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>)

    # Silence Clang host warnings about GNU-style line markers produced by NVCC
    # Only apply when compiling CUDA with a Clang host to avoid breaking GCC/MSVC.
    target_compile_options(test_vode_gpu PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:Clang>>:-Xcompiler=-Wno-gnu-line-marker>)
    target_link_libraries(test_vode_gpu PRIVATE integrators)
    add_test(NAME vode_gpu_test COMMAND test_vode_gpu)
    # Mark as GPU test so CI can identify/skip running if no GPU is present
    set_tests_properties(vode_gpu_test PROPERTIES LABELS "GPU")
endif()
