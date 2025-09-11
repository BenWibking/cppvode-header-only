# ABOUTME: CMake build configuration for header-only integrator library
# ABOUTME: Builds examples and tests with modern C++ standards
cmake_minimum_required(VERSION 3.14)

project(HeaderOnlyIntegrators 
    VERSION 1.0.0
    DESCRIPTION "Header-only ODE integrator library extracted from AMReX Microphysics"
    LANGUAGES CXX Fortran)

# Optionally enable CUDA if a compiler is available (for GPU tests)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "CUDA compiler detected: enabling CUDA language")
    # Suppress host compiler pedantic warnings when compiling CUDA sources
    # Route to host via nvcc's -Xcompiler to avoid nvcc parsing issues
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-pedantic>)
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add the header-only library as interface target
add_library(integrators INTERFACE)
target_include_directories(integrators INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# System LAPACK support removed; always use built-in LU/solve

# Optional verbose debug logging for VODE internals
# Enable with: -DINTEGRATORS_VODE_DEBUG=ON
option(INTEGRATORS_VODE_DEBUG "Enable verbose VODE debug logging" OFF)
if(INTEGRATORS_VODE_DEBUG)
    target_compile_definitions(integrators INTERFACE VODE_DEBUG)
endif()

# Treat warnings as errors (opt-in, excludes CUDA/Fortran)
option(WARNINGS_AS_ERRORS "Treat C/C++ warnings as errors (excluding CUDA/Fortran)" OFF)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(integrators INTERFACE
        -Wall -Wextra -Wpedantic -O3
        $<$<CONFIG:Debug>:-g -O0 -fsanitize=address>
    )
    if(WARNINGS_AS_ERRORS)
        target_compile_options(integrators INTERFACE
            $<$<COMPILE_LANGUAGE:C>:-Werror>
            $<$<COMPILE_LANGUAGE:CXX>:-Werror>
        )
    endif()
    target_link_options(integrators INTERFACE
        $<$<CONFIG:Debug>:-fsanitize=address>
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(integrators INTERFACE /W4 /O2)
    if(WARNINGS_AS_ERRORS)
        target_compile_options(integrators INTERFACE
            $<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:/WX>
        )
    endif()
endif()

# Enable testing
enable_testing()

# Build examples
option(BUILD_EXAMPLES "Build example programs" ON)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Build tests
option(BUILD_TESTS "Build test programs" ON)
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

# Installation (header-only)
install(DIRECTORY include/ 
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp")

# Export targets
install(TARGETS integrators
    EXPORT integrators-targets
    INCLUDES DESTINATION include)

install(EXPORT integrators-targets
    FILE integrators-config.cmake
    DESTINATION lib/cmake/integrators)

# Generate version file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/integrators-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES 
    "${CMAKE_CURRENT_BINARY_DIR}/integrators-config-version.cmake"
    DESTINATION lib/cmake/integrators)
