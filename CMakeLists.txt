cmake_minimum_required(VERSION 3.21)

project(primordial_ros2s LANGUAGES CXX)

option(PRIMORDIAL_ROS2S_ENABLE_CUDA "Build primordial_ros2s with a CUDA grid-step launcher" OFF)
option(PRIMORDIAL_ROS2S_ENABLE_HIP "Build primordial_ros2s with a HIP grid-step launcher" OFF)
option(PRIMORDIAL_ROS2S_CUDA_FMA "Enable fused multiply-add contraction in NVIDIA CUDA builds" OFF)
set(PRIMORDIAL_ROS2S_CUDA_THREADS_PER_BLOCK "128" CACHE STRING
    "GPU threads per block for the collapse grid-step launcher")
set(PRIMORDIAL_ROS2S_CUDA_MAXREGCOUNT "255" CACHE STRING
    "nvcc --maxrregcount value for primordial_ros2s CUDA builds")

if(PRIMORDIAL_ROS2S_ENABLE_CUDA AND PRIMORDIAL_ROS2S_ENABLE_HIP)
    message(FATAL_ERROR "Enable at most one of PRIMORDIAL_ROS2S_ENABLE_CUDA or PRIMORDIAL_ROS2S_ENABLE_HIP")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(PRIMORDIAL_ROS2S_ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
endif()
if(PRIMORDIAL_ROS2S_ENABLE_HIP)
    enable_language(HIP)
    set(CMAKE_HIP_STANDARD 20)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    set(CMAKE_HIP_EXTENSIONS OFF)
endif()

find_package(Threads REQUIRED)

add_executable(primordial_ros2s main.cpp)
target_include_directories(primordial_ros2s PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(PRIMORDIAL_ROS2S_ENABLE_CUDA)
    set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
    target_compile_definitions(primordial_ros2s PRIVATE
        PRIMORDIAL_ROS2S_ENABLE_CUDA
        PRIMORDIAL_ROS2S_CUDA_THREADS_PER_BLOCK=${PRIMORDIAL_ROS2S_CUDA_THREADS_PER_BLOCK})
endif()
if(PRIMORDIAL_ROS2S_ENABLE_HIP)
    set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
    target_compile_definitions(primordial_ros2s PRIVATE
        PRIMORDIAL_ROS2S_ENABLE_HIP
        PRIMORDIAL_ROS2S_CUDA_THREADS_PER_BLOCK=${PRIMORDIAL_ROS2S_CUDA_THREADS_PER_BLOCK})
endif()

add_executable(primordial_ros2s_ref main_ref.cpp)
target_include_directories(primordial_ros2s_ref PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(primordial_ros2s_ref PRIVATE Threads::Threads)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(primordial_ros2s PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -ffp-contract=off>)
    target_compile_options(primordial_ros2s_ref PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -ffp-contract=off>)
endif()

if(PRIMORDIAL_ROS2S_ENABLE_CUDA AND CMAKE_CUDA_COMPILER_ID MATCHES "NVIDIA")
    target_compile_options(primordial_ros2s PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    if(PRIMORDIAL_ROS2S_CUDA_FMA)
        target_compile_options(primordial_ros2s PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--fmad=true>)
    else()
        target_compile_options(primordial_ros2s PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
    endif()
    if(NOT "${PRIMORDIAL_ROS2S_CUDA_MAXREGCOUNT}" STREQUAL "")
        target_compile_options(primordial_ros2s PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--maxrregcount=${PRIMORDIAL_ROS2S_CUDA_MAXREGCOUNT}>)
    endif()
endif()
