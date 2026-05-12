cmake_minimum_required(VERSION 3.14)

project(primordial_ros2s LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(primordial_ros2s main.cpp)
target_include_directories(primordial_ros2s PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(primordial_ros2s PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
endif()
