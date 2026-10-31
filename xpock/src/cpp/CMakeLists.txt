cmake_minimum_required(VERSION 3.10)
project(xpock_cpp_kernels LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")

add_library(xpock_kernels STATIC wave_kernels.cpp)

target_include_directories(xpock_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Optional shared library target
add_library(xpock_kernels_shared SHARED wave_kernels.cpp)
target_include_directories(xpock_kernels_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
