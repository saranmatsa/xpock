cmake_minimum_required(VERSION 3.10)
project(xpock_fortran_wave_ops LANGUAGES Fortran)

set(CMAKE_Fortran_STANDARD 90)
set(CMAKE_Fortran_STANDARD_REQUIRED ON)
set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -O3 -Wall")

add_library(xpock_fortran STATIC wave_ops.f90)
target_include_directories(xpock_fortran PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(xpock_fortran_shared SHARED wave_ops.f90)
