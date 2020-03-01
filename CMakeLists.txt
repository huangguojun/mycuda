cmake_minimum_required(VERSION 2.8)
message(STATUS "CMake version: ${CMAKE_VERSION}")

project(diana)

file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cu")

enable_language("CUDA")
add_executable(diana ${cpu_source_files} ${gpu_source_files})

add_definitions(-DUSE_CUDA)
