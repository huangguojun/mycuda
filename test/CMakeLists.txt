cmake_minimum_required(VERSION 2.8)
message(STATUS "CMake version: ${CMAKE_VERSION}")

project(diana)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})


set(cpu_source_files src/main.cpp)

set(gpu_source_files src/add_loop_gpu.cu
                     src/gpu.cu
                     src/julia_gpu.cu )

enable_language("CUDA")

add_executable(diana ${cpu_source_files} ${gpu_source_files} )

target_link_libraries(diana ${OpenCV_LIBS})

add_definitions(-DUSE_CUDA)
