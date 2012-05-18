cmake_minimum_required(VERSION 2.8)
project(GPU_GVF)

add_subdirectory(SIPL)
include_directories(${SIPL_INCLUDE_DIRS})
add_subdirectory(OpenCLUtilities)
include_directories(${OPENCL_INCLUDE_DIRS})

add_executable(host main.cpp)
target_link_libraries(host OCL-Utilities SIPL )
