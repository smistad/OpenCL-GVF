cmake_minimum_required(VERSION 2.8)
project(GPU_GVF)

add_subdirectory(SIPL)
include_directories(${SIPL_INCLUDE_DIRS})
add_subdirectory(OpenCLUtilities)
include_directories(${OPENCL_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++0x -O3 -DKERNELS_DIR=\\\"${PROJECT_SOURCE_DIR}/\\\"")

add_executable(GVF main.cpp)
target_link_libraries(GVF OCL-Utilities SIPL )
