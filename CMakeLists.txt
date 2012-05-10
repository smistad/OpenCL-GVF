cmake_minimum_required(VERSION 2.8)
project(GPU_GVF)

find_package(ITK REQUIRED)
message("using ${ITK_USE_FILE}")
include(${ITK_USE_FILE})

add_subdirectory(OpenCLUtilities)
include_directories(${OPENCL_INCLUDE_DIRS})

add_executable(host main.cpp)

target_link_libraries(host OCL-Utilities ${ITK_LIBRARIES} )
