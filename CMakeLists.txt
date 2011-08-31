cmake_minimum_required(VERSION 2.8)
project(GPU_GVF)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

find_package(OpenCL REQUIRED)
message("using ${OPENCL_INCLUDE_DIR}")
include_directories(${OPENCL_INCLUDE_DIR})

find_package(ITK REQUIRED)
message("using ${ITK_USE_FILE}")
include(${ITK_USE_FILE})

add_executable(host main.cpp openCLUtilities.cpp)

target_link_libraries(host ${OPENCL_LIBRARIES} ${ITK_LIBRARIES} )

