cmake_minimum_required(VERSION 2.8)
project(GPU_GVF)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

find_package(OpenCL REQUIRED)
message("using ${OPENCL_INCLUDE_DIR}")
include_directories(${OPENCL_INCLUDE_DIR})

find_package(ITK REQUIRED)
message("using ${ITK_USE_FILE}")
include(${ITK_USE_FILE})

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

set(ItkVtkGlue_DIR ${PROJECT_SOURCE_DIR}/ItkVtkGlue)

find_package(ItkVtkGlue REQUIRED)
include(${ItkVtkGlue_USE_FILE})

add_executable(host main.cpp openCLUtilities.cpp)

target_link_libraries(host ItkVtkGlue ${OPENCL_LIBRARIES} ${ITK_LIBRARIES} ${VTK_LIBRARIES})

