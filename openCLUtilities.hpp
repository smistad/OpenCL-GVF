#ifndef OPENCL_UTILITIES_H
#define OPENCL_UTILITIES_H

#define __NO_STD_VECTOR // Use cl::vector instead of STL version
#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include <string>
#include <iostream>
#include <fstream>

#ifdef GL_SHARING_EXTENSION
    #if defined __APPLE__ || defined(MACOSX)
    #else
        #if defined WIN32
        #else
            #include <GL/glx.h>
        #endif
    #endif
#endif

enum cl_vendor {
    VENDOR_ALL,
    VENDOR_NVIDIA,
    VENDOR_AMD,
    VENDOR_INTEL
};

cl::Context createCLContext(cl_device_type type, cl_vendor vendor = VENDOR_ALL, bool GLInterop = false);

cl::Program buildProgramFromSource(cl::Context context, std::string filename);

char *getCLErrorString(cl_int err);

#endif
