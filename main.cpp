#define __NO_STD_VECTOR // Use cl::vector instead of STL version
#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include "openCLUtilities.hpp"
#include <iostream>
#include <string>
#include <fstream>
#include <utility>

#define MU = 0.05
#define ITERATIONS 500
#define FILENAME test.raw

using namespace cl;


int main(void) {
   try { 
        // Get available platforms
        vector<Platform> platforms;
        Platform::get(&platforms);

        // Select the default platform and create a context using this platform and the GPU
        cl_context_properties cps[] = { 
            CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 
            0 
        };
		Context context = Context( CL_DEVICE_TYPE_GPU, cps);

        // Get a list of devices on this platform
		vector<Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

        // Create a command queue and use the first device
        CommandQueue queue = CommandQueue(context, devices[0]);

        // Read source file
        std::ifstream sourceFile("kernels.cl");
        if(sourceFile.fail()) {
            std::cout << "Failed to open OpenCL source file" << std::endl;
            exit(-1);
        }
        std::string sourceCode(
            std::istreambuf_iterator<char>(sourceFile),
            (std::istreambuf_iterator<char>()));
        Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));

        // Make program of the source code in the context
        Program program = Program(context, source);
    
        // Build program for these specific devices
        try{
            program.build(devices);
        } catch(Error error) {
            if(error.err() == CL_BUILD_PROGRAM_FAILURE) {
                std::cout << "Build log:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
            }   
            throw error;
        } 

        // Create Kernels
        Kernel InitKernel = Kernel(program, "GVFInit");
        Kernel IterationKernel = Kernel(program, "GVFIteration");

        // Load volume to GPU

        // Run initialization kernel

        // Run iterations
        for(int i = 0; i < ITERATIONS; i++) {
        }

        // Read the result in some way

    } catch(Error error) {
       std::cout << error.what() << "(" << error.err() << ")" << std::endl;
       std::cout << getCLErrorString(error.err()) << std::endl;
    }

   return 0;
}
