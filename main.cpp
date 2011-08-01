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
#define SIZE_X 256
#define SIZE_Y 256
#define SIZE_Z 256

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
        Kernel initKernel = Kernel(program, "GVFInit");
        Kernel iterationKernel = Kernel(program, "GVFIteration");

        // Load volume to GPU
        Image3D volume = Image3D(context, CL_MEM_READ, ImageFormat(CL_R, CL_FLOAT), SIZE_X, SIZE_Y, SIZE_Z);

        // Run initialization kernel
        Image3D initVectorField = Image3D(context, CL_MEM_READ_WRITE, ImageFormat(CL_RGBA, CLFLOAT), SIZE_X, SIZE_Y, SIZE_Z);
        initKernel.setArg(0, volume);
        initKernel.setArg(1, initVectorField);

        queue.enqueueNDRangeKernel(
                initKernel,
                NullRange,
                NDRange(SIZE_X,SIZE_Y,SIZE_Z),
                Nullrange
        );

        // Delete volume from device
        volume.release();

        // copy vector field and create double buffer
        Image3D vectorField = Image3D(context, CL_MEM_READ_WRITE, ImageFormat(CL_RGBA, CLFLOAT), SIZE_X, SIZE_Y, SIZE_Z);
        Image3D vectorField2 = Image3D(context, CL_MEM_READ_WRITE, ImageFormat(CL_RGBA, CLFLOAT), SIZE_X, SIZE_Y, SIZE_Z);
        queue.enqueueCopyImage(initVectorField, vectorField);

        // Run iterations
        iterationKernel.setArg(0, initVectorField);
        iterationKernel.setArg(3, MU);
        for(int i = 0; i < ITERATIONS; i++) {
            if(i % 2 == 0) {
                iterationKernel.setArg(1, vectorField);
                iterationKernel.setArg(2, vectorField2);
            } else {
                iterationKernel.setArg(1, vectorField2);
                iterationKernel.setArg(2, vectorField);
            }
            queue.enqueueNDRangeKernel(
                    iterationKernel,
                    NullRange,
                    NDRange(SIZE_X,SIZE_Y,SIZE_Z),
                    NullRange
            );
        }

        // Read the result in some way

    } catch(Error error) {
       std::cout << error.what() << "(" << error.err() << ")" << std::endl;
       std::cout << getCLErrorString(error.err()) << std::endl;
    }

   return 0;
}
