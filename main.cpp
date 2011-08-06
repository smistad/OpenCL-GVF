#define __NO_STD_VECTOR // Use cl::vector instead of STL version
#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include "openCLUtilities.hpp"
#include <string>
#include <iostream>
#include <utility>
#include <math.h>


using namespace cl;
typedef unsigned char uchar;

float * parseRawFile(char * filename, int SIZE_X, int SIZE_Y, int SIZE_Z) {
    // Parse the specified raw file
    int rawDataSize = SIZE_X*SIZE_Y*SIZE_Z;

    uchar * rawVoxels = new uchar[rawDataSize];
    FILE * file = fopen(filename, "rb");
    if(file == NULL) {
        printf("File not found: %s\n", filename);
        exit(-1);
    }

    fread(rawVoxels, sizeof(uchar), rawDataSize, file);

    // Find min and max
    int min = 257;
    int max = 0;
    for(int i = 0; i < rawDataSize; i++) {
        if(rawVoxels[i] > max)
            max = rawVoxels[i];

        if(rawVoxels[i] < min)
            min = rawVoxels[i];

    }

    std::cout << "Min: " << min << " Max: " << max << std::endl;

    // Normalize result
    float * voxels = new float[rawDataSize];
    for(int i = 0; i < rawDataSize; i++) {
        voxels[i] = (float)(rawVoxels[i] - min) / (max - min);
    }
    delete[] rawVoxels;

    return voxels;
} 

void writeToRaw(float * voxels, char * filename, int SIZE_X, int SIZE_Y, int SIZE_Z) {
    FILE * file = fopen(filename, "wb");
    fwrite(voxels, sizeof(float), SIZE_X*SIZE_Y*SIZE_Z, file);
}

int main(int argc, char ** argv) {
    char * filename;
    int SIZE_X, SIZE_Y, SIZE_Z, ITERATIONS;
    float mu;
    if(argc > 5) {
        filename = argv[1];
        SIZE_X = atoi(argv[2]);
        SIZE_Y = atoi(argv[3]);
        SIZE_Z = atoi(argv[4]);
        mu = atof(argv[5]);
        if(argc == 7) {
            ITERATIONS = atoi(argv[6]);
        } else {
            ITERATIONS = (int)sqrt(SIZE_X*SIZE_Y*SIZE_Z);
        }
    } else {
        std::cout << "usage: filename of raw file size_x size_y size_z mu [iterations]" << std::endl;
        exit(-1);
    }

   try { 
		Context context = createCLContext(CL_DEVICE_TYPE_GPU);

        // Get a list of devices
		vector<Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

        // Create a command queue and use the first device
        CommandQueue queue = CommandQueue(context, devices[0]);

        Program program = buildProgramFromSource(context, "kernels.cl");

        // Create Kernels
        Kernel initKernel = Kernel(program, "GVFInit");
        Kernel iterationKernel = Kernel(program, "GVFIteration");
        Kernel resultKernel = Kernel(program, "GVFResult");

        // Load volume to GPU
        std::cout << "Reading RAW file " << filename << std::endl;
        float * voxels = parseRawFile(filename, SIZE_X, SIZE_Y, SIZE_Z);
        Image3D volume = Image3D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ImageFormat(CL_R, CL_FLOAT), SIZE_X, SIZE_Y, SIZE_Z, 0, 0, voxels);
        delete[] voxels;

        // Query the size of available memory
        unsigned int memorySize = devices[0].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();

        std::cout << "Available memory on selected device " << memorySize << " bytes "<< std::endl;

        ImageFormat storageFormat;
        if(memorySize > SIZE_X*SIZE_Y*SIZE_Z*4*4*3) {
            storageFormat = ImageFormat(CL_RGBA, CL_FLOAT);
            std::cout << "Using 32 bits floats texture storage" << std::endl;
        } else if(memorySize > SIZE_X*SIZE_Y*SIZE_Z*2*4*3) {
            storageFormat = ImageFormat(CL_RGBA, CL_SNORM_INT16);
            std::cout << "Not enough memory on device for 32 bit floats, using 16bit for texture storage instead." << std::endl;
        } else {
            std::cout << "There is not enough memory on this device to calculate the GVF for this dataset!" << std::endl;
            exit(-1);
        }

        // Run initialization kernel
        Image3D initVectorField = Image3D(context, CL_MEM_READ_WRITE, storageFormat, SIZE_X, SIZE_Y, SIZE_Z);
        initKernel.setArg(0, volume);
        initKernel.setArg(1, initVectorField);

        queue.enqueueNDRangeKernel(
                initKernel,
                NullRange,
                NDRange(SIZE_X,SIZE_Y,SIZE_Z),
                NullRange
        );

        // Delete volume from device
        //volume.~Image3D();

        // copy vector field and create double buffer
        Image3D vectorField = Image3D(context, CL_MEM_READ_WRITE, storageFormat, SIZE_X, SIZE_Y, SIZE_Z);
        Image3D vectorField2 = Image3D(context, CL_MEM_READ_WRITE, storageFormat, SIZE_X, SIZE_Y, SIZE_Z);
        cl::size_t<3> offset;
        offset[0] = 0;
        offset[1] = 0;
        offset[2] = 0;
        cl::size_t<3> region;
        region[0] = SIZE_X;
        region[1] = SIZE_Y;
        region[2] = SIZE_Z;
        queue.enqueueCopyImage(initVectorField, vectorField, offset, offset, region);

        //queue.enqueueCopyImage(initVectorField, vectorField2, offset, offset, region);
        queue.finish();
        std::cout << "Running iterations... ( " << ITERATIONS << " )" << std::endl; 
        // Run iterations
        iterationKernel.setArg(0, initVectorField);
        iterationKernel.setArg(3, mu);

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
                    NDRange(1,1,1)
            );
        }
        queue.finish();

        // Read the result in some way (maybe write to a seperate raw file)
        volume = Image3D(context, CL_MEM_WRITE_ONLY, ImageFormat(CL_R,CL_FLOAT), SIZE_X, SIZE_Y, SIZE_Z);
        resultKernel.setArg(0, volume);
        resultKernel.setArg(1, vectorField);
        queue.enqueueNDRangeKernel(
                resultKernel,
                NullRange,
                NDRange(SIZE_X, SIZE_Y, SIZE_Z),
                NullRange
        );
        queue.finish();
        voxels = new float[SIZE_X*SIZE_Y*SIZE_Z];
        std::cout << "Reading vector field from device..." << std::endl;
        queue.enqueueReadImage(volume, CL_TRUE, offset, region, 0, 0, voxels);
        std::cout << "Writing vector field to RAW file..." << std::endl;
        writeToRaw(voxels, "result.raw", SIZE_X, SIZE_Y, SIZE_Z);
        delete[] voxels;

    } catch(Error error) {
       std::cout << error.what() << "(" << error.err() << ")" << std::endl;
       std::cout << getCLErrorString(error.err()) << std::endl;
    }

   return 0;
}
