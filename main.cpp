#include <time.h>
#include "OpenCLUtilities/openCLUtilities.hpp"
#include <string>
#include <iostream>
#include <utility>
#include <math.h>
#include <limits.h>
#include "SIPL/Core.hpp"
using namespace cl;

#ifndef KERNELS_DIR
#define KERNELS_DIR ""
#endif

SIPL::float2 * run2DKernels(Context context, CommandQueue queue, float * voxels, int SIZE_X, int SIZE_Y, float mu, int ITERATIONS, int datatype) {
    ImageFormat storageFormat;
    if(datatype == sizeof(short)) {
        std::cout << "Using 16 bit floats" << std::endl;
        storageFormat = ImageFormat(CL_RG, CL_SNORM_INT16);
    } else {
        std::cout << "Using 32 bit floats" << std::endl;
        storageFormat = ImageFormat(CL_RG, CL_FLOAT);
    }

    std::string filaname = std::string(KERNELS_DIR) + std::string("2Dkernels.cl");
    Program program = buildProgramFromSource(context, filaname);

    // Create kernels
    Kernel initKernel = Kernel(program, "GVF2DInit");
    Kernel iterationKernel = Kernel(program, "GVF2DIteration");
    Kernel resultKernel = Kernel(program, "GVF2DResult");

    // Load volume to device
    Image2D volume = Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ImageFormat(CL_R, CL_FLOAT), SIZE_X, SIZE_Y, 0, voxels);
    delete[] voxels;

    Image2D initVectorField = Image2D(context, CL_MEM_READ_WRITE, storageFormat, SIZE_X, SIZE_Y);

    // Run initialization kernel
    initKernel.setArg(0, volume);
    initKernel.setArg(1, initVectorField);

    queue.enqueueNDRangeKernel(
            initKernel,
            NullRange,
            NDRange(SIZE_X,SIZE_Y),
            NullRange
    );

    // copy vector field and create double buffer
    Image2D vectorField = Image2D(context, CL_MEM_READ_WRITE,storageFormat, SIZE_X, SIZE_Y);
    Image2D vectorField2 = Image2D(context, CL_MEM_READ_WRITE,storageFormat, SIZE_X, SIZE_Y);
    cl::size_t<3> offset;
    offset[0] = 0;
    offset[1] = 0;
    offset[2] = 0;
    cl::size_t<3> region;
    region[0] = SIZE_X;
    region[1] = SIZE_Y;
    region[2] = 1;
    queue.enqueueCopyImage(initVectorField, vectorField, offset, offset, region);
    queue.finish();

    iterationKernel.setArg(0, initVectorField);
    iterationKernel.setArg(3, mu);

    // Find a SIZE_X and SIZE_Y that is divisable by 14
    int rangeX = SIZE_X;
    int rangeY = SIZE_Y;

    Event startEvent;
    Event event;

    for(int i = 0; i < ITERATIONS; i++) {
        if(i % 2 == 0) {
            iterationKernel.setArg(1, vectorField);
            iterationKernel.setArg(2, vectorField2);
        } else {
            iterationKernel.setArg(1, vectorField2);
            iterationKernel.setArg(2, vectorField);
        }
        if(i == 0) {
            // Profile first iteration
            queue.enqueueNDRangeKernel(
                    iterationKernel,
                    NullRange,
                    NDRange(rangeX, rangeY),
                    NDRange(16,16),
                    NULL,
                    &startEvent
            );
        } else if(i == ITERATIONS-1) {
            // Profile last iteration
            queue.enqueueNDRangeKernel(
                    iterationKernel,
                    NullRange,
                    NDRange(rangeX, rangeY),
                    NDRange(16,16),
                    NULL,
                    &event
            );
        } else {
            queue.enqueueNDRangeKernel(
                    iterationKernel,
                    NullRange,
                    NDRange(rangeX, rangeY),
                    NDRange(16,16)
            );
        }
    }
    queue.finish();

    cl_ulong start, end;
    event.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &start);
    event.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_END, &end);
    std::cout << "One iteration processed in: " << (end-start)* 1.0e-6 << " ms " << std::endl;
    startEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &start);
    std::cout << "All iterations processed in: " << (end-start)* 1.0e-6 << " ms " << std::endl;

    // Read result back to host
    SIPL::float2 * vector2 = new SIPL::float2[SIZE_X*SIZE_Y];
    if(datatype == sizeof(short)) {
        Image2D vectorFieldFinal = Image2D(context, CL_MEM_WRITE_ONLY, ImageFormat(CL_RG,CL_FLOAT), SIZE_X, SIZE_Y);
        resultKernel.setArg(0, vectorField);
        resultKernel.setArg(1, vectorFieldFinal);
        queue.enqueueNDRangeKernel(
                resultKernel,
                NullRange,
                NDRange(SIZE_X, SIZE_Y),
                NullRange
        );
        queue.finish();

        queue.enqueueReadImage(vectorFieldFinal, CL_TRUE, offset, region, 0, 0, vector2);
    } else {
        queue.enqueueReadImage(vectorField, CL_TRUE, offset, region, 0, 0, vector2);
    }
    
    return vector2;
}

SIPL::float3 * run3DKernels(Context context, CommandQueue queue, float * voxels, int SIZE_X, int SIZE_Y, int SIZE_Z, float mu, int ITERATIONS, int datatype) {
    ImageFormat storageFormat;
    ImageFormat storageFormat2;
    if(datatype == sizeof(short)) {
        std::cout << "Using 16 bit floats" << std::endl;
        storageFormat = ImageFormat(CL_RGBA, CL_SNORM_INT16);
        storageFormat2 = ImageFormat(CL_RG, CL_SNORM_INT16);
    } else {
        std::cout << "Using 32 bit floats" << std::endl;
        storageFormat = ImageFormat(CL_RGBA, CL_FLOAT);
        storageFormat2 = ImageFormat(CL_RG, CL_FLOAT);
    }
    std::string filaname = std::string(KERNELS_DIR) + std::string("3Dkernels.cl");
    Program program = buildProgramFromSource(context, filaname);

    // Create Kernels
    Kernel initKernel = Kernel(program, "GVF3DInit");
    Kernel iterationKernel = Kernel(program, "GVF3DIteration");
    Kernel resultKernel = Kernel(program, "GVF3DResult");

    // Load volume to GPU
    Image3D volume = Image3D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ImageFormat(CL_R, CL_FLOAT), SIZE_X, SIZE_Y, SIZE_Z, 0, 0, voxels);
    delete[] voxels;

    // Run initialization kernel
    cl::size_t<3> offset;
    offset[0] = 0;
    offset[1] = 0;
    offset[2] = 0;
    cl::size_t<3> region;
    region[0] = SIZE_X;
    region[1] = SIZE_Y;
    region[2] = SIZE_Z;


    Image3D vectorField = Image3D(context, CL_MEM_READ_WRITE, storageFormat, SIZE_X, SIZE_Y, SIZE_Z);
    Image3D initVectorField = Image3D(context, CL_MEM_READ_WRITE, storageFormat2, SIZE_X, SIZE_Y, SIZE_Z);
    initKernel.setArg(0, volume);
    initKernel.setArg(1, initVectorField);
    initKernel.setArg(2, vectorField);

    queue.enqueueNDRangeKernel(
            initKernel,
            NullRange,
            NDRange(SIZE_X,SIZE_Y,SIZE_Z),
            NullRange
    );

    // copy vector field and create double buffer
    Image3D vectorField2 = Image3D(context, CL_MEM_READ_WRITE, storageFormat, SIZE_X, SIZE_Y, SIZE_Z);

    std::cout << "Running iterations... ( " << ITERATIONS << " )" << std::endl; 
    // Run iterations
    iterationKernel.setArg(0, initVectorField);
    iterationKernel.setArg(3, mu);

    int rangeX = SIZE_X;
    int rangeY = SIZE_Y;
    int rangeZ = SIZE_Z;

    Event event,startEvent;
    for(int i = 0; i < ITERATIONS; i++) {
        if(i % 2 == 0) {
            iterationKernel.setArg(1, vectorField);
            iterationKernel.setArg(2, vectorField2);
        } else {
            iterationKernel.setArg(1, vectorField2);
            iterationKernel.setArg(2, vectorField);
        }
        if(i == 0) {
            // Profile first iteration
            queue.enqueueNDRangeKernel(
                    iterationKernel,
                    NullRange,
                    NDRange(rangeX,rangeY,rangeZ),
                    NDRange(4,4,4),
                    NULL,
                    &startEvent
            );
        } else if(i == ITERATIONS-1) {
            // Profile last iteration
            queue.enqueueNDRangeKernel(
                    iterationKernel,
                    NullRange,
                    NDRange(rangeX,rangeY,rangeZ),
                    NDRange(4,4,4),
                    NULL,
                    &event
            );
        } else {
            queue.enqueueNDRangeKernel(
                    iterationKernel,
                    NullRange,
                    NDRange(rangeX,rangeY,rangeZ),
                    NDRange(4,4,4)
            );
        }
    }
    queue.finish();

    // Do some profiling
    cl_ulong start, end;
    event.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &start);
    event.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_END, &end);
    std::cout << "One iteration processed in: " << (end-start)* 1.0e-6 << " ms " << std::endl;
    startEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &start);
    std::cout << "All iterations processed in: " << (end-start)* 1.0e-6 << " ms " << std::endl;

    // Read result back to host
    float * vector2 = new float[SIZE_X*SIZE_Y*SIZE_Z*4];
    if(datatype == sizeof(short)) {
        Image3D vectorFieldFinal = Image3D(context, CL_MEM_WRITE_ONLY, ImageFormat(CL_RGBA,CL_FLOAT), SIZE_X, SIZE_Y, SIZE_Z);
        resultKernel.setArg(0, vectorField);
        resultKernel.setArg(1, vectorFieldFinal);
        queue.enqueueNDRangeKernel(
                resultKernel,
                NullRange,
                NDRange(SIZE_X, SIZE_Y),
                NullRange
        );
        queue.finish();

        queue.enqueueReadImage(vectorFieldFinal, CL_TRUE, offset, region, 0, 0, vector2);
    } else {
        queue.enqueueReadImage(vectorField, CL_TRUE, offset, region, 0, 0, vector2);
    }

    SIPL::float3 * vectorFieldResult = new SIPL::float3[SIZE_X*SIZE_Y*SIZE_Z];
    for(int i = 0; i < SIZE_X*SIZE_Y*SIZE_Z; ++i) {
        vectorFieldResult[i].x = vector2[i*4];
        vectorFieldResult[i].y = vector2[i*4+1];
        vectorFieldResult[i].z = vector2[i*4+2];
    }
    delete[] vector2;

    return vectorFieldResult;
}

SIPL::float3 * run3DKernelsWithoutTexture(Context context, CommandQueue queue, float * voxels, int SIZE_X, int SIZE_Y, int SIZE_Z, float mu, int ITERATIONS, int datatype) {
    
    std::string filaname = std::string(KERNELS_DIR) + std::string("3DkernelsNO_WRITE_TEX.cl");
    Program program = buildProgramFromSource(context, filaname);

    // Create Kernels
    Kernel initKernel = Kernel(program, "GVF3DInit");
    Kernel iterationKernel = Kernel(program, "GVF3DIteration");
    Kernel resultKernel = Kernel(program, "GVF3DResult");

    // Load volume to GPU
    Image3D volume = Image3D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ImageFormat(CL_R, CL_FLOAT), SIZE_X, SIZE_Y, SIZE_Z, 0, 0, voxels);
    delete[] voxels;

    // Run initialization kernel
    cl::size_t<3> offset;
    offset[0] = 0;
    offset[1] = 0;
    offset[2] = 0;
    cl::size_t<3> region;
    region[0] = SIZE_X;
    region[1] = SIZE_Y;
    region[2] = SIZE_Z;
    
    Buffer initVectorField = Buffer(context, CL_MEM_READ_WRITE, 4*sizeof(float)*SIZE_X*SIZE_Y*SIZE_Z);
    Buffer vectorField = Buffer(context, CL_MEM_READ_WRITE, 3*sizeof(float)*SIZE_X*SIZE_Y*SIZE_Z);

    initKernel.setArg(0, volume);
    initKernel.setArg(1, initVectorField);
    initKernel.setArg(2, vectorField);

    queue.enqueueNDRangeKernel(
            initKernel,
            NullRange,
            NDRange(SIZE_X,SIZE_Y,SIZE_Z),
            NullRange
    );

    ImageFormat storageFormat = ImageFormat(CL_RGBA, CL_FLOAT);
    Image3D initVectorFieldImage = Image3D(context, CL_MEM_READ_WRITE, storageFormat, SIZE_X, SIZE_Y, SIZE_Z);
    queue.enqueueCopyBufferToImage(initVectorField, initVectorFieldImage, 0, offset, region);

    // Copy init vector field buffer to vectorField buffer and 3D image 
    Buffer vectorField2 = Buffer(context, CL_MEM_READ_WRITE, 3*sizeof(float)*SIZE_X*SIZE_Y*SIZE_Z);

    queue.finish();

    std::cout << "Running iterations... ( " << ITERATIONS << " )" << std::endl; 
    // Run iterations
    iterationKernel.setArg(0, initVectorFieldImage);
    iterationKernel.setArg(3, mu);

    int rangeX = SIZE_X;
    int rangeY = SIZE_Y;
    int rangeZ = SIZE_Z;

    Event event, startEvent;
    for(int i = 0; i < ITERATIONS; i++) {
        if(i % 2 == 0) {
            iterationKernel.setArg(1, vectorField);
            iterationKernel.setArg(2, vectorField2);
        } else {
            iterationKernel.setArg(1, vectorField2);
            iterationKernel.setArg(2, vectorField);
        }
        if(i == 0) {
            // Profile first iteration
            queue.enqueueNDRangeKernel(
                    iterationKernel,
                    NullRange,
                    NDRange(rangeX,rangeY,rangeZ),
                    NDRange(4,4,4),
                    NULL,
                    &startEvent
            );
        } else if(i == ITERATIONS-1) {
            // Profile last iteration
            queue.enqueueNDRangeKernel(
                    iterationKernel,
                    NullRange,
                    NDRange(rangeX,rangeY,rangeZ),
                    NDRange(4,4,4),
                    NULL,
                    &event
            );
        } else {
            queue.enqueueNDRangeKernel(
                    iterationKernel,
                    NullRange,
                    NDRange(rangeX,rangeY,rangeZ),
                    NDRange(4,4,4)
            );
        }
    }
    queue.finish();
    
    // Do some profiling
    cl_ulong start, end;
    event.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &start);
    event.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_END, &end);
    std::cout << "One iteration processed in: " << (end-start)* 1.0e-6 << " ms " << std::endl;
    startEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &start);
    std::cout << "All iterations processed in: " << (end-start)* 1.0e-6 << " ms " << std::endl;

    float * vector = new float[SIZE_X*SIZE_Y*SIZE_Z*4];
    queue.enqueueReadBuffer(vectorField, CL_TRUE, 0, sizeof(SIPL::float3)*SIZE_X*SIZE_Y*SIZE_Z, vector);
    SIPL::float3 * vectorFieldResult = new SIPL::float3[SIZE_X*SIZE_Y*SIZE_Z];
    for(int i = 0; i < SIZE_X*SIZE_Y*SIZE_Z; ++i) {
        vectorFieldResult[i].x = vector[i*4];
        vectorFieldResult[i].y = vector[i*4+1];
        vectorFieldResult[i].z = vector[i*4+2];
    }
    delete[] vector;
    return vectorFieldResult;
}

int main(int argc, char ** argv) {
    char * filename;
    int SIZE_X, SIZE_Y, SIZE_Z, ITERATIONS, bytes;
    float mu;
    Context context;
    CommandQueue queue;

   try { 
    Context context = createCLContextFromArguments(argc, argv);

    // Get a list of devices
    VECTOR_CLASS<Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    std::cout << "Using device: " << devices[0].getInfo<CL_DEVICE_NAME>() << std::endl;

    // Create a command queue and use the first device
    CommandQueue queue = CommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);


    // Query the size of available memory
    unsigned int memorySize = devices[0].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();

    std::cout << "Available memory on selected device " << (double)memorySize/(1024*1024) << " MB "<< std::endl;

    int newArgc = argc;
    for(int i = 0; i < argc; i++) {
        if(argv[i][0] == '-' && argv[i][1] == '-') {
            newArgc -= 2;
        }
    }
    argc = newArgc;
    int bytes = 4;
    if(strcmp(argv[argc-1], "-16bit") == 0) {
        bytes = 2;
        argc--;
    }

    if(argc == 4) {
        filename = argv[1];
        mu = atof(argv[2]);
        ITERATIONS = atoi(argv[3]);
        std::string strFilename = filename;
        if(strFilename.substr(strFilename.size()-3) == "mhd") {
            // Is 3D image (mhd file)
            SIPL::Volume<float> * volume = new SIPL::Volume<float>(filename, SIPL::IntensityTransformation(SIPL::NORMALIZED));
            SIZE_X = volume->getWidth();
            SIZE_Y = volume->getHeight();
            SIZE_Z = volume->getDepth();
            float * voxels = (float *)volume->getData();
            SIPL::Volume<SIPL::float3> * GVF = new SIPL::Volume<SIPL::float3>(volume->getWidth(), volume->getHeight(), volume->getDepth());
            SIPL::float3 * output;

            if((int)devices[0].getInfo<CL_DEVICE_EXTENSIONS>().find("cl_khr_3d_image_writes") > -1) {
                output = run3DKernels(context, queue, voxels, SIZE_X, SIZE_Y, SIZE_Z, mu, ITERATIONS, bytes);
            } else {
                std::cout << "This device doesn't support writing to 3D textures. Using buffers instead. 16bit storage format is also disabled." << std::endl;
                output = run3DKernelsWithoutTexture(context, queue, voxels, SIZE_X, SIZE_Y, SIZE_Z, mu, ITERATIONS, bytes);
            }
            GVF->setData(output);
            GVF->display(0, 0.1);
        }else{
            // Is 2D image
            std::cout << "Reading image file " << filename << std::endl;
            // TODO: need to normalized this image!! Current code will not work
            SIPL::Image<float> * image = new SIPL::Image<float>(filename);
            float * pixels = (float *)image->getData();
            SIPL::Image<SIPL::float2> * GVF = new SIPL::Image<SIPL::float2>(image->getWidth(), image->getHeight());
            SIPL::float2 * output = run2DKernels(context, queue, pixels, image->getWidth(), image->getHeight(), mu, ITERATIONS, bytes);
            GVF->setData(output);
            GVF->display(0, 0.1);
        }
    } else {
        std::cout << "Usage:" << std::endl << "---------------------------------" << std::endl <<
            "For 2D images:" << std::endl << 
            "./host filename.jpg mu #iterations [-16bit] [--device cpu/gpu]" << std::endl <<
            "For 3D images:" << std::endl <<
            "./host filename.mhd mu #iterations [-16bit] [--device cpu/gpu] " << std::endl;
        exit(-1);
    }
        
    } catch(Error error) {
       std::cout << error.what() << "(" << error.err() << ")" << std::endl;
       std::cout << getCLErrorString(error.err()) << std::endl;
    }

   return 0;
}
