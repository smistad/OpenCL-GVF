#define __NO_STD_VECTOR // Use cl::vector instead of STL version
#define __CL_ENABLE_EXCEPTIONS

#include <time.h>
#include <CL/cl.hpp>
#include "OpenCLUtilities/openCLUtilities.hpp"
#include <string>
#include <iostream>
#include <utility>
#include <math.h>
#include <limits.h>
#include <itkImage.h>
#include <itkRawImageIO.h>
#include <itkImageFileReader.h>
#include <itkImageRegionIterator.h>
#include <itkImageFileWriter.h>
#include <itkCastImageFilter.h>
#include <itkRescaleIntensityImageFilter.h>
using namespace cl;
typedef unsigned char uchar;




/**
 * Reads a RAW file and normalizes it
 */
float * parseRawFile(char * filename, int SIZE_X, int SIZE_Y, int SIZE_Z) {
    itk::RawImageIO<uchar, 3>::Pointer io;
    io = itk::RawImageIO<uchar, 3>::New();
    io->SetFileName(filename);
    io->SetDimensions(0,SIZE_X);
    io->SetDimensions(1,SIZE_Y);
    io->SetDimensions(2,SIZE_Z);
    io->SetHeaderSize(0);

    typedef itk::Image<uchar, 3> ImageType;
    itk::ImageFileReader<ImageType>::Pointer reader;
    reader = itk::ImageFileReader<ImageType>::New();
    reader->SetFileName(filename);
    reader->SetImageIO(io);
    reader->Update();

    // Find min and max
    int min = INT_MAX;
    int max = 0;
    ImageType::Pointer im = reader->GetOutput();
    itk::ImageRegionIterator<ImageType> it(im, im->GetRequestedRegion() );
    for(it = it.Begin(); !it.IsAtEnd(); ++it) {
        if(it.Get() > max)
            max = it.Get();

        if(it.Get() < min)
            min = it.Get();

    }

    std::cout << "Min: " << min << " Max: " << max << std::endl;

    // Normalize result
    float * voxels = new float[SIZE_X*SIZE_Y*SIZE_Z];
    it = it.Begin();
    for(int i = 0; i < SIZE_X*SIZE_Y*SIZE_Z; i++) {
        voxels[i] = (float)(it.Get() - min) / (max - min);
        ++it;
    }

    return voxels;
} 

void writeToRaw(float * voxels, char * filename, int SIZE_X, int SIZE_Y, int SIZE_Z) {
    FILE * file = fopen(filename, "wb");
    fwrite(voxels, sizeof(float), SIZE_X*SIZE_Y*SIZE_Z, file);
}


/**
 * Read Image and normalize it
 */
float * parseImageFile(char * filename, int SIZE_X, int SIZE_Y) {
    typedef itk::Image<float, 2> ImageType;
    itk::ImageFileReader<ImageType>::Pointer reader;
    reader = itk::ImageFileReader<ImageType>::New();
    reader->SetFileName(filename);
    reader->Update();
    //
    // Find min and max
    int min = INT_MAX;
    int max = 0;
    ImageType::Pointer im = reader->GetOutput();
    itk::ImageRegionIterator<ImageType> it(im, im->GetRequestedRegion() );
    for(it = it.Begin(); !it.IsAtEnd(); ++it) {
        if(it.Get() > max)
            max = it.Get();

        if(it.Get() < min)
            min = it.Get();

    }

    std::cout << "Min: " << min << " Max: " << max << std::endl;

    // Normalize result
    float * voxels = new float[SIZE_X*SIZE_Y];
    it = it.Begin();
    for(int i = 0; i < SIZE_X*SIZE_Y; i++) {
        voxels[i] = (float)(it.Get() - min) / (max - min);
        ++it;
    }

    return voxels;

}

void writeImage(float * pixels, int SIZE_X, int SIZE_Y) {

    typedef itk::Image<float, 2> ImageType;
    ImageType::Pointer image = ImageType::New();
    ImageType::RegionType region;
    ImageType::SizeType size;
    size[0] = SIZE_X;
    size[1] = SIZE_Y;
    region.SetSize(size);
    image->SetRegions(region);
    image->Allocate();

    itk::ImageRegionIterator<ImageType> it(image, image->GetRequestedRegion() );
    int i = 0;
    for(it = it.Begin(); !it.IsAtEnd(); ++it) {
        it.Set(pixels[i]);
        i++;
    }

    typedef itk::Image<uchar,2> ScalarImageType;
    typedef itk::ImageFileWriter<ScalarImageType> WriterType;
    WriterType::Pointer writer = WriterType::New();

    ScalarImageType::Pointer scalarImage = ScalarImageType::New();
    typedef itk::RescaleIntensityImageFilter< ImageType, ScalarImageType> CastFilterType;
    CastFilterType::Pointer castFilter = CastFilterType::New();
    castFilter->SetInput(image);
    castFilter->SetOutputMaximum(255);
    castFilter->SetOutputMinimum(0);
    castFilter->Update();

    writer->SetFileName("output.jpg");
    writer->SetInput(castFilter->GetOutput());
    writer->Update();
}

/**
 * Use VTK to display image
 */
void displaySlice(float * voxels, int SIZE_X, int SIZE_Y, int SIZE_Z, int slice) {
    typedef itk::Image<float, 2> ImageType;
    ImageType::Pointer image = ImageType::New();
    ImageType::RegionType region;
    ImageType::SizeType size;
    size[0] = SIZE_X;
    size[1] = SIZE_Y;
    size[2] = SIZE_Z;
    region.SetSize(size);
    image->SetRegions(region);
    image->Allocate();

    itk::ImageRegionIterator<ImageType> it(image, image->GetRequestedRegion() );
    int i = 0;
    for(it = it.Begin(); !it.IsAtEnd(); ++it) {
        it.Set(voxels[i + SIZE_X*SIZE_Y*slice]);
        i++;
    }

    typedef itk::Image<uchar,2> ScalarImageType;
    typedef itk::ImageFileWriter<ScalarImageType> WriterType;
    WriterType::Pointer writer = WriterType::New();

    ScalarImageType::Pointer scalarImage = ScalarImageType::New();
    typedef itk::RescaleIntensityImageFilter< ImageType, ScalarImageType> CastFilterType;
    CastFilterType::Pointer castFilter = CastFilterType::New();
    castFilter->SetInput(image);
    castFilter->SetOutputMaximum(255);
    castFilter->SetOutputMinimum(0);
    castFilter->Update();

    writer->SetFileName("test.jpg");
    writer->SetInput(castFilter->GetOutput());
    writer->Update();

    /*
    typedef itk::ImageToVTKImageFilter<ScalarImageType> ConnectorType;
    ConnectorType::Pointer connector = ConnectorType::New();
    connector->SetInput(castFilter->GetOutput());
    QuickView viewer;
    viewer.AddImage(connector->GetOutput());
    viewer.Visualize();
    */
}

float relativeMagnitudeError(float * voxelsFloat, float * voxels, int size, int channels) {
    float error;
    if(channels == 2) {
        for(int i = 0; i < size; i += 2) {
            error += abs(sqrt(pow(voxelsFloat[i],2)+pow(voxelsFloat[i+1],2))
                    - sqrt(pow(voxels[i],2)+pow(voxels[i+1],2)));
        }
    } else if(channels == 4) {
        for(int i = 0; i < size; i += 4) {
            error += abs(sqrt(pow(voxelsFloat[i],2)+pow(voxelsFloat[i+1],2)+pow(voxelsFloat[i+2],2))
                    - sqrt(pow(voxels[i],2)+pow(voxels[i+1],2)+pow(voxels[i+2],2)));
        }
    }

    return error;
}

float relativeAngleError(float * voxelsFloat, float * voxels, int size, int channels) {
    float error;
    if(channels == 2) {
        for(int i = 0; i < size; i += 2) {
            float mag = sqrt(pow(voxelsFloat[i],2)+pow(voxelsFloat[i+1],2))
                *sqrt(pow(voxels[i],2)+pow(voxels[i+1],2));
            float dotProd = voxelsFloat[i]*voxels[i]+voxelsFloat[i+1]*voxels[i+1];
            error += acos(dotProd/mag);
        }
    } else if(channels == 4) {
        for(int i = 0; i < size; i += 4) {
            float mag = sqrt(pow(voxelsFloat[i],2)+pow(voxelsFloat[i+1],2)+pow(voxelsFloat[i+2],2))
                    *sqrt(pow(voxels[i],2)+pow(voxels[i+1],2)+pow(voxels[i+2],2));
            float dotProd = voxelsFloat[i]*voxels[i]+voxelsFloat[i+1]*voxels[i+1]+voxelsFloat[i+2]*voxels[i+2];
            error += acos(dotProd/mag);
        }
    }
    return error;
}

void writeVectorField(float * vector, int size) {
    
    float * x = new float[size];
    float * y = new float[size];
    for(int i = 0; i < size; i ++) {
        x[i] = vector[i*2];
        y[i] = vector[i*2+1];
    }

    FILE * fx = fopen("result_x.raw", "wb");
    fwrite(x, sizeof(float), size, fx);
    fclose(fx);

    FILE * fy = fopen("result_y.raw", "wb");
    fwrite(y, sizeof(float), size, fy);
    fclose(fy);
}

float * run2DKernels(Context context, CommandQueue queue, float * voxels, int SIZE_X, int SIZE_Y, float mu, int ITERATIONS, int datatype) {
    ImageFormat storageFormat;
    if(datatype == sizeof(short)) {
        std::cout << "Using 16 bit floats" << std::endl;
        storageFormat = ImageFormat(CL_RG, CL_SNORM_INT16);
    } else {
        std::cout << "Using 32 bit floats" << std::endl;
        storageFormat = ImageFormat(CL_RG, CL_FLOAT);
    }

    Program program = buildProgramFromSource(context, "2Dkernels.cl");

    // Create kernels
    Kernel initKernel = Kernel(program, "GVF2DInit");
    Kernel iterationKernel = Kernel(program, "GVF2DIteration");
    Kernel resultKernel = Kernel(program, "GVF2DResult");

    // Load volume to GPU
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



    // Read the result in some way (maybe write to a seperate raw file)
    volume = Image2D(context, CL_MEM_WRITE_ONLY, ImageFormat(CL_R,CL_FLOAT), SIZE_X, SIZE_Y);
    Image2D vectorFieldFinal = Image2D(context, CL_MEM_WRITE_ONLY, ImageFormat(CL_RG,CL_FLOAT), SIZE_X, SIZE_Y);
    resultKernel.setArg(0, volume);
    resultKernel.setArg(1, vectorField);
    resultKernel.setArg(2, vectorFieldFinal);
    queue.enqueueNDRangeKernel(
            resultKernel,
            NullRange,
            NDRange(SIZE_X, SIZE_Y),
            NullRange
    );
    queue.finish();
    voxels = new float[SIZE_X*SIZE_Y];
    std::cout << "Reading vector field from device..." << std::endl;
    queue.enqueueReadImage(volume, CL_TRUE, offset, region, 0, 0, voxels);
    std::cout << "Writing vector field to PNG file..." << std::endl;
    writeImage(voxels, SIZE_X,SIZE_Y);

    // Create vector raw files
    float * vector = new float[SIZE_X*SIZE_Y*2];
    queue.enqueueReadImage(vectorFieldFinal, CL_TRUE, offset, region, 0, 0, vector);
    writeVectorField(vector, SIZE_X*SIZE_Y);
    
    return voxels;
}

float * run3DKernels(Context context, CommandQueue queue, float * voxels, int SIZE_X, int SIZE_Y, int SIZE_Z, float mu, int ITERATIONS, int datatype) {
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
    Program program = buildProgramFromSource(context, "3Dkernels.cl");

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

    // Delete volume from device
    //volume.~Image3D();

    // copy vector field and create double buffer
    Image3D vectorField2 = Image3D(context, CL_MEM_READ_WRITE, storageFormat, SIZE_X, SIZE_Y, SIZE_Z);
    //queue.enqueueCopyImage(initVectorField, vectorField, offset, offset, region);
    //queue.finish();

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
    displaySlice(voxels, SIZE_X,SIZE_Y,SIZE_Y,100);
    return voxels;
}

float * run3DKernelsWithoutTexture(Context context, CommandQueue queue, float * voxels, int SIZE_X, int SIZE_Y, int SIZE_Z, float mu, int ITERATIONS, int datatype) {
    
    Program program = buildProgramFromSource(context, "3DkernelsNO_WRITE_TEX.cl");

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

    // Read the result in some way (maybe write to a seperate raw file)
    Buffer result = Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*SIZE_X*SIZE_Y*SIZE_Z);
    resultKernel.setArg(0, result);
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
    queue.enqueueReadBuffer(result, CL_TRUE, 0, sizeof(float)*SIZE_X*SIZE_Y*SIZE_Z, voxels);
    std::cout << "Writing vector field to RAW file..." << std::endl;
    writeToRaw(voxels, "result.raw", SIZE_X, SIZE_Y, SIZE_Z);
    displaySlice(voxels, SIZE_X,SIZE_Y,SIZE_Y,100);
    return voxels;
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
    vector<Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
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

    if(argc == 7) {
        filename = argv[1];
        SIZE_X = atoi(argv[2]);
        SIZE_Y = atoi(argv[3]);
        SIZE_Z = atoi(argv[4]);
        mu = atof(argv[5]);
        ITERATIONS = atoi(argv[6]);

        std::cout << "Reading RAW file " << filename << std::endl;
        float * voxels = parseRawFile(filename, SIZE_X, SIZE_Y, SIZE_Z);

        float * output;
        if((int)devices[0].getInfo<CL_DEVICE_EXTENSIONS>().find("cl_khr_3d_image_writes") > -1) {
            run3DKernels(context, queue, voxels, SIZE_X, SIZE_Y, SIZE_Z, mu, ITERATIONS, bytes);
        } else {
            run3DKernelsWithoutTexture(context, queue, voxels, SIZE_X, SIZE_Y, SIZE_Z, mu, ITERATIONS, bytes);
        }
    } else if(argc == 6) {
        filename = argv[1];
        SIZE_X = atoi(argv[2]);
        SIZE_Y = atoi(argv[3]);
        mu = atof(argv[4]);
        ITERATIONS = atoi(argv[5]);

        std::cout << "Reading image file " << filename << std::endl;
        float * pixels = parseImageFile(filename, SIZE_X, SIZE_Y);
        run2DKernels(context, queue, pixels, SIZE_X, SIZE_Y, mu, ITERATIONS, bytes);
    } else {
        std::cout << "Usage:" << std::endl << "---------------------------------" << std::endl <<
            "For 2D images:" << std::endl << 
            "./host filename.jpg size_x size_y mu #iterations [-16bit] [--device cpu/gpu]" << std::endl <<
            "For 3D images:" << std::endl <<
            "./host filename.raw size_x size_y size_z mu #iterations [-16bit] [--device cpu/gpu]" << std::endl;
        exit(-1);
    }
        
    } catch(Error error) {
       std::cout << error.what() << "(" << error.err() << ")" << std::endl;
       std::cout << getCLErrorString(error.err()) << std::endl;
    }

   return 0;
}
