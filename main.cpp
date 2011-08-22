#define __NO_STD_VECTOR // Use cl::vector instead of STL version
#define __CL_ENABLE_EXCEPTIONS

#include <time.h>
#include <CL/cl.hpp>
#include "openCLUtilities.hpp"
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

int main(int argc, char ** argv) {
    char * filename;
    int SIZE_X, SIZE_Y, SIZE_Z, ITERATIONS;
    float mu;
    bool run3D;
    if(argc == 7) {
        filename = argv[1];
        SIZE_X = atoi(argv[2]);
        SIZE_Y = atoi(argv[3]);
        SIZE_Z = atoi(argv[4]);
        mu = atof(argv[5]);
        ITERATIONS = atoi(argv[6]);
        run3D = true;
    } else if(argc == 6) {
        filename = argv[1];
        SIZE_X = atoi(argv[2]);
        SIZE_Y = atoi(argv[3]);
        mu = atof(argv[4]);
        ITERATIONS = atoi(argv[5]);
        run3D = false;
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

        // Query the size of available memory
        unsigned int memorySize = devices[0].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();

        std::cout << "Available memory on selected device " << memorySize << " bytes "<< std::endl;

        
        ImageFormat storageFormat;
        if(run3D) {
        if(memorySize > SIZE_X*SIZE_Y*SIZE_Z*4*4*3) {
            storageFormat = ImageFormat(CL_RGBA, CL_FLOAT);
            std::cout << "Using 32 bits floats texture storage" << std::endl;
        } else if(memorySize > SIZE_X*SIZE_Y*SIZE_Z*2*4*3) {
            storageFormat = ImageFormat(CL_RGBA, CL_SNORM_INT16);
            std::cout << "Not enough memory on device for 32 bit floats, using 16bit for texture storage instead (WARNING: Reduced accuracy)." << std::endl;
        } else {
            std::cout << "There is not enough memory on this device to calculate the GVF for this dataset!" << std::endl;
            exit(-1);
        }

        // Create Kernels
        Kernel initKernel = Kernel(program, "GVF3DInit");
        Kernel iterationKernel = Kernel(program, "GVF3DIteration");
        Kernel resultKernel = Kernel(program, "GVF3DResult");

        // Load volume to GPU
        std::cout << "Reading RAW file " << filename << std::endl;
        float * voxels = parseRawFile(filename, SIZE_X, SIZE_Y, SIZE_Z);

        

        Image3D volume = Image3D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ImageFormat(CL_R, CL_FLOAT), SIZE_X, SIZE_Y, SIZE_Z, 0, 0, voxels);
        delete[] voxels;

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

        int rangeX = SIZE_X;
        int rangeY = SIZE_Y;
        int rangeZ = SIZE_Z;
        while(rangeX % 6 != 0)
            rangeX++;
        while(rangeY % 6 != 0)
            rangeY++;
        while(rangeZ % 2 != 0)
            rangeZ++;

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
                    NDRange(8*rangeX/6,8*rangeY/6,4*rangeZ/2),
                    NDRange(8,8,4)
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
        displaySlice(voxels, SIZE_X,SIZE_Y,SIZE_Y,100);
        delete[] voxels;
        } else { // 2D!

            // Create kernels
            Kernel initKernel = Kernel(program, "GVF2DInit");
            Kernel iterationKernel = Kernel(program, "GVF2DIteration");
            Kernel resultKernel = Kernel(program, "GVF2DResult");

            // Load volume to GPU
            std::cout << "Reading image file " << filename << std::endl;
            float * voxels = parseImageFile(filename, SIZE_X, SIZE_Y);

            clock_t start = clock();

            Image2D volume = Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ImageFormat(CL_R, CL_FLOAT), SIZE_X, SIZE_Y, 0, voxels);
            delete[] voxels;

            storageFormat = ImageFormat(CL_RG, CL_SNORM_INT16);
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
            while(rangeX % 14 != 0)
                rangeX++;

            while(rangeY % 14 != 0)
                rangeY++;

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
                        NDRange(16*rangeX/14, 16*rangeY/14),
                        NDRange(16,16)
                );
            }
            queue.finish();

            // Read the result in some way (maybe write to a seperate raw file)
            volume = Image2D(context, CL_MEM_WRITE_ONLY, ImageFormat(CL_R,CL_FLOAT), SIZE_X, SIZE_Y);
            resultKernel.setArg(0, volume);
            resultKernel.setArg(1, vectorField);
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
            std::cout << "Processing finished: " << (double)(clock()-start)/CLOCKS_PER_SEC << " seconds used " << std::endl;
            std::cout << "Writing vector field to RAW file..." << std::endl;
            writeImage(voxels, SIZE_X,SIZE_Y);
            delete[] voxels;
        }

    } catch(Error error) {
       std::cout << error.what() << "(" << error.err() << ")" << std::endl;
       std::cout << getCLErrorString(error.err()) << std::endl;
    }

   return 0;
}
