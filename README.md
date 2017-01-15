About
=============================
This is an optimized OpenCL implementation of Gradient Vector Flow (GVF) that runs on GPUs and CPUs for both 2D and 3D.
For more details about the implementation, see the scientific article ["Real-time gradient vector flow on GPUs using OpenCL"](http://www.springerlink.com/content/v0071r27706u5135/) and https://www.eriksmistad.no/gpu-based-gradient-vector-flow-using-opencl/.

*Note that this software has been moved into the [FAST framework](https://github.com/smistad/FAST/) where it is further developed and maintained.*

Dependencies
=============================
* OpenCL
* GTK+ 2 (see SIPL installation notes)

Note: When you download the code, the content of the SIPL and OpenCLUtilities folders will be empty. This is a bug in GitHub and you must download these libs as well and put the content inside the corresponding folders (don't worry it is easy).

Compile
=============================
* Use cmake on CMakeList.txt

Usage
=============================
For 2D images:
------------------------------
./GVF filename.jpg mu #iterations [-16bit] [--device cpu/gpu]

For 3D images:
------------------------------
./GVF filename.mhd mu #iterations [-16bit] [--device cpu/gpu]

Note: The default implementation will use a 32-bit floating point storage format, but if -16bit is specified it will use a 16-bit storage format.
