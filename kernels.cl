#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable
#pragma OPENCL EXTENSION cl_amd_printf : enable

__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void GVF3DInit(__read_only image3d_t volume, __write_only image3d_t vector_field ) {
    // Calculate gradient using a 1D central difference for each dimension, with spacing 1
    int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};

    float f100 = read_imagef(volume, sampler, pos + (int4)(1,0,0,0)).x;
    float f_100 = read_imagef(volume, sampler, pos - (int4)(1,0,0,0)).x;
    float f010 = read_imagef(volume, sampler, pos + (int4)(0,1,0,0)).x;
    float f0_10 = read_imagef(volume, sampler, pos - (int4)(0,1,0,0)).x;
    float f001 = read_imagef(volume, sampler, pos + (int4)(0,0,1,0)).x;
    float f00_1 = read_imagef(volume, sampler, pos - (int4)(0,0,1,0)).x;

    float4 gradient = {
        0.5f*(f100-f_100), 
        0.5f*(f010-f0_10),
        0.5f*(f001-f00_1),
        0};

    gradient.w = gradient.x*gradient.x + gradient.y*gradient.y + gradient.z*gradient.z;

    write_imagef(vector_field, pos, gradient); 
}

#define LA3D(x,y,z) (x + (y<<3) + (z<<6))
__kernel __attribute__((reqd_work_group_size(8,8,4))) void GVF3DIteration(__read_only image3d_t init_vector_field, __read_only image3d_t read_vector_field, __write_only image3d_t write_vector_field, __private float mu) {

    int4 writePos = {
        get_global_id(0)-(get_group_id(0)*2+1), 
        get_global_id(1)-(get_group_id(1)*2+1), 
        get_global_id(2)-(get_group_id(2)*2+1), 
        0
    };
    int3 localPos = {get_local_id(0), get_local_id(1), get_local_id(2)};
    
    // Enforce mirror boundary conditions
    //int4 size = {get_global_size(0), get_global_size(1), get_global_size(2), 0};
    //int4 pos = writePos;
    //pos = select(pos, (int4)(2,2,2,0), pos == (int4)(0,0,0,0));
    //pos = select(pos, size-3, pos == size-1);
   
    // Allocate shared memory
    __local float2 sharedMemory[256];
	__local float sharedMemorySingle[256];

    // Read into shared memory
    float4 v = read_imagef(read_vector_field, sampler, writePos);
    const uint pos = LA3D(localPos.x,localPos.y,localPos.z);
    sharedMemory[pos]= v.xy;
    sharedMemorySingle[pos] = v.z;

    /*
    int x = localPos.x;
    int y = localPos.y;
    int z = localPos.z;
    int w = 0;
    int rowID2 = (LA(x,y,z) >> 4) & 0x1F;
    int bankID2 = ((LA(x,y,z)) & 0xF) << 1;
    int rowID = (LA(x,y,z) >> 5) & 0x1F;
	int bankID = ((LA(x,y,z)) & 0x1F);
    printf("Float2s: %d %d %d - %d - %d\n", x,y,z, bankID2, rowID2);
	printf("Floats: %d %d %d - %d - %d\n", x,y,z, bankID, rowID);
    */
    int3 comp = (localPos == (int3)(0,0,0)) +
        (localPos == (int3)(7,7,3));
	
    // Synchronize the threads in the group
    barrier(CLK_LOCAL_MEM_FENCE);

    if(comp.x+comp.y+comp.z ==  0) {
        // Load data from shared memory and do calculations
        float4 init_vector = read_imagef(init_vector_field, sampler, writePos); // should read from pos

        float3 fx1, fx_1, fy1, fy_1, fz1, fz_1;
        // x+1 
        fx1.xy = sharedMemory[pos+1];
        fx1.z = sharedMemorySingle[pos+1];
        // x-1
        fx_1.xy = sharedMemory[pos-1];
        fx_1.z = sharedMemorySingle[pos-1];
        // y+1
        fy1.xy = sharedMemory[pos+8];
        fy1.z = sharedMemorySingle[pos+8];
        // y-1
        fy_1.xy = sharedMemory[pos-8];
        fy_1.z = sharedMemorySingle[pos-8];
        // z+1
        fz1.xy = sharedMemory[pos+64];
        fz1.z = sharedMemorySingle[pos+64];
        // z-1
        fz_1.xy = sharedMemory[pos-64];
        fz_1.z = sharedMemorySingle[pos-64];

        // Update the vector field: Calculate Laplacian using a 3D central difference scheme
        float3 laplacian = -6*v.xyz + fx1 + fx_1 + fy1 + fy_1 + fz1 + fz_1;

        v.xyz += mu * laplacian - (v.xyz - init_vector.xyz)*init_vector.w;

        write_imagef(write_vector_field, writePos, v);
    }
}

__kernel void GVF3DResult(__write_only image3d_t result, __read_only image3d_t vectorField) {

    int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};
    float4 vector = read_imagef(vectorField, sampler, pos);
    vector.w = 0;
    //if(1 < length(vector))
    //    printf("%f %f %f\n", vector.x, vector.y, vector.z);
    write_imagef(result, pos, length(vector));
}




__kernel void GVF2DInit(__read_only image2d_t volume, __write_only image2d_t vector_field ) {
    // Calculate gradient using a 1D central difference for each dimension, with spacing 1
    int2 pos = {get_global_id(0), get_global_id(1)};

    float f10 = read_imagef(volume, sampler, pos + (int2)(1,0)).x;
    float f_10 = read_imagef(volume, sampler, pos - (int2)(1,0)).x;
    float f01 = read_imagef(volume, sampler, pos + (int2)(0,1)).x;
    float f0_1 = read_imagef(volume, sampler, pos - (int2)(0,1)).x;

    float4 gradient = {
        0.5f*(f10-f_10), 
        0.5f*(f01-f0_1),
        0,
        0
    };

    write_imagef(vector_field, pos, gradient); 
}

#define LA2D(x,y) (x + (y<<4))
__kernel __attribute__((reqd_work_group_size(16,16,1))) void GVF2DIteration(__read_only image2d_t init_vector_field, __read_only image2d_t read_vector_field, __write_only image2d_t write_vector_field, __private float mu) {

    int2 writePos = {
        get_global_id(0)-(get_group_id(0)*2+1), 
        get_global_id(1)-(get_group_id(1)*2+1) 
    };
    int2 localPos = {get_local_id(0), get_local_id(1)};
    
    // TODO: Enforce mirror boundary conditions
   
    // Allocate shared memory
    __local float2 sharedMemory[256];

    // Read into shared memory
    float2 v = read_imagef(read_vector_field, sampler, writePos).xy;
    sharedMemory[LA2D(localPos.x,localPos.y)]= v;

    int2 comp = (localPos == (int2)(0,0)) +
        (localPos == (int2)(15,15));
	
    // Synchronize the threads in the group
    barrier(CLK_LOCAL_MEM_FENCE);

    if(comp.x+comp.y ==  0) {
        // Load data from shared memory and do calculations
        float2 init_vector = read_imagef(init_vector_field, sampler, writePos).xy; // should read from pos

        float2 fx1, fx_1, fy1, fy_1;
        fx1 = sharedMemory[LA2D(localPos.x+1,localPos.y)];
        fy1 = sharedMemory[LA2D(localPos.x,localPos.y+1)];
        fx_1 = sharedMemory[LA2D(localPos.x-1,localPos.y)];
        fy_1 = sharedMemory[LA2D(localPos.x,localPos.y-1)];

        // Update the vector field: Calculate Laplacian using a 3D central difference scheme
        float2 laplacian = -4*v + fx1 + fx_1 + fy1 + fy_1;

        v += mu * laplacian - (v - init_vector)*(init_vector.x*init_vector.x+init_vector.y*init_vector.y);

        write_imagef(write_vector_field, writePos, (float4)(v.x,v.y,0,0));
    }
}

__kernel void GVF2DResult(__write_only image2d_t result, __read_only image2d_t vectorField) {
    int2 pos = {get_global_id(0), get_global_id(1)};
    float4 vector = read_imagef(vectorField, sampler, pos);
    vector.z = 0;
    vector.w = 0;
    write_imagef(result, pos, length(vector));
}
