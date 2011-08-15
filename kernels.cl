#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable
#pragma OPENCL EXTENSION cl_amd_printf : enable

__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void GVFInit(__read_only image3d_t volume, __write_only image3d_t vector_field ) {
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

__kernel __attribute__((reqd_work_group_size(6,6,6))) void GVFIteration(__read_only image3d_t init_vector_field, __read_only image3d_t read_vector_field, __write_only image3d_t write_vector_field, __private float mu) {

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
    __local float2 sharedMemory[12][6][6];

    // Read into shared memory
    float4 v = read_imagef(read_vector_field, sampler, writePos);
    sharedMemory[localPos.y*2][localPos.z][localPos.x] = v.xy;
    sharedMemory[localPos.y*2+1][localPos.z][localPos.x] = v.zw;

    int3 comp = (localPos == (int3)(0,0,0)) +
        (localPos == (int3)(get_local_size(0)-1,get_local_size(1)-1,get_local_size(2)-1));

    // Synchronize the threads in the group
    barrier(CLK_LOCAL_MEM_FENCE);

    if(comp.x+comp.y+comp.z == 0) {
    // Load data from shared memory and do calculations
    float4 init_vector = read_imagef(init_vector_field, sampler, writePos); // should read from pos
    float3 vector;
    vector.xy = sharedMemory[localPos.y*2][localPos.z][localPos.x];
    vector.z = sharedMemory[localPos.y*2+1][localPos.z][localPos.x].x;
    float3 fx1;
    fx1.xy = sharedMemory[(localPos.y)*2][localPos.z][localPos.x+1];
    fx1.z = sharedMemory[(localPos.y)*2+1][localPos.z][localPos.x+1].x;
    float3 fx_1;
    fx_1.xy = sharedMemory[(localPos.y)*2][localPos.z][localPos.x-1];
    fx_1.z = sharedMemory[(localPos.y)*2+1][localPos.z][localPos.x-1].x;
    float3 fy1;
    fy1.xy = sharedMemory[(localPos.y+1)*2][localPos.z][localPos.x];
    fy1.z = sharedMemory[(localPos.y+1)*2+1][localPos.z][localPos.x].x;
    float3 fy_1;
    fy_1.xy = sharedMemory[(localPos.y-1)*2][localPos.z][localPos.x];
    fy_1.z = sharedMemory[(localPos.y-1)*2+1][localPos.z][localPos.x].x;
    float3 fz1;
    fz1.xy = sharedMemory[(localPos.y)*2][localPos.z+1][localPos.x];
    fz1.z = sharedMemory[(localPos.y)*2+1][localPos.z+1][localPos.x].x;
    float3 fz_1;
    fz_1.xy = sharedMemory[(localPos.y)*2][localPos.z-1][localPos.x];
    fz_1.z = sharedMemory[(localPos.y)*2+1][localPos.z-1][localPos.x].x;

    // Update the vector field: Calculate Laplacian using a 3D central difference scheme
    float3 laplacian = -6*vector + fx1 + fx_1 + fy1 + fy_1 + fz1 + fz_1;

    vector += mu * laplacian - (vector - init_vector.xyz)*init_vector.w;


    write_imagef(write_vector_field, writePos, (float4)(vector.x,vector.y,vector.z,0));
    }
}

__kernel void GVFResult(__write_only image3d_t result, __read_only image3d_t vectorField) {

    int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};
    float4 vector = read_imagef(vectorField, sampler, pos);
    vector.w = 0;
    //if(1 < length(vector))
    //    printf("%f %f %f\n", vector.x, vector.y, vector.z);
    write_imagef(result, pos, length(vector));
}
