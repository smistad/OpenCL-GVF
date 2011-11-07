
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void GVF2DInit(__read_only image2d_t volume, __write_only image2d_t vector_field ) {
    // Calculate gradient using a 1D central difference for each dimension, with spacing 1
    int2 writePos = {get_global_id(0), get_global_id(1)};

    int2 size = {get_global_size(0), get_global_size(1)};
    int2 pos = writePos;
    //pos = select(pos, (int2)(2,2), pos == (int2)(0,0));
    //pos = select(pos, size-3, pos >= size-1);

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

    write_imagef(vector_field, writePos, gradient); 
}

#define LA2D(x,y) (x + (y<<4))
__kernel __attribute__((reqd_work_group_size(16,16,1))) void GVF2DIteration(__read_only image2d_t init_vector_field, __read_only image2d_t read_vector_field, __write_only image2d_t write_vector_field, __private float mu) {

    int2 writePos = {
        get_global_id(0)-(get_group_id(0)*2+1), 
        get_global_id(1)-(get_group_id(1)*2+1) 
    };
    int2 localPos = {get_local_id(0), get_local_id(1)};
    
    // TODO: Enforce mirror boundary conditions
    int2 size = {get_image_width(init_vector_field), get_image_height(init_vector_field)};
    int2 pos = writePos;
    //pos = select(pos, (int2)(2,2), pos == (int2)(0,0));
    //pos = select(pos, size-3, pos >= size-1);

    // Allocate shared memory
    __local float2 sharedMemory[256];

    // Read into shared memory
    float2 v = read_imagef(read_vector_field, sampler, pos).xy;
    sharedMemory[LA2D(localPos.x,localPos.y)]= v;

    // Ensure that it don't write outside of the image
    int2 comp = (localPos == (int2)(0,0)) +
        (localPos == (int2)(15,15)) + (writePos > size-1);
	
    // Synchronize the threads in the group
    barrier(CLK_LOCAL_MEM_FENCE);

    if(comp.x+comp.y ==  0) {
        // Load data from shared memory and do calculations
        float2 init_vector = read_imagef(init_vector_field, sampler, pos).xy;

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
