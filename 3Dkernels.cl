#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable

__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void GVF3DInit(__read_only image3d_t volume, __write_only image3d_t init_vector_field, __write_only image3d_t vector_field ) {
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
        0.5f*(f001-f00_1)
    };

    write_imagef(init_vector_field, pos, gradient);
    write_imagef(vector_field, pos, gradient); 
}

__kernel __attribute__((reqd_work_group_size(4,4,4))) void GVF3DIteration(__read_only image3d_t init_vector_field, __read_only image3d_t read_vector_field, __write_only image3d_t write_vector_field, __private float mu) {

    int4 writePos = {
        get_global_id(0),
        get_global_id(1),
        get_global_id(2),
        0
    };
    // Enforce mirror boundary conditions
    int4 size = {get_image_width(init_vector_field), get_image_height(init_vector_field), get_image_depth(init_vector_field), 0};
    int4 pos = writePos;
    pos = select(pos, (int4)(2,2,2,0), pos == (int4)(0,0,0,0));
    pos = select(pos, size-3, pos >= size-1);

    // Load data from shared memory and do calculations
    float2 init_vector = read_imagef(init_vector_field, sampler, pos).xy;

    float4 v = read_imagef(read_vector_field, sampler, pos);
    float3 fx1 = read_imagef(read_vector_field, sampler, pos + (int4)(1,0,0,0)).xyz; 
    float3 fx_1 = read_imagef(read_vector_field, sampler, pos - (int4)(1,0,0,0)).xyz; 
    float3 fy1 = read_imagef(read_vector_field, sampler, pos + (int4)(0,1,0,0)).xyz; 
    float3 fy_1 = read_imagef(read_vector_field, sampler, pos - (int4)(0,1,0,0)).xyz; 
    float3 fz1 = read_imagef(read_vector_field, sampler, pos + (int4)(0,0,1,0)).xyz; 
    float3 fz_1 = read_imagef(read_vector_field, sampler, pos - (int4)(0,0,1,0)).xyz; 
    
    // Update the vector field: Calculate Laplacian using a 3D central difference scheme
    float3 laplacian = -6*v.xyz + fx1 + fx_1 + fy1 + fy_1 + fz1 + fz_1;

    v.xyz += mu * laplacian - (v.xyz - (float3)(init_vector.x, init_vector.y, v.w))*
        (init_vector.x*init_vector.x+init_vector.y*init_vector.y+v.w*v.w);

    write_imagef(write_vector_field, writePos, v);

}

__kernel void GVF3DResult(__read_only image3d_t vectorField, __write_only image3d_t vectorFieldFinal) {
    int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};
    float4 vector = read_imagef(vectorField, sampler, pos);
    write_imagef(vectorFieldFinal, pos, vector);
}
