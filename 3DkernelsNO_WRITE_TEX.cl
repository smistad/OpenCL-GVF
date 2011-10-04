__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void GVF3DInit(__read_only image3d_t volume, __global float * init_vector_field, __global float * vector_field ) {
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
        0
        };
    gradient.w = gradient.x*gradient.x+gradient.y*gradient.y+gradient.z*gradient.z;
    vstore4(gradient, pos.x + pos.y*get_global_size(0) + pos.z*get_global_size(0)*get_global_size(1), init_vector_field);
    vstore3(gradient.xyz, pos.x + pos.y*get_global_size(0) + pos.z*get_global_size(0)*get_global_size(1), vector_field);

}

__kernel void GVF3DIteration(__read_only image3d_t init_vector_field, __global float const * restrict read_vector_field, __global float * write_vector_field, __private float mu) {
    int4 writePos = {
        get_global_id(0),
        get_global_id(1),
        get_global_id(2),
        0
    };
    // Enforce mirror boundary conditions
    int4 size = {get_global_size(0), get_global_size(1), get_global_size(2), 0};
    int4 pos = writePos;
    pos = select(pos, (int4)(2,2,2,0), pos == (int4)(0,0,0,0));
    pos = select(pos, size-3, pos >= size-1);
    int offset = pos.x+pos.y*size.x+pos.z*size.x*size.y;

    // Load data from shared memory and do calculations
    float4 init_vector = read_imagef(init_vector_field, sampler, pos);

    float3 v = vload3(offset, read_vector_field);
    float3 fx1 = vload3(offset+1, read_vector_field);
    float3 fx_1 = vload3(offset-1, read_vector_field);
    float3 fy1 = vload3(offset+size.x, read_vector_field);
    float3 fy_1 = vload3(offset-size.x, read_vector_field);
    float3 fz1 = vload3(offset+size.x*size.y, read_vector_field);
    float3 fz_1 = vload3(offset-size.x*size.y, read_vector_field);
    
    // Update the vector field: Calculate Laplacian using a 3D central difference scheme
    float3 laplacian = -6*v.xyz + fx1 + fx_1 + fy1 + fy_1 + fz1 + fz_1;

    v += mu * laplacian - (v - init_vector.xyz)*init_vector.w;

    vstore3(v, writePos.x+writePos.y*size.x+writePos.z*size.x*size.y, write_vector_field);

}

__kernel void GVF3DResult(__global float * result, __global float * vectorField) {

    int pos = get_global_id(0) + get_global_id(1)*get_global_size(0) + 
        get_global_id(2)*get_global_size(0)*get_global_size(1);
    float3 v = vload3(pos, vectorField);
    result[pos] = length(v);
}


