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

__kernel void GVFIteration(__read_only image3d_t init_vector_field, __read_only image3d_t read_vector_field, __write_only image3d_t write_vector_field, __private float mu) {

    int4 writePos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};
    
    // Enforce mirror boundary conditions
    int4 size = {get_global_size(0), get_global_size(1), get_global_size(2), 0};
    int4 pos = writePos;
    pos = select(pos, (int4)(2,2,2,0), pos == (int4)(0,0,0,0));
    pos = select(pos, size-3, pos == size-1);
   
    /*
    if(pos.x == 0)
        pos.x = 2;
    if(pos.y == 0)
        pos.y = 2;
    if(pos.z == 0)
        pos.z = 2;
    if(pos.x == get_global_size(0)-1)
        pos.x = get_global_size(0)-3;
    if(pos.y == get_global_size(1)-1)
        pos.y = get_global_size(1)-3;
    if(pos.z == get_global_size(2)-1)
        pos.z = get_global_size(2)-3;
    
    int4 diff = (pos2 != pos);
    if(diff.x+diff.y+diff.z+diff.w > 0)
        printf("iiiik!\n");
    */

    float4 vector = read_imagef(read_vector_field, sampler, pos);
    float4 fx1 = read_imagef(read_vector_field, sampler, pos + (int4)(1,0,0,0));
    float4 fx_1 = read_imagef(read_vector_field, sampler, pos - (int4)(1,0,0,0));
    float4 fy1 = read_imagef(read_vector_field, sampler, pos + (int4)(0,1,0,0));
    float4 fy_1 = read_imagef(read_vector_field, sampler, pos - (int4)(0,1,0,0));
    float4 fz1 = read_imagef(read_vector_field, sampler, pos + (int4)(0,0,1,0));
    float4 fz_1 = read_imagef(read_vector_field, sampler, pos - (int4)(0,0,1,0));
    float4 init_vector = read_imagef(init_vector_field, sampler, pos);

    // Update the vector field: Calculate Laplacian using a 3D central difference scheme
    float4 laplacian = -6*vector + fx1 + fx_1 + fy1 + fy_1 + fz1 + fz_1;

    vector += mu * laplacian - (vector - init_vector)*init_vector.w;


    write_imagef(write_vector_field, writePos, vector);
}

__kernel void GVFResult(__write_only image3d_t result, __read_only image3d_t vectorField) {

    int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};
    float4 vector = read_imagef(vectorField, sampler, pos);
    vector.w = 0;
    //if(1 < length(vector))
    //    printf("%f %f %f\n", vector.x, vector.y, vector.z);
    write_imagef(result, pos, length(vector));
}
