__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void GVF3DInit(__read_only image3d_t volume, __global short * vector_field ) {
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
    short4 converted = convert_short4_sat_rte(gradient * 32767.0f);
    vstore4(converted, pos.x+pos.y*get_global_size(0)+pos.z*get_global_size(0)*get_global_size(1), vector_field);
}

#define LA3D(x,y,z) (x + (y<<3) + (z<<6))
__kernel __attribute__((reqd_work_group_size(8,8,4))) void GVF3DIteration(__read_only image3d_t init_vector_field, __global short const * restrict read_vector_field, __global short * write_vector_field, __private float mu) {

    int4 writePos = {
        get_global_id(0)-(get_group_id(0)*2+1), 
        get_global_id(1)-(get_group_id(1)*2+1), 
        get_global_id(2)-(get_group_id(2)*2+1),
        0
    };
    if(writePos.x > 255 || writePos.y > 255 || writePos.z > 255
            || writePos.x < 0 || writePos.y < 0 || writePos.z < 0)
        writePos = (int4)(50, 50, 50, 0);
    int3 localPos = {get_local_id(0), get_local_id(1), get_local_id(2)};
    
    // TODO: Enforce mirror boundary conditions
   
    // Allocate shared memory
    __local float2 sharedMemory[256];
	__local float sharedMemorySingle[256];

    // Read into shared memory
    short3 tempV = vload3(writePos.x+writePos.y*256+writePos.z*256*256, read_vector_field);
    float3 v = max((float3)(-1.0f,-1.0f,-1.0f), 
                convert_float3(tempV) / 32767.0f);
    sharedMemory[LA3D(localPos.x,localPos.y,localPos.z)]= v.xy;
    sharedMemorySingle[LA3D(localPos.x,localPos.y,localPos.z)] = v.z;

    int3 comp = (localPos == (int3)(0,0,0)) +
        (localPos == (int3)(7,7,3));
	
    // Synchronize the threads in the group
    barrier(CLK_LOCAL_MEM_FENCE);

    if(comp.x+comp.y+comp.z ==  0) {
        // Load data from shared memory and do calculations
        float4 init_vector = read_imagef(init_vector_field, sampler, writePos);

        float3 fx1, fx_1, fy1, fy_1, fz1, fz_1;
        fx1.xy = sharedMemory[LA3D(localPos.x+1,localPos.y,localPos.z)];
        fx1.z = sharedMemorySingle[LA3D(localPos.x+1,localPos.y,localPos.z)];
        fx_1.xy = sharedMemory[LA3D(localPos.x-1,localPos.y,localPos.z)];
        fx_1.z = sharedMemorySingle[LA3D(localPos.x-1,localPos.y,localPos.z)];
        fy1.xy = sharedMemory[LA3D(localPos.x,localPos.y+1,localPos.z)];
        fy1.z = sharedMemorySingle[LA3D(localPos.x,localPos.y+1,localPos.z)];
        fy_1.xy = sharedMemory[LA3D(localPos.x,localPos.y-1,localPos.z)];
        fy_1.z = sharedMemorySingle[LA3D(localPos.x,localPos.y-1,localPos.z)];
        fz1.xy = sharedMemory[LA3D(localPos.x,localPos.y,localPos.z+1)];
        fz1.z = sharedMemorySingle[LA3D(localPos.x,localPos.y,localPos.z+1)];
        fz_1.xy = sharedMemory[LA3D(localPos.x,localPos.y,localPos.z-1)];
        fz_1.z = sharedMemorySingle[LA3D(localPos.x,localPos.y,localPos.z-1)];

        // Update the vector field: Calculate Laplacian using a 3D central difference scheme
        float3 laplacian = -6*v.xyz + fx1 + fx_1 + fy1 + fy_1 + fz1 + fz_1;

        v += mu * laplacian - (v - init_vector.xyz)*init_vector.w;


        tempV = convert_short3_sat_rte(v * 32767.0f);
        vstore3(tempV, writePos.x + writePos.y*256 + writePos.z*256*256, write_vector_field);
    }
}

__kernel void GVF3DResult(__global float * result, __global short * vectorField) {

    int pos = get_global_id(0) + get_global_id(1)*get_global_size(0) + 
        get_global_id(2)*get_global_size(0)*get_global_size(1);
    short3 vectorTemp = vload3(pos, vectorField);
    float3 v = (float3)(
            max(-1.0f, (float)vectorTemp.x / 32767.0f),
            max(-1.0f, (float)vectorTemp.y / 32767.0f),
            max(-1.0f, (float)vectorTemp.z / 32767.0f));
    result[pos] = length(v);
}


