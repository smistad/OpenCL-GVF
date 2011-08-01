__kernel void GVFInit(__read_only image3d_t volume, __write_only image3d_t vector_field ) {
    // gradient of volume and store it in vector_field
    // Calculate gradient using a 1D central difference for each dimension, with spacing 1
    int4 pos = {get_global_id(0), get_global_id(1), get_global_id(2), 0};

    int f = read_imagei(volume, sampler, pos).x;
    int fx1 = read_imagei(volume, sampler, pos + (int4)(1,0,0,0)).x;
    int fx_1 = read_imagei(volume, sampler, pos - (int4)(1,0,0,0)).x;
    int fy1 = read_imagei(volume, sampler, pos + (int4)(0,1,0,0)).x;
    int fy_1 = read_imagei(volume, sampler, pos - (int4)(0,1,0,0)).x;
    int fz1 = read_imagei(volume, sampler, pos + (int4)(0,0,1,0)).x;
    int fz_1 = read_imagei(volume, sampler, pos - (int4)(0,0,1,0)).x;

    int4 gradient = {
        fx1 - 2*f + fx_1,
        fy1 - 2*f + fy_1,
        fz1 - 2*f + fz_1,
        0};

    gradient.w = gradient.x*gradient.x + gradient.y*gradient.y + gradient.z*gradient-z;

    write_imagei(vector_field, pos, gradient); 

}

__kernel void GVFIteration(__read_only image3d_t read_vector_field, __write_only image3d_t write_vector_field, __private float mu) {
    // Enforce mirror boundary conditions

    // Update the vector field

    // Calculate laplacian using a 3D central difference scheme
}
