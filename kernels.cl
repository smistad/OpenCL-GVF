__kernel void GVFInit(__read_only image3d_t volume, __write_only image3d_t vector_field ) {
    // gradient of volume and store it in vector_field
}

__kernel void GVFIteration(__read_only image3d_t read_vector_field, __write_only image3d_t write_vector_field, __private float mu) {
    // Enforce mirror boundary conditions

    // Update the vector field
}
