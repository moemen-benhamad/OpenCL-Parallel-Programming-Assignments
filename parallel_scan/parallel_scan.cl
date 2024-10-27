__kernel void parallel_scan(__global int* input_array, __global int* output_array, const int s) {
	int id = get_global_id(0);

    if(id < s)
        output_array[id] = input_array[id];
    else 
        output_array[id] = input_array[id] + input_array[id - s];
}

