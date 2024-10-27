// ----------------------------------------------------------

#include <cstdio>
#include <cstdlib>

#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

// ----------------------------------------------------------

// We provide a small library so we can easily setup OpenCL
#include "clutils.h"

// ----------------------------------------------------------

void print_array(int* array, int size, const std::string& label) {
    std::cout << label << ": [";
    for (int i = 0; i < size; i++) {
        std::cout << array[i];
        if (i < size - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

bool compare_array(int* array1, int* array2, int size) {
    for (int i = 0; i < size; i++) {
        if (array1[i] != array2[i]) {
            return false;
        }
    }
    return true;
}

void parallel_scan_cpu(int* input, int* output, int N){
    output[0] = input[0];
    for (int i = 1; i < N; i++) {
        output[i] = output[i - 1] + input[i];
    }
}

int main(int argc,char **argv)
{
    const char *clu_File = SRC_PATH "parallel_scan.cl"; 

    // Initialize OpenCL
    cluInit();

    // After this call you have access to
    // clu_Context;      <= OpenCL context (pointer)
    // clu_Devices;      <= OpenCL device list (vector)
    // clu_Queue;        <= OpenCL queue (pointer)

    // Load Kernel
    cl::Program *program = cluLoadProgram(clu_File);
    cl::Kernel  *kernel = cluLoadKernel(program, "parallel_scan");

    // allocate memory and opencl buffer
    const int size = 16;
    cl::Buffer input_buffer(*clu_Context, CL_MEM_READ_ONLY, size * sizeof(int));
    cl::Buffer output_buffer(*clu_Context, CL_MEM_WRITE_ONLY, size * sizeof(int));

    int* input_array = new int[size];
    int* output_array = new int[size];
    int* cpu_output_array = new int[size];

    // fill input_buffer with values
    for (int i = 0; i < size; i++) {
        input_array[i] = i;
    }
    print_array(input_array, size, "input array");

    clu_Queue->enqueueWriteBuffer(input_buffer, true, 0, size * sizeof(int), input_array);
    parallel_scan_cpu(input_array, cpu_output_array, size);
    delete[] input_array;
    
    for (int s = 1; s < size; s *= 2){
        // execute kernel
        kernel->setArg(0, input_buffer);
        kernel->setArg(1, output_buffer);
        kernel->setArg(2, s);
        cl_int clerr = clu_Queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(size), cl::NDRange(size/4));
        cluCheckError(clerr, "Error running the kernel");
        clu_Queue->enqueueCopyBuffer(output_buffer, input_buffer, 0, 0, size * sizeof(int));
    }

    // copy back the results
    clu_Queue->enqueueReadBuffer(output_buffer, true, 0, size * sizeof(int), output_array);
    print_array(cpu_output_array, size, "(CPU) output array");
    print_array(output_array, size, "(GPU) output array");
    
    if(compare_array(output_array, cpu_output_array, size))
        std::cout << "Correctness verified!";
    else
        std::cout << "Results does not match";

    delete[] output_array;
    delete[] cpu_output_array;

    return 0;
}

// ----------------------------------------------------------

