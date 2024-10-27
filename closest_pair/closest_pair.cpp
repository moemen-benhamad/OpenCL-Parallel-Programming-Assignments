// ----------------------------------------------------------
#include <iostream>
#include <fstream>
#include <sstream>

#include <cmath>

using namespace std;

// ----------------------------------------------------------

// We provide input_array small library so we can easily setup OpenCL
#include "clutils.h"

// ----------------------------------------------------------

#define SQUARE(x) ((x) * (x))

struct Point{
    int x, y;
};

float cpu_closest_pair(Point* a, const int N){
    float min_dist = FLT_MAX;
    float dist = 0;
    for(int i = 0; i < N; i++){
        for(int j = i + 1; j < N; j++){
            dist = sqrt((float)(SQUARE(a[i].x - a[j].x) + SQUARE(a[i].y - a[j].y)));
            if (dist < min_dist)
                min_dist = dist;
        }
    }
    return min_dist;
}

int main(int argc,char **argv)
{

    const char *clu_File = SRC_PATH "closest_pair.cl"; 

    // Initialize OpenCL
    cluInit();

    // After this call you have access to
    // clu_Context;      <= OpenCL context (pointer)
    // clu_Devices;      <= OpenCL device list (vector)
    // clu_Queue;        <= OpenCL queue (pointer)

    // Load Kernel
    cl::Program *program = cluLoadProgram(clu_File);
    cl::Kernel  *kernel = cluLoadKernel(program, "closest_pair");

    // allocate memory and opencl buffer
    const int N = 16;
    const int size  = N * N;
    cl::Buffer input_buffer(*clu_Context, CL_MEM_READ_ONLY, N * sizeof(Point));
    cl::Buffer output_buffer(*clu_Context, CL_MEM_WRITE_ONLY, size * sizeof(float));

    // fill input_buffer and b_buffer with values
    Point* input_array = new Point[N];
    for (int i = 0; i < N - 1; i++) {
        input_array[i].x = SQUARE(i+1);
        input_array[i].y = i + 1;
    }

    // set a closest point to the last input_array filled value (for testing purposes)
    // expected min_dist = sqrt(5)
    int offset = 2;
    input_array[N - 1].x = SQUARE(N - 1) + offset;
    input_array[N - 1].y = N;
    clu_Queue->enqueueWriteBuffer(input_buffer, true, 0, N * sizeof(Point), input_array);

    std::cout << "Points : [ ";
    for(int i = 0; i <N; i++){
        std::cout << "(" << input_array[i].x << "," << input_array[i].y << ") ";
    }
    std::cout << "]" << std::endl << std::endl;

    float cpu_min_dist = cpu_closest_pair(input_array, N);

    delete[] input_array;
    
    kernel->setArg(0, input_buffer);
    kernel->setArg(1, output_buffer);
    kernel->setArg(2, N);

    cl_int clerr = clu_Queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(size), cl::NDRange(size/4));
    cluCheckError(clerr, "Error running the kernel");

    // copy back the results
    float* output_array = new float[size];
    clu_Queue->enqueueReadBuffer(output_buffer, true, 0, size * sizeof(float), output_array);

    // find minimum distance
    float &gpu_min_dist = *min_element(output_array, output_array + size );
    
    // print results
    std::cout << "[CPU] closest distance = " <<cpu_min_dist;
    std::cout << std::endl;
    std::cout << "[GPU] closest distance = " <<gpu_min_dist;
    std::cout << std::endl;

    double epsilon =  10e-6;
    if(fabs(gpu_min_dist - cpu_min_dist) < epsilon)
        std::cout << "Correctness verified!";
    else
        std::cout << "Results does not match";

    delete[] output_array;

    return 0;
}

// ----------------------------------------------------------

