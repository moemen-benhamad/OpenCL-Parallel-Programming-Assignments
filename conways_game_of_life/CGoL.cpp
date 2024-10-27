// ----------------------------------------------------------

#include <cstdio>
#include <cstdlib>

#include <iostream>
#include <fstream>
#include <sstream>

#include <thread>
#include <chrono>

#define ANSI_CLEAR_ON
#define DELAY_MS 500

using namespace std;

// ----------------------------------------------------------

// We provide a small library so we can easily setup OpenCL
#include "clutils.h"

// ----------------------------------------------------------

// Function to print the grid state
void printGrid(int* grid, int W, int H) {
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            cout << (grid[i * W + j] ? "o" : ".") << " "; // 'O' for alive, '.' for dead
        }
        cout << endl;
    }
    cout << endl;
}

int main(int argc,char **argv)
{
    const char *clu_File = SRC_PATH "CGoL.cl"; 

    // Initialize OpenCL
    cluInit();

    // After this call you have access to
    // clu_Context;      <= OpenCL context (pointer)
    // clu_Devices;      <= OpenCL device list (vector)
    // clu_Queue;        <= OpenCL queue (pointer)

    // Load Kernel
    cl::Program *program = cluLoadProgram(clu_File);
    cl::Kernel  *kernel = cluLoadKernel(program, "game_of_life");

    // allocate memory and opencl buffer
    const int W = 5;
    const int H = 5;
    const int size = W * H;

    cl::Buffer current_grid_buffer(*clu_Context, CL_MEM_READ_WRITE, size * sizeof(int));
    cl::Buffer next_grid_buffer(*clu_Context, CL_MEM_READ_WRITE, size * sizeof(int));

    // Allocate grid
    int* grid = new int[size];
    #define GRID(i, j) grid[(i) * W + (j)]

    // Init current grid with zeros
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            GRID(i, j) = 0;
        }
    }

    // Init for Blinker example
    // GRID(1, 2) = 1; GRID(2, 2) = 1; GRID(3, 2) = 1;

    // Init for Glider example
    GRID(1, 2) = 1; GRID(2, 3) = 1; GRID(3, 1) = 1; GRID(3, 2) = 1; GRID(3, 3) = 1;

    // Write grid to buffer
    clu_Queue->enqueueWriteBuffer(current_grid_buffer, true, 0, size * sizeof(int), grid);

    // Number of generations to evolve
    const int N = 20;

    // Print the first generation (init)
    cout << "Generation 0 " << ":" << endl;
    printGrid(grid, W, H);
    std::this_thread::sleep_for(std::chrono::milliseconds(DELAY_MS));

    // Evolve for N generations
    for (int gen = 0; gen < N; ++gen) {

        kernel->setArg(0, current_grid_buffer);
        kernel->setArg(1, next_grid_buffer);
        kernel->setArg(2, W);
        kernel->setArg(3, H);
        kernel->setArg(4, 1); // '0' for non periodic, '1' for periodic

        cl_int clerr = clu_Queue->enqueueNDRangeKernel(*kernel, cl::NullRange, cl::NDRange(size), cl::NullRange);
        cluCheckError(clerr, "Error running the kernel");

        // Read back the updated next grid from the device to host memory
        clu_Queue->enqueueReadBuffer(next_grid_buffer, CL_TRUE, 0, size * sizeof(int), grid);

        // Print the current generation
        #ifdef ANSI_CLEAR_ON
        cout << "\033[H\033[J"; // ANSI escape codes to clear the screen and move cursor to home position
        #endif
        cout << "Generation " << (gen + 1) << ":" << endl;
        printGrid(grid, W, H);

        // Swap the buffers for the next generation
        std::swap(current_grid_buffer, next_grid_buffer);

        // Delay
        std::this_thread::sleep_for(std::chrono::milliseconds(DELAY_MS));
    }

    // Cleanup
    delete[] grid;

    return 0;
}

// ----------------------------------------------------------

