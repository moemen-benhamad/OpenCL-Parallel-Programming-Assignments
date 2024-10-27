__kernel void game_of_life(__global int* current_grid, __global int* next_grid, const int W, const int H, int periodic) {

    int id = get_global_id(0);
    int i = id / W; // self row
    int j = id % W; // self column
    int live_neighbors = 0;

    // Iterate over all neighbors
    for (int di = -1; di <= 1; ++di) {
        for (int dj = -1; dj <= 1; ++dj) {
            if (di == 0 && dj == 0) continue;  // skip self
            
            if(!periodic) {  
                int ni = i + di; // neighbor row 
                int nj = j + dj; // neighbor column
                if (ni >= 0 && ni < H && nj >= 0 && nj < W){
                    live_neighbors += current_grid[ni * W + nj]; // Count live neighbors while checking for out-of-bounds conditions
                }
            }
            
            else {               
                int ni = (i + di + H) % H; // neighbor row (wrapped)
                int nj = (j + dj + W) % W; // neighbor column (wrapped)
                live_neighbors += current_grid[ni * W + nj]; // Count live neighbors (wrapped)
            }
        }
    }

    // Apply rules
    int current_state = current_grid[id];

    if (current_state == 1) {
        next_grid[id] = (live_neighbors == 2 || live_neighbors == 3) ? 1 : 0;
    } 
    
    else {
        next_grid[id] = (live_neighbors == 3) ? 1 : 0;
    }
}
