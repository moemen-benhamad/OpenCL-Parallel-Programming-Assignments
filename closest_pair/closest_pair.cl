#define SQUARE(x) ((x) * (x))

struct Point{
    int x, y;
};

__kernel void closest_pair(__global struct Point* a, __global float* b, const int N) {
    int id = get_global_id(0);

    int i = id / N;
    int j = id % N;

    if (i < j && i < N && j < N) {
        b[id] = sqrt((float)(SQUARE(a[i].x - a[j].x) + SQUARE(a[i].y - a[j].y)));
    } 
    
    else {
        b[id] = FLT_MAX;
    }
}