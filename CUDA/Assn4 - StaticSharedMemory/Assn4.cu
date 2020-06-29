#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Warp Size - 32
// Shared Memory Banks - 32
// Shared Memory Bank Width - 4 bytes (sizeof(float))

#define MAX_THREADS 1024            // Found out before hand on system locally and COLAB
#define PADDING 1                   // Found out before hand from system locally and COLAB

#define TILE_HEIGHT 64              // Highest Multiple of 16 which will lie within 48KB total size
#define TILE_WIDTH (2*TILE_HEIGHT)  // Mentioned in Question
#define BLOCK_WIDTH 32
#define BLOCK_HEIGHT (MAX_THREADS/BLOCK_WIDTH)

__device__ int between(int Value, int leftInc, int rightExc);
__global__ void transpose_kernel(float* IN, int Size, float* OUT);
void host_execute(float* IN, int Size, float* OUT);
void printDevProp ( cudaDeviceProp devProp );
void printShmProp ( cudaSharedMemConfig devShm );

int main ()
{
    int devCount;
    cudaGetDeviceCount (&devCount);
    for (int i = 0; i < devCount ; ++i)
    {
        cudaDeviceProp devp ;
        cudaGetDeviceProperties (&devp , i);
        // printDevProp ( devp );

        cudaSharedMemConfig devShm;
        cudaDeviceGetSharedMemConfig (&devShm);
        // printShmProp ( devShm );
    }

    int t; scanf("%d", &t);
    while(t--)
    {
        int n; scanf("%d", &n);
        float* arr = (float*)malloc(n*n * sizeof(float));
        if (arr == NULL) {
            fprintf(stderr, "Unable to Allocate Memory to Host Arrays\n");
            exit(EXIT_FAILURE);
        }
        for(int i = 0; i < n*n; i++) scanf("%f", &arr[i]);

        float* out = (float*)malloc(n*n * sizeof(float));
        if (out == NULL) {
            fprintf(stderr, "Unable to Allocate Memory to Host Arrays\n");
            exit(EXIT_FAILURE);
        }

        host_execute(arr, n, out);

        for(int i = 0 ; i < n ; i++){
            for(int j = 0; j < n; j++)
                printf("%.2f ", out[i*n + j]);
            printf("\n");
        }

        free(arr), free(out);
    }
    return 0;
}

__device__
int between(int value, int left, int right){
    return (left <= value && value < right);
}

__global__
void transpose_kernel(float* IN, int n, float* OUT)
{
    __shared__ float tile[TILE_HEIGHT][TILE_WIDTH + PADDING];

    int x_meshoff = blockIdx.x * TILE_WIDTH;                // Offset of tile in x-direction in IN[][]
    int y_meshoff = blockIdx.y * TILE_HEIGHT;               // Offset of tile in y-direction in IN[][]

    for(int x_tileoff = threadIdx.x; x_tileoff < TILE_WIDTH; x_tileoff += BLOCK_WIDTH)
    {
        for(int y_tileoff = threadIdx.y; y_tileoff < TILE_HEIGHT; y_tileoff += BLOCK_HEIGHT)
        {
            int x = x_meshoff + x_tileoff;                  // Offset of thread in tile in x-direction
            int y = y_meshoff + y_tileoff;                  // Offset of thread in tile in y-direction
            if(between(x,0,n) && between(y,0,n))
            {
                tile[y_tileoff][x_tileoff] = IN[y*n + x];   // Global Coalesced Load into Shared Memory
            }                                               // The column parameter x is governed by tid.x
        }
    }
    __syncthreads();

    for(int x_tileoff = threadIdx.y; x_tileoff < TILE_WIDTH; x_tileoff += BLOCK_WIDTH)
    {
        for(int y_tileoff = threadIdx.x; y_tileoff < TILE_HEIGHT; y_tileoff += BLOCK_HEIGHT)
        {
            int x = x_meshoff + x_tileoff;                  // Offset of thread in tile in x-direction
            int y = y_meshoff + y_tileoff;                  // Offset of thread in tile in y-direction
            if(between(x,0,n) && between(y,0,n))
            {
                OUT[x*n + y] = tile[y_tileoff][x_tileoff];  // Global Coalesced Store from Shared Memory
            }                                               // The column parameter y is governed by tid.x
        }
    }
    __syncthreads();
}

void host_execute (float* in, int n, float* out)
{
    // dim3 cuboid(x,y,z);
    dim3 grid((n+TILE_WIDTH-1)/TILE_WIDTH, (n+TILE_HEIGHT-1)/TILE_HEIGHT, 1);
    dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT , 1);

    int sizeInput  = n*n * sizeof(float);
    int sizeOutput = n*n * sizeof(float);

    cudaError_t err;

    float* IN = NULL;
    if((err = cudaMalloc((void**)&IN, sizeInput)) != cudaSuccess){
        fprintf(stderr, "Error in Allocating Memory to IN at Device: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    if((err = cudaMemcpy(IN, in, sizeInput, cudaMemcpyHostToDevice)) != cudaSuccess){
        fprintf(stderr, "Error in Copy in(host) to IN(device): %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float* OUT = NULL;
    if((err = cudaMalloc((void**)&OUT, sizeOutput)) != cudaSuccess){
        fprintf(stderr, "Error in Allocating Memory to OUT at Device: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // fprintf(stderr, "N - %d\n", n);
    // fprintf(stderr, "Grid - (%d %d %d)\n", grid.x, grid.y, grid.z);
    // fprintf(stderr, "Block - (%d %d %d)\n", block.x, block.y, block.z);

    transpose_kernel<<<grid,block>>>(IN, n, OUT);
    if ((err = cudaGetLastError()) != cudaSuccess) {
        fprintf(stderr, "Error in executing Transpose Kernel: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    if((err = cudaMemcpy(out, OUT, sizeOutput, cudaMemcpyDeviceToHost)) != cudaSuccess){
        fprintf(stderr, "Error in Copy OUT(device) to out(host): %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaFree(IN), cudaFree(OUT);
}

void printDevProp ( cudaDeviceProp devProp )
{
    fprintf (stderr, " Major revision number : %d\n",devProp.major );
    fprintf (stderr, " Minor revision number : %d\n",devProp.minor );
    fprintf (stderr, " Name : %s\n",devProp.name );
    fprintf (stderr, " Total global memory : %lu\n",devProp.totalGlobalMem );
    fprintf (stderr, " Total shared memory per block :%lu\n", devProp.sharedMemPerBlock );
    fprintf (stderr, " Total registers per block : %d\n", devProp.regsPerBlock );
    fprintf (stderr, " Warp size : %d\n",devProp.warpSize );
    fprintf (stderr, " Maximum memory pitch : %lu\n",devProp.memPitch );
    fprintf (stderr, " Maximum threads per block : %d\n",devProp.maxThreadsPerBlock );
    for (int i = 0; i < 3; ++i)
        fprintf (stderr, " Maximum dimension %d of block : %d\n",i, devProp.maxThreadsDim [i]);
    for (int i = 0; i < 3; ++i)
        fprintf (stderr, " Maximum dimension %d of grid : %d\n", i, devProp.maxGridSize [i]);

    fprintf (stderr, " Clock rate : %d\n",devProp.clockRate );
    fprintf (stderr, " Total constant memory :%lu\n", devProp.totalConstMem );
    fprintf (stderr, " Texture alignment : %lu\n", devProp.textureAlignment );
    fprintf (stderr, " Concurrent copy and execution : %s\n", ( devProp.deviceOverlap ? "Yes " : "No"));
    fprintf (stderr, " Number of multiprocessors : %d\n",devProp.multiProcessorCount);
}

void printShmProp ( cudaSharedMemConfig devShm )
{
    if(devShm == cudaSharedMemBankSizeFourByte)
        fprintf (stderr, "Bankwidth - 4bytes\n");
    if(devShm == cudaSharedMemBankSizeEightByte)
        fprintf (stderr, "Bankwidht - 8bytes\n");
}
