#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Warp Size - 32
// Shared Memory Banks - 32

// Exploit Unified Memory for these Variables whose values' are set in Host Side
__managed__ int MAX_THREADS;
__managed__ int SIZE_SHM;
__managed__ int PADDING;

__managed__ int TILE_HEIGHT;
#define TILE_WIDTH (2*TILE_HEIGHT)
#define BLOCK_WIDTH 32
#define BLOCK_HEIGHT (MAX_THREADS/BLOCK_WIDTH)

__device__ int between(int Value, int leftInc, int rightExc);
__global__ void transpose_kernel(float* IN, int Size, float* OUT);
void host_execute(float* IN, int Size, float* OUT);
void printDevProp ( cudaDeviceProp devProp );
void printShmProp ( cudaSharedMemConfig devShm );

int main ()
{
    cudaDeviceProp devp ;
    cudaGetDeviceProperties (&devp , 0);

    MAX_THREADS = (int)devp.maxThreadsPerBlock;
    SIZE_SHM = (int)devp.sharedMemPerBlock;

    cudaSharedMemConfig devShm;
    cudaDeviceGetSharedMemConfig (&devShm);
    if(devShm == cudaSharedMemBankSizeFourByte)
        PADDING = 4/sizeof(float);
    if(devShm == cudaSharedMemBankSizeEightByte)
        PADDING = 8/sizeof(float);

    TILE_HEIGHT = sqrt( (float)((0.5)*(SIZE_SHM/sizeof(float))) );
    if(TILE_HEIGHT * (TILE_WIDTH + PADDING) > SIZE_SHM) TILE_HEIGHT--;
    TILE_HEIGHT = TILE_HEIGHT - TILE_HEIGHT%16;     // To ensure TILE_WIDTH % 32 = 0

    // fprintf(stderr, "%d %d %d %d %d\n", MAX_THREADS, BLOCK_WIDTH, PADDING, TILE_HEIGHT, TILE_WIDTH);

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
    if(blockIdx.x + blockIdx.y + threadIdx.x + threadIdx.y == 0)
    {
        // printf("%d %d %d %d %d\n", MAX_THREADS, BLOCK_WIDTH, PADDING, TILE_HEIGHT, TILE_WIDTH);
    }
    extern __shared__ float tile[];

    int x_meshoff = blockIdx.x * TILE_WIDTH;
    int y_meshoff = blockIdx.y * TILE_HEIGHT;

    for(int x_tileoff = threadIdx.x; x_tileoff < TILE_WIDTH; x_tileoff += BLOCK_WIDTH)
    {
        for(int y_tileoff = threadIdx.y; y_tileoff < TILE_HEIGHT; y_tileoff += BLOCK_HEIGHT)
        {
            int x = x_meshoff + x_tileoff;
            int y = y_meshoff + y_tileoff;
            if(between(x,0,n) && between(y,0,n))
            {
                // tile[y_tileoff][x_tileoff] = IN[y*n + x];
                tile[y_tileoff * (TILE_WIDTH + PADDING) + x_tileoff] = IN[y*n + x];
            }
        }
    }
    __syncthreads();

    for(int x_tileoff = threadIdx.y; x_tileoff < TILE_WIDTH; x_tileoff += BLOCK_WIDTH)
    {
        for(int y_tileoff = threadIdx.x; y_tileoff < TILE_HEIGHT; y_tileoff += BLOCK_HEIGHT)
        {
            int x = x_meshoff + x_tileoff;
            int y = y_meshoff + y_tileoff;
            if(between(x,0,n) && between(y,0,n))
            {
                // OUT[x*n + y] = tile[y_tileoff][x_tileoff];
                OUT[x*n + y] = tile[y_tileoff * (TILE_WIDTH + PADDING) + x_tileoff];
            }
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

    transpose_kernel<<<grid, block, sizeof(float)*(TILE_HEIGHT*(TILE_WIDTH + PADDING))>>>(IN, n, OUT);
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
