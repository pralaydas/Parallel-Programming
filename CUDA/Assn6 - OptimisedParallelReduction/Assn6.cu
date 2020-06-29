#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <int MAX_THREADS> __global__  void reduction_kernel(float* IN, int Size, float* OUT);
__global__  void product_kernel(float* A, float* B, int Size);
__device__ int between(int Value, int leftInc, int rightExc);
__host__ float dot_product(float* A, float* B, int Size);

int MAX_THREADS;

int main()
{
    cudaDeviceProp devp;
    cudaGetDeviceProperties (&devp , 0);
    MAX_THREADS = (int)devp.maxThreadsPerBlock;

    int t; scanf("%d", &t);
    while(t--)
    {
        int n; scanf("%d", &n);

        float* a = (float*)malloc(n * sizeof(float));
        if (a == NULL) {
            fprintf(stderr, "Unable to Allocate Memory to Host Array - A\n");
            exit(EXIT_FAILURE);
        }
        float* b = (float*)malloc(n * sizeof(float));
        if (b == NULL) {
            fprintf(stderr, "Unable to Allocate Memory to Host Array - B\n");
            exit(EXIT_FAILURE);
        }
        for(int i = 0; i < n; i++) scanf("%f", &a[i]);
        for(int i = 0; i < n; i++) scanf("%f", &b[i]);

        printf("%.2f\n", dot_product(a, b, n));

        free(a), free(b);
    }
    return 0;
}

float dot_product(float* a, float* b, int n)
{
    dim3 grid1(n/MAX_THREADS,1,1);
    dim3 block1(MAX_THREADS,1,1);

    cudaError_t err;

    float* A = NULL;
    if((err = cudaMalloc((void**)&A, n*sizeof(float))) != cudaSuccess){
        fprintf(stderr, "Error in Allocating Memory to A at Device: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    if((err = cudaMemcpy(A, a, n*sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess){
        fprintf(stderr, "Error in Copy a(Host) to A(Device): %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float* B = NULL;
    if((err = cudaMalloc((void**)&B, n*sizeof(float))) != cudaSuccess){
        fprintf(stderr, "Error in Allocating Memory to B at Device: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    if((err = cudaMemcpy(B, b, n*sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess){
        fprintf(stderr, "Error in Copy b(Host) to B(Device): %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Get element-wise product calculated and stored in A as A[i] *= B[i] for i in range(n)
    product_kernel <<<grid1, block1>>> (A, B, n);
    if ((err = cudaGetLastError()) != cudaSuccess) {
        fprintf(stderr, "Error in executing Product Kernel: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaFree(B); 
    
    for( ; n > 1024; n/=MAX_THREADS)
    {   
        int NUM_BLOCKS = n/MAX_THREADS;
        dim3 grid(NUM_BLOCKS, 1, 1);
        dim3 block(MAX_THREADS, 1, 1);

        float* TEMP = NULL;
        if((err = cudaMalloc((void**)&TEMP, NUM_BLOCKS*sizeof(float))) != cudaSuccess){
            fprintf(stderr, "Error in Allocating Memory to TEMP at Device: %s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
     
        // fprintf(stderr, "N - %d\n", n);
        // fprintf(stderr, "Grid - (%d %d %d)\n", grid.x, grid.y, grid.z);
        // fprintf(stderr, "Block - (%d %d %d)\n", block.x, block.y, block.z);

        switch(MAX_THREADS)
        {
            case 1024: reduction_kernel <1024> <<<grid, block, 1024*sizeof(float)>>> (A, n, TEMP); break;
            case  512: reduction_kernel < 512> <<<grid, block,  512*sizeof(float)>>> (A, n, TEMP); break;
            case  256: reduction_kernel < 256> <<<grid, block,  256*sizeof(float)>>> (A, n, TEMP); break;
            case  128: reduction_kernel < 128> <<<grid, block,  128*sizeof(float)>>> (A, n, TEMP); break;
            case   64: reduction_kernel <  64> <<<grid, block,   64*sizeof(float)>>> (A, n, TEMP); break;
            case   32: reduction_kernel <  32> <<<grid, block,   32*sizeof(float)>>> (A, n, TEMP); break;
            case   16: reduction_kernel <  16> <<<grid, block,   16*sizeof(float)>>> (A, n, TEMP); break;
            case    8: reduction_kernel <   8> <<<grid, block,    8*sizeof(float)>>> (A, n, TEMP); break;
            case    4: reduction_kernel <   4> <<<grid, block,    4*sizeof(float)>>> (A, n, TEMP); break;
            case    2: reduction_kernel <   2> <<<grid, block,    2*sizeof(float)>>> (A, n, TEMP); break;
            case    1: reduction_kernel <   1> <<<grid, block,    1*sizeof(float)>>> (A, n, TEMP); break;
        }
        if ((err = cudaGetLastError()) != cudaSuccess) {
            fprintf(stderr, "Error in executing Reduction Kernel: %s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        if((err = cudaMemcpy(A, TEMP, NUM_BLOCKS*sizeof(float), cudaMemcpyDeviceToDevice)) != cudaSuccess){
            fprintf(stderr, "Error in Copy TEMP(Device) to A(Device): %s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        cudaFree(TEMP);
    }

    float* temp = (float*)malloc(n*sizeof(float));
    if(temp == NULL){
        fprintf(stderr, "Unable to Allocate Memory to Host Array - TEMP\n");
        exit(EXIT_FAILURE);
    }
    if((err = cudaMemcpy(temp, A, n*sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess){
        fprintf(stderr, "Error in Copy A(Device) to temp(Host): %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaFree(A);

    float sum = 0;
    for(int i = 0; i < n; i++) sum += temp[i];
    free(temp);

    return sum;
}

__global__
void product_kernel(float* A, float* B, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n)
    {
        A[tid] *= B[tid];
    }
}

template<int MAX_THREADS>
__global__ void reduction_kernel(float* IN, int n, float* OUT)
{
    extern __shared__ volatile float partial[];
    
    int tid = threadIdx.x;
    int block_off = blockIdx.x * blockDim.x;

    partial[tid] = IN[block_off + tid];
    
    __syncthreads();

    if(MAX_THREADS >= 1024){
        if(tid < 512) partial[tid] += partial[tid + 512]; __syncthreads();
    }
    if(MAX_THREADS >= 512){
        if(tid < 256) partial[tid] += partial[tid + 256]; __syncthreads();
    }
    if(MAX_THREADS >= 256){
        if(tid < 128) partial[tid] += partial[tid + 128]; __syncthreads();
    }
    if(MAX_THREADS >= 128){
        if(tid <  64) partial[tid] += partial[tid +  64]; __syncthreads();        
    }
    
    // No need of thread sync since only one warp executes here (hence in locksteps)
    if(tid < 32)
    {
        if(MAX_THREADS >= 64) partial[tid] += partial[tid +  32];
        if(MAX_THREADS >= 32) partial[tid] += partial[tid +  16];
        if(MAX_THREADS >= 16) partial[tid] += partial[tid +   8];
        if(MAX_THREADS >=  8) partial[tid] += partial[tid +   4];
        if(MAX_THREADS >=  4) partial[tid] += partial[tid +   2];
        if(MAX_THREADS >=  2) partial[tid] += partial[tid +   1];
    }
    
    if(tid == 0){
        OUT[blockIdx.x] = partial[0];   // Stores Partial Sum of this Block Reduction in OUT
    }
}
