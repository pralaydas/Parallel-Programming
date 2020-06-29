#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ int between(int Value, int leftInc, int rightExc);
__global__ void convolution_kernel(float* IN, float* KERNEL, int Size, float* OUT);
void host_execute(float* IN, float* KERNEL, int Size, float* OUT);

int main(int argc, char* argv[])
{
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

        float kernel[] = {1, 1, 1,
                          1, 1, 1,
                          1, 1, 1};

        float* out = (float*)malloc(n*n * sizeof(float));
        if (out == NULL) {
            fprintf(stderr, "Unable to Allocate Memory to Host Arrays\n");
            exit(EXIT_FAILURE);
        }

        host_execute(arr, kernel, n, out);

        for(int i = 0 ; i < n ; i++){
            for(int j = 0; j < n; j++)
                printf("%.2f ", out[i*n + j]);
            printf("\n");
        }

        free(arr), free(out);
    }
}

void host_execute(float* in, float* kernel, int n, float* out)
{
    dim3 grid(n,n,1);
    dim3 block(3,3,1);

    int sizeInput  = n*n * sizeof(float);
    int sizeKernel = 3*3 * sizeof(float);
    int sizeOutput = n*n * sizeof(float);

    cudaError_t err;

    float* IN = NULL;
    if ((err = cudaMalloc((void**)&IN, sizeInput)) != cudaSuccess){
        fprintf(stderr, "Error in Allocating Memory to IN at Device: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    if((err = cudaMemcpy(IN, in, sizeInput, cudaMemcpyHostToDevice)) != cudaSuccess){
        fprintf(stderr, "Error in Copy in(host) to IN(device): %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float* KERNEL = NULL;
    if((err = cudaMalloc((void**)&KERNEL, sizeKernel)) != cudaSuccess){
        fprintf(stderr, "Error in Allocating Memory to KERNEL at Device: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    if((err = cudaMemcpy(KERNEL, kernel, sizeKernel, cudaMemcpyHostToDevice)) != cudaSuccess){
        fprintf(stderr, "Error in Copy kernel(host) to KERNEL(device): %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float* OUT = NULL;
    if((err = cudaMalloc((void**)&OUT, sizeOutput)) != cudaSuccess){
        fprintf(stderr, "Error in Allocating Memory to OUT at Device: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    convolution_kernel<<<grid,block>>>(IN, KERNEL, n, OUT);
    if ((err = cudaGetLastError()) != cudaSuccess) {
        fprintf(stderr, "Error in executing Convolution Kernel: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    if((err = cudaMemcpy(out, OUT, sizeOutput, cudaMemcpyDeviceToHost)) != cudaSuccess){
        fprintf(stderr, "Error in Copy OUT(device) to out(host): %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaFree(IN), cudaFree(KERNEL), cudaFree(OUT);
}

__device__
int between(int value, int left, int right)
{
    return (left <= value && value < right);
}

__global__
void convolution_kernel(float* arr, float* kernel, int n, float* out)
{
    int i  = blockIdx.y;  int j  = blockIdx.x;
    int ii = threadIdx.y; int jj = threadIdx.x;
    int bid = i * gridDim.x + j;
    int tid = ii * blockDim.x + jj;

    if(between(i,0,n) && between(j,0,n) && between(tid,0,9))
    {
        int iii = i-1 + ii;
        int jjj = j-1 + jj;

        __shared__ float temp[9];
        if(between(iii,0,n) && between(jjj,0,n))
        {
            temp[tid] = kernel[tid] * arr[iii * n + jjj];
        }
        else
        {
            temp[tid] = 0;
        }
        __syncthreads();
        // Simple Parallel Reduction to get sum of 9 element array.
        // Can also do it serially in tid(0) as commented below.
        for(int iter = 1; iter <= 4; iter++)            
        {
            if( (tid % (1<<iter)) == (1<<(iter-1)) )
            {
                temp[ tid - (1<<(iter-1)) ] += temp[tid];
            }
            __syncthreads();
        }

        if (tid == 0)
        {
            // for(int it = 1; it < 8; it++) temp[0] += temp[it];
            // temp[0] += temp[8];
            out[i * n + j] = temp[0] / 9.0;
        }
    }
}
