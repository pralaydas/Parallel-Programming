#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__
int getGlobalIdx(){

    int block_Id = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    int thread_Id = block_Id * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    return thread_Id;
}

__global__
void process_kernel1(float *input1, float *input2, float *output, int datasize){

    int id = getGlobalIdx();

    if (id < datasize)
    {
        output[id] = sinf(input1[id]) + cosf(input2[id]);
    }
}

__global__
void process_kernel2(float *input, float *output, int datasize){

    int id = getGlobalIdx();

    if (id < datasize)
    {
        output[id] = logf(input[id]);
    }
}

__global__
void process_kernel3(float *input, float *output, int datasize){

    int id = getGlobalIdx();

    if (id < datasize)
    {
        output[id] = sqrtf(input[id]);
    }
}

void memAlloc(float** ptr, int size){
    if (cudaMalloc((void**)ptr, size) != cudaSuccess){
        fprintf(stderr,"Error in memory allocation");
        exit(EXIT_FAILURE);
    }
}

void kernel_launch(float**Z, float** output, int size){
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        fprintf(stderr,"Error in launching process_kernel");
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(*Z, *output, size, cudaMemcpyDeviceToHost);
    if(err != cudaSuccess){
        fprintf(stderr,"Error in copying output from device to host");
        exit(EXIT_FAILURE);
    }
}

void hostExecute(float* X, float* Y, float* Z, int n){

    int i;
    int flag = 0;
    int size = sizeof(float)*n;
    float* in1 = NULL; float* in2 = NULL;
    float* out1 = NULL; float* out2 = NULL; float* out3 = NULL;

    memAlloc(&in1, size);
    memAlloc(&in2, size);
    memAlloc(&out1, size);
    memAlloc(&out2, size);
    memAlloc(&out3, size);

    if(cudaMemcpy(in1, X, size, cudaMemcpyHostToDevice) != cudaSuccess){
      fprintf(stderr,"Error in copying input1 from host to device");
      exit(EXIT_FAILURE);
    }

    if(cudaMemcpy(in2, Y, size, cudaMemcpyHostToDevice) != cudaSuccess){
      fprintf(stderr,"Error in copying input2 from host to device");
      exit(EXIT_FAILURE);
    }

    // Launching Kernel1

    dim3 grid1(4, 2, 2);
    dim3 block1(32, 32, 1);
    process_kernel1<<<grid1,block1>>>(in1, in2, out1, n);
    kernel_launch(&Z, &out1, size);

    if(flag){
        i = 0;
        while(i < n){
            printf("%.2f ", Z[i++]);
        }
    }
    // Ending Kernel1

    // Launching Kernel2

    dim3 grid2(2, 8, 1);
    dim3 block2(8, 8, 16);
    process_kernel2<<<grid2,block2>>>(out1, out2, n);
    kernel_launch(&Z, &out2, size);

    if(flag){
        i = 0;
        while(i < n){
            printf("%.2f ", Z[i++]);
        }
    }
    // Ending Kernel2

    // Launching Kernel3

    dim3 grid3(16,1,1);
    dim3 block3(128,8,1);
    process_kernel3<<<grid3,block3>>>(out2, out3, n);
    kernel_launch(&Z, &out3, size);

    i = 0;
    while(i < n){
        printf("%.2f ", Z[i++]);
    }
    // Ending Kernel3

    cudaFree(in1); cudaFree(in2);
    cudaFree(out1); cudaFree(out2); cudaFree(out3);
}

int main(){

    float* X, *Y, *Z;
    int i;
    int num = (4*2*2)*(32*32*1);
    int size = sizeof(float)*num;
    X = (float*)malloc(size);
    Y = (float*)malloc(size);
    Z = (float*)malloc(size);

    if (X == NULL || Y == NULL || Z == NULL){
        fprintf(stderr, "Error in allocating memory to input and output vectors");
        exit(EXIT_FAILURE);
    }

    i = 0;
    while(i < num){
        scanf("%f", &X[i++]);
    }

    i = 0;
    while(i < num){
        scanf("%f", &Y[i++]);
    }

    hostExecute(X, Y, Z, num);

    return 0;
}

