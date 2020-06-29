#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__
void reduction_kernel(float* In, int n, float* Out, int m, int q)
{
    int blockNum = blockIdx.y * gridDim.x + blockIdx.x;
    int blockOff = blockNum * blockDim.x;
    int tid = threadIdx.x;

    if (blockNum < m && (blockOff + tid) < n)
    {
        for(int i = 1; i <= q; i++)
        {
            if((tid % (1<<i)) == (1<<(i-1)))
            {
                In[blockOff + tid - (1<<(i-1))] += In[blockOff + tid];
            }
            __syncthreads();
        }
        if (tid == 0)
        {
            Out[blockNum] = In[blockOff]/(1<<q);
        }
    }
}

void host_execute(float* a, int p, float* b, int q)
{
    long long n = 1<<p;
    long long k = 1<<q;
    long long m = 1<<(p%q);

    float* IN = NULL;
    for(long long i = n; i>=k ; i/=k)
    {
        dim3 grid(ceil(sqrt(i/k)), ceil(sqrt(i/k)), 1);
        dim3 block(k, 1, 1);

        long long sizeInput = (i)*sizeof(float);
        long long sizeOutput = (i/k)*sizeof(float);

        if (cudaMalloc((void**)&IN, sizeInput) != cudaSuccess) {
            fprintf(stderr, "Error in Allocating Memory to IN at Device");
            exit(EXIT_FAILURE);
        }
        if (cudaMemcpy(IN, a, sizeInput, cudaMemcpyHostToDevice) != cudaSuccess) {
            fprintf(stderr, "Error in Copy a(host) to IN(device)");
            exit(EXIT_FAILURE);
        }

        float* OUT = NULL;
        if (cudaMalloc((void**)&OUT, sizeOutput) != cudaSuccess) {
            fprintf(stderr, "Error in Allocatinf Memory to OUT at Device");
            exit(EXIT_FAILURE);
        }

        reduction_kernel<<<grid,block>>>(IN, i, OUT, i/k, q);
        if (cudaGetLastError() != cudaSuccess) {
            fprintf(stderr, "Error in executing Reduction Kernel");
            exit(EXIT_FAILURE);
        }

        /*
        free(a);
        if ((a = (float*)malloc(sizeOutput)) == NULL) {
            fprintf(stderr, "Error in Reallocating Memory to a");
            exit(EXIT_FAILURE);
        }
        */
        if (cudaMemcpy(a, OUT, sizeOutput, cudaMemcpyDeviceToHost) != cudaSuccess) {
            fprintf(stderr, "Error in Copy OUT(device) to a(host)");
            exit(EXIT_FAILURE);
        }

        cudaFree(IN), cudaFree(OUT);
    }
    memcpy(b, a, m*sizeof(float));
}


int main()
{
    int t; scanf("%d", &t);
    while(t--)
    {
        int p, q; scanf("%d %d", &p, &q);

        /*
        if(p%2 - q%2) {
            fprintf(stderr, "p and q should both be even or odd\n");
            exit(EXIT_FAILURE);
        }
        */

        long long n = 1<<p;
        long long m = 1<<(p%q);

        float* a = (float*)malloc(n*sizeof(float));
        float* b = (float*)malloc(m*sizeof(float));
        if (a == NULL || b == NULL) {
            fprintf(stderr, "Unable to Allocate Memory to Host Arrays");
            exit(EXIT_FAILURE);
        }
        for(int i = 0; i < n; i++) scanf("%f", &a[i]);

        host_execute(a, p, b, q);

        for(int i = 0; i < m; i++) printf("%.2f ", b[i]); printf("\n");

        free(a), free(b);
    }
    return 0;
}