#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_SAFECALL(call)                                                 \
    {                                                                       \
        call;                                                               \
        cudaError err = cudaGetLastError();                                 \
        if (cudaSuccess != err) {                                           \
            fprintf(                                                        \
                stderr,                                                     \
                "Cuda error in function '%s' file '%s' in line %i : %s.\n", \
                #call, __FILE__, __LINE__, cudaGetErrorString(err));        \
            fflush(stderr);                                                 \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    }


// CUDA kernel. Each thread takes care of one element of c
__global__ void vecAdd(double *a, double *b, double *c, int n) {
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n) c[id] = a[id] + b[id];
}

__global__ void vecMult(double *a, double *b, double *c, int n) {
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n) c[id] = a[id] * b[id];
}

int main(int argc, char *argv[]) {
    // Size of vectors
    int n = 4096;
    if (argc > 1) n = atoi(argv[1]);

    // Host input vectors
    double *h_a;
    double *h_b;
    double *h_c;
    double *h_d;
    // Host output vector
    double *h_e;

    // Device input vectors
    double *d_a;
    double *d_b;
    double *d_c;
    double *d_d;
    // Device output vector
    double *d_e;

    // Size, in bytes, of each vector
    size_t bytes = n * sizeof(double);

    // Allocate memory for each vector on host
    h_a = (double *)malloc(bytes);
    h_b = (double *)malloc(bytes);
    h_c = (double *)malloc(bytes);
    h_d = (double *)malloc(bytes);
    h_e = (double *)malloc(bytes);

    // Allocate memory for each vector on GPU
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMalloc(&d_d, bytes);
    cudaMalloc(&d_e, bytes);

    int i;
    // Initialize vectors on host
    for (i = 0; i < n; i++) {
        h_a[i] = sin(i);
        h_b[i] = cos(i);
        h_c[i] = 0;
        h_d[i] = 0;
        h_e[i] = 0;
    }

    // Copy host vectors to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, h_d, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_e, h_e, bytes, cudaMemcpyHostToDevice);

    int blockSize, gridSize;

    // Number of threads in each thread block
    blockSize = 1024;

    // Number of thread blocks in grid
    gridSize = (int)ceil((float)n / blockSize);

    // Execute the kernel
    CUDA_SAFECALL((vecMult<<<gridSize, blockSize>>>(d_a, d_a, d_c, n)));
    CUDA_SAFECALL((vecMult<<<gridSize, blockSize>>>(d_b, d_b, d_d, n)));
    CUDA_SAFECALL((vecAdd<<<gridSize, blockSize>>>(d_c, d_d, d_e, n)));

    // Copy array back to host
    cudaMemcpy(h_e, d_e, bytes, cudaMemcpyDeviceToHost);

    // Sum up vector c and print result divided by n, this should equal 1 within
    // error
    double sum = 0;
    for (i = 0; i < n; i++) sum += h_e[i];
    printf("Final sum = %f; sum/n = %f (should be ~1)\n", sum, sum / n);

    // Release device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_d);
    cudaFree(d_e);

    // Release host memory
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_d);
    free(h_e);

    return 0;
}
