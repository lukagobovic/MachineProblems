#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>
#include <time.h>
#include <device_launch_parameters.h>
#pragma nvcc-- ptxas - options = -v
#define BLOCK_WIDTH 16
#define TILE_WIDTH 2

__global__ void matrixMultiplication(float *M, float *N, float *P, int matrixSize)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if (row < matrixSize && col < matrixSize)
    {
        for (int k = 0; k < matrixSize; k++)
        {
            sum += M[row * matrixSize + k] * N[k * matrixSize + col];
        }
        P[row * matrixSize + col] = sum;
    }
}

// Function to perform matrix multiplication on CPU
void matMulCPU(float *M, float *N, float *P, int matrixSize)
{
    for (int i = 0; i < matrixSize; i++)
    {
        for (int j = 0; j < matrixSize; j++)
        {
            float Pvalue = 0;
            for (int k = 0; k < matrixSize; k++)
            {
                Pvalue += M[j * matrixSize + k] * N[k * matrixSize + i];
            }
            P[j * matrixSize + i] = Pvalue;
        }
    }
}
__global__ void tiled_matrix_multiply_boundaries(float *A, float *B, float *C, int m, int n, int p, int tile_size_x, int tile_size_y)
{
    // calculate the row and column indices of the current thread
    int row = blockIdx.y * tile_size_y + threadIdx.y;
    int col = blockIdx.x * tile_size_x + threadIdx.x;

    // allocate shared memory for the tile of A and B
    extern __shared__ float tile[];
    float *tile_A = &tile[0];
    float *tile_B = &tile[tile_size_x * tile_size_y];

    // initialize the tile of C to zero
    float tile_C = 0;

    // iterate over tiles of A and B
    for (int i = 0; i < (n + tile_size_x - 1) / tile_size_x; ++i)
    {
        // check if the current thread is within the bounds of A and B
        if (row < m && i * tile_size_x + threadIdx.x < n)
        {
            tile_A[threadIdx.y * tile_size_x + threadIdx.x] = A[row * n + i * tile_size_x + threadIdx.x];
        }
        else
        {
            tile_A[threadIdx.y * tile_size_x + threadIdx.x] = 0;
        }
        if (col < p && i * tile_size_x + threadIdx.y < n)
        {
            tile_B[threadIdx.y * tile_size_x + threadIdx.x] = B[(i * tile_size_x + threadIdx.y) * p + col];
        }
        else
        {
            tile_B[threadIdx.y * tile_size_x + threadIdx.x] = 0;
        }

        // synchronize threads to ensure tiles have been loaded
        __syncthreads();

        // perform the matrix multiplication for the current tile
        for (int j = 0; j < tile_size_x; ++j)
        {
            tile_C += tile_A[threadIdx.y * tile_size_x + j] * tile_B[j * tile_size_x + threadIdx.x];
        }

        // synchronize threads to ensure previous calculation has completed
        __syncthreads();
    }

    // write the result to global memory if within the bounds of C
    if (row < m && col < p)
    {
        C[row * p + col] = tile_C;
    }
}

__global__ void tiled_matrix_multiply(float *A, float *B, float *C, int n)
{
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float sum = 0;

    for (int i = 0; i < n / TILE_WIDTH; i++)
    {
        tileA[ty][tx] = A[row * n + (i * TILE_WIDTH + tx)];
        tileB[ty][tx] = B[(i * TILE_WIDTH + ty) * n + col];
        __syncthreads();

        for (int j = 0; j < TILE_WIDTH; j++)
        {
            sum += tileA[ty][j] * tileB[j][tx];
        }
        __syncthreads();
    }

    C[row * n + col] = sum;
}

int main()
{

    const int M_rows = 350;
    const int M_cols = 400;
    const int N_rows = 400;
    const int N_cols = 500;

    printf("Matrix size is %d by %d\n\n", size, size);
    size_t hostSize = size * size * sizeof(float);

    float gpu_time1 = 0.0f;
    float gpu_time2 = 0.0f;

    // Allocate memory for input matrices on host
    float *h_M = (float *)malloc(hostSize);
    float *h_N = (float *)malloc(hostSize);
    float *h_C_GPU = (float *)malloc(hostSize);
    float *h_C_CPU = (float *)malloc(hostSize);

    srand(time(NULL));
    for (int i = 0; i < size * size; i++)
    {
        h_M[i] = (float)rand() / RAND_MAX;
        h_N[i] = (float)rand() / RAND_MAX;
    }

    // Allocate memory for input matrices on device
    float *d_M, *d_N, *d_C;
    cudaMalloc(&d_M, hostSize);
    cudaMalloc(&d_N, hostSize);
    cudaMalloc(&d_C, hostSize);

    // Create events to measure time
    cudaEvent_t start1, stop1, start2, stop2;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    // Host Multiplication
    // cudaEventRecord(start1, 0);
    // matMulCPU(h_M, h_N, h_C_CPU, size);
    // cudaEventRecord(stop1, 0);

    // cudaEventElapsedTime(&gpu_time1, start1, stop1);
    // printf("Host Multiplication time: %0.2f\n", gpu_time1);

    // Copy input matrices from host to device and measure time
    // cudaEventRecord(start1);
    cudaMemcpy(d_M, h_M, hostSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, hostSize, cudaMemcpyHostToDevice);
    // cudaEventRecord(stop1);
    // cudaEventSynchronize(stop1);
    // float transfer_time = 0;
    // cudaEventElapsedTime(&transfer_time, start1, stop1);
    // printf("Matrix size %d x %d: Host to device transfer time = %f ms\n", size, size, transfer_time);

    int NumBlocks = size / sizeOfBlock;
    if (size % sizeOfBlock)
        NumBlocks++;
    // dim3 numberOfBlocks(NumBlocks, NumBlocks);
    // dim3 threadsPerBlock(sizeOfBlock, sizeOfBlock);

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 numberOfBlocks(size / TILE_WIDTH, size / TILE_WIDTH);

    // //Part 2 ---------------------------------------------------------------------
    cudaEventRecord(start2, 0);
    tiled_matrix_multiply<<<numberOfBlocks, threadsPerBlock>>>(d_M, d_N, d_C, size);
    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&gpu_time2, start2, stop2);
    cudaMemcpy(h_C_GPU, d_C, hostSize, cudaMemcpyDeviceToHost);
    printf("Normal Multiplication time: %0.2f\n", gpu_time2);

    // int passedFlag = 1;
    // for (int i = 0; i < size * size; i++) {
    //	if (abs(h_C_CPU[i] - h_C_GPU[i]) > 0.001) {
    //		passedFlag = 0;
    //		break;
    //	}
    // }
    // if (passedFlag){
    //	printf("Test PASSED\n\n");
    // }
    // else {
    //	printf("Test FAILED\n");
    // }

    //-----------------------------------------------------------------------------

    // Copy input matrices from device to host and measure time
    // cudaEventRecord(start1);
    cudaMemcpy(h_M, d_M, hostSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_N, d_N, hostSize, cudaMemcpyDeviceToHost);
    // cudaEventRecord(stop1);
    // cudaEventSynchronize(stop1);
    // transfer_time = 0;
    // cudaEventElapsedTime(&transfer_time, start1, stop1);
    // printf("Matrix size %d x %d: Device to host transfer time = %f ms\n", size, size, transfer_time);

    // Free memory
    cudaFreeHost(h_M);
    cudaFreeHost(h_N);
    cudaFreeHost(h_C_CPU);
    cudaFreeHost(h_C_GPU);
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_C);

    //}
    printf("\n");
}
return 0;
}