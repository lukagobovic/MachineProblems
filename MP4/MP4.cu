//Luka Gobovic
//20215231
//MP4 Part 1
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>
#include <time.h>
#include <device_launch_parameters.h>

#define TILE_WIDTH 2

// Function to perform matrix multiplication on CPU
void matMulCPU(float *M, float *N, float *P, int matrixSize) {
	for (int i = 0; i < matrixSize; i++) {
		for (int j = 0; j < matrixSize; j++) {
			float Pvalue = 0;
			for (int k = 0; k < matrixSize; k++) {
				Pvalue += M[j * matrixSize + k] * N[k * matrixSize + i];
			}
			P[j * matrixSize + i] = Pvalue;
		}
	}
}

//Tiled Matrix Multiplication
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

	for (int i = 0; i < n / TILE_WIDTH; i++) {
		tileA[ty][tx] = A[row * n + (i * TILE_WIDTH + tx)];
		tileB[ty][tx] = B[(i * TILE_WIDTH + ty) * n + col];
		__syncthreads();

		for (int j = 0; j < TILE_WIDTH; j++) {
			sum += tileA[ty][j] * tileB[j][tx];
		}
		__syncthreads();
	}

	C[row * n + col] = sum;
}

int main()
{
	// Sizes of input matrices to test
	int sizes[] = { 125, 250, 500, 1000, 2000 };
	printf("Tile width of: %d\n", TILE_WIDTH);

	for (int i = 0; i < 5; i++)
	{
		int size = sizes[i];
		printf("Matrix size is %d by %d\n\n", size, size);
		size_t hostSize = size * size * sizeof(float);

		float gpu_time1 = 0.0f;
		float gpu_time2 = 0.0f;

		// Allocate memory for input matrices on host
		float* h_M = (float*)malloc(hostSize);
		float* h_N = (float*)malloc(hostSize);
		float* h_C_GPU = (float*)malloc(hostSize);
		float* h_C_CPU = (float*)malloc(hostSize);

		srand(time(NULL));
		for (int i = 0; i < size * size; i++) {
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

		//Host Multiplication
		cudaEventRecord(start1, 0);
		matMulCPU(h_M, h_N, h_C_CPU, size);
		cudaEventRecord(stop1, 0);

		cudaEventElapsedTime(&gpu_time1, start1, stop1);
		printf("Host Multiplication time: %0.2f\n", gpu_time1);

		cudaMemcpy(d_M, h_M, hostSize, cudaMemcpyHostToDevice);
		cudaMemcpy(d_N, h_N, hostSize, cudaMemcpyHostToDevice);

		dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
		dim3 numberOfBlocks(size / TILE_WIDTH, size / TILE_WIDTH);

		//Tiled Mul ---------------------------------------------------------------------
		cudaEventRecord(start2, 0);
		tiled_matrix_multiply << < numberOfBlocks, threadsPerBlock >> >(d_M, d_N, d_C, size);
		cudaEventRecord(stop2, 0);
		cudaEventSynchronize(stop2);
		cudaEventElapsedTime(&gpu_time2, start2, stop2);
		cudaMemcpy(h_C_GPU, d_C, hostSize, cudaMemcpyDeviceToHost);
		printf("Normal Multiplication time: %0.2f\n", gpu_time2);

		for (int i = 0; i < size * size; i++) {
			if (abs(h_C_CPU[i] - h_C_GPU[i]) > 0.1) {
				printf("Test FAILED\n");
				break;
			}
		}
		printf("Test PASSED\n\n");

		cudaFreeHost(h_M);
		cudaFreeHost(h_N);
		cudaFreeHost(h_C_CPU);
		cudaFreeHost(h_C_GPU);
		cudaFree(d_M);
		cudaFree(d_N);
		cudaFree(d_C);
	}
	return 0;
}