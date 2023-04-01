#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>
#include <time.h>
#include <device_launch_parameters.h>

#define BLOCK_WIDTH 16
#define TILE_WIDTH 2

__global__ void matrixMultiplication(float *M, float *N, float *P, int matrixSize) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0.0f;
	if (row < matrixSize && col < matrixSize) {
		for (int k = 0; k < matrixSize; k++) {
			sum += M[row * matrixSize + k] * N[k * matrixSize + col];
		}
		P[row * matrixSize + col] = sum;
	}
}

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
	int sizeOfBlock = 1;
	int sizes[] = { 125, 250, 500, 1000, 2000 };
	int blockSizes[] = { 2,4,10,20,25 };
	//sizeOfBlock = blockSizes[1];
	//printf("Tile width of: %d\n", TILE_WIDTH);

	// Loop over matrix sizes
	for (int x = 0; x < 5; x++) {
		sizeOfBlock = blockSizes[x];
		printf("Block width of: %d\n", sizeOfBlock);
		for (int i = 0; i < 5; i++)
		{
			int size = sizes[i];
			//sizeOfBlock = blockSizes[x];
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
			printf("test");
			matMulCPU(h_M, h_N, h_C_CPU, size);
			cudaEventRecord(stop1, 0);

			cudaEventElapsedTime(&gpu_time1, start1, stop1);
			printf("Host Multiplication time: %0.2f\n", gpu_time1);

			// Copy input matrices from host to device and measure time
			//cudaEventRecord(start1);
			cudaMemcpy(d_M, h_M, hostSize, cudaMemcpyHostToDevice);
			cudaMemcpy(d_N, h_N, hostSize, cudaMemcpyHostToDevice);
			// cudaEventRecord(stop1);
			// cudaEventSynchronize(stop1);
			// float transfer_time = 0;
			// cudaEventElapsedTime(&transfer_time, start1, stop1);
			// printf("Matrix size %d x %d: Host to device transfer time = %f ms\n", size, size, transfer_time);

			int NumBlocks = size / sizeOfBlock;
			if (size % sizeOfBlock) NumBlocks++;
			//dim3 numberOfBlocks(NumBlocks, NumBlocks);
			//dim3 threadsPerBlock(sizeOfBlock, sizeOfBlock);

			dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
			dim3 numberOfBlocks(size / TILE_WIDTH, size / TILE_WIDTH);


			// //Part 2 ---------------------------------------------------------------------
			cudaEventRecord(start2, 0);
			matrixMultiplication << < numberOfBlocks, threadsPerBlock >> >(d_M, d_N, d_C, size);
			cudaEventRecord(stop2, 0);
			cudaEventSynchronize(stop2);
			cudaEventElapsedTime(&gpu_time2, start2, stop2);
			cudaMemcpy(h_C_GPU, d_C, hostSize, cudaMemcpyDeviceToHost);
			printf("Normal Multiplication time: %0.2f\n", gpu_time2);

			//for (int i = 0; i < size * size; i++) {
			//	if (abs(h_C_CPU[i] - h_C_GPU[i]) > 0.1) {
			//		printf("Test FAILED\n");
			//		break;
			//	}
			//}
			//printf("Test PASSED\n\n");

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

		}
		printf("\n");
	}
	return 0;
}