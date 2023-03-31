#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>
#include <time.h>
#include <device_launch_parameters.h>

#define BLOCK_WIDTH 16

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
void matMulCPU(float *M, float *N, float *P, int matrixSize){
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


int main()
{
	// Sizes of input matrices to test
	int sizes[] = { 125, 250, 500, 1000, 2000 };
	int num_sizes = sizeof(sizes) / sizeof(int);

	// Loop over matrix sizes
	for (int i = 0; i < num_sizes; i++)
	{
		int size = sizes[i];
		int num_elements = size * size;

		// Allocate memory for input matrices on host
		float *h_M, *h_N;
		cudaMallocHost(&h_M, num_elements * sizeof(float));
		cudaMallocHost(&h_N, num_elements * sizeof(float));

		srand(time(NULL));
		for (int i = 0; i < size * size; i++) {
			h_M[i] = (float)rand() / RAND_MAX;
			h_N[i] = (float)rand() / RAND_MAX;
		}

		// Allocate memory for input matrices on device
		float *d_M, *d_N;
		cudaMalloc(&d_M, num_elements * sizeof(float));
		cudaMalloc(&d_N, num_elements * sizeof(float));

		// Create events to measure time
		cudaEvent_t start1, stop1, start2, stop2;
		cudaEventCreate(&start1);
		cudaEventCreate(&stop1);
		cudaEventCreate(&start2);
		cudaEventCreate(&stop2);

		// Copy input matrices from host to device and measure time
		cudaEventRecord(start1);
		cudaMemcpy(d_M, h_M, num_elements * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_N, h_N, num_elements * sizeof(float), cudaMemcpyHostToDevice);
		cudaEventRecord(stop1);
		cudaEventSynchronize(stop1);
		float transfer_time = 0;
		cudaEventElapsedTime(&transfer_time, start1, stop1);
		printf("Matrix size %d x %d: Host to device transfer time = %f ms\n", size, size, transfer_time);

		// Copy input matrices from device to host and measure time
		cudaEventRecord(start1);
		cudaMemcpy(h_M, d_M, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_N, d_N, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
		cudaEventRecord(stop1);
		cudaEventSynchronize(stop1);
		transfer_time = 0;
		cudaEventElapsedTime(&transfer_time, start1, stop1);
		printf("Matrix size %d x %d: Device to host transfer time = %f ms\n", size, size, transfer_time);

		// Free memory
		cudaFreeHost(h_M);
		cudaFreeHost(h_N);
		cudaFree(d_M);
		cudaFree(d_N);
	}

	return 0;
}
