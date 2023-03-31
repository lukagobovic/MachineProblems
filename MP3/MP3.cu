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


int main()
{
	// Sizes of input matrices to test
	int sizeOfBlock = 16;
	int sizes[] = { 128, 256, 250, 1024, 2048 };
	int blockSizes[] = { 2,4,10,20,25 };
	//sizeOfBlock = blockSizes[4];
	printf("Block width of: %d\n", sizeOfBlock);

	// Loop over matrix sizes
	//for (int x = 0; x < 5; ++x) {
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
			float* h_C = (float*)malloc(hostSize);
			float* h_P = (float*)malloc(hostSize);


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

			// Copy input matrices from host to device and measure time
			//cudaEventRecord(start1);
			cudaMemcpy(d_M, h_M, hostSize, cudaMemcpyHostToDevice);
			cudaMemcpy(d_N, h_N, hostSize, cudaMemcpyHostToDevice);
			// cudaEventRecord(stop1);
			// cudaEventSynchronize(stop1);
			// float transfer_time = 0;
			// cudaEventElapsedTime(&transfer_time, start1, stop1);
			// printf("Matrix size %d x %d: Host to device transfer time = %f ms\n", size, size, transfer_time);

			int n_blocks = ceil(size / BLOCK_WIDTH);

			dim3 threadsPerBlock(sizeOfBlock, sizeOfBlock);
			dim3 numberOfBlocks(n_blocks, n_blocks);
			//intf("%d\n", ceil((size + threadsPerBlock.x - 1) / threadsPerBlock.x));
			//dim3 numberOfBlocks(ceil((size + threadsPerBlock.x - 1) / threadsPerBlock.x), ceil((size + threadsPerBlock.y - 1) / threadsPerBlock.y));


			// //Part 2 ---------------------------------------------------------------------
			cudaEventRecord(start2, 0);
			matrixMultiplication << <numberOfBlocks, threadsPerBlock >> >(d_N, d_M, d_C, size);
			cudaEventRecord(stop2, 0);
			cudaEventSynchronize(stop2);
			cudaEventElapsedTime(&gpu_time2, start2, stop2);
			cudaMemcpy(h_C, d_C, hostSize, cudaMemcpyDeviceToHost);
			printf("Normal Multiplication time: %0.2f\n", gpu_time2);



			//cudaEventRecord(start1, 0);
			//matMulCPU(h_M, h_N, h_P, size);
			//cudaEventRecord(stop1, 0);

			//cudaEventElapsedTime(&gpu_time1, start1, stop1);
			//printf("Host Multiplication time: %0.2f\n\n", gpu_time1);

			//-----------------------------------------------------------------------------

			// Copy input matrices from device to host and measure time
			// cudaEventRecord(start1);
			// cudaMemcpy(h_M, d_M, hostSize, cudaMemcpyDeviceToHost);
			// cudaMemcpy(h_N, d_N, hostSize, cudaMemcpyDeviceToHost);
			// cudaEventRecord(stop1);
			// cudaEventSynchronize(stop1);
			// transfer_time = 0;
			// cudaEventElapsedTime(&transfer_time, start1, stop1);
			// printf("Matrix size %d x %d: Device to host transfer time = %f ms\n", size, size, transfer_time);

			// Free memory
			cudaFreeHost(h_M);
			cudaFreeHost(h_N);
			cudaFreeHost(h_C);
			cudaFreeHost(h_P);
			cudaFree(d_M);
			cudaFree(d_N);
			cudaFree(d_C);
		}
	//}
	return 0;
}
