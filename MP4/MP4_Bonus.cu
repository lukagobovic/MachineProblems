//Luka Gobovic
//20215231
//MP4 Part 2 (BONUS)
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>
#include <time.h>
#include <device_launch_parameters.h>

#define BLOCK_WIDTH 16
#define TILE_WIDTH 8

//CPU Multiply
void matrix_multiply(float *A, float *B, float *C, int m, int n, int p) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < p; j++) {
			float sum = 0;
			for (int k = 0; k < n; k++) {
				sum += A[i * n + k] * B[k * p + j];
			}
			C[i * p + j] = sum;
		}
	}
}

__global__ void tiled_matrix_multiply(float *A, float *B, float *C, int m, int n, int p, int tile_size_x, int tile_size_y) {
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
		if (row < m && i * tile_size_x + threadIdx.x < n) {
			tile_A[threadIdx.y * tile_size_x + threadIdx.x] = A[row * n + i * tile_size_x + threadIdx.x];
		}
		else {
			tile_A[threadIdx.y * tile_size_x + threadIdx.x] = 0;
		}
		if (col < p && i * tile_size_x + threadIdx.y < n) {
			tile_B[threadIdx.y * tile_size_x + threadIdx.x] = B[(i * tile_size_x + threadIdx.y) * p + col];
		}
		else {
			tile_B[threadIdx.y * tile_size_x + threadIdx.x] = 0;
		}
		// synchronize threads to ensure tiles have been loaded
		__syncthreads();
		// perform the matrix multiplication for the current tile
		for (int j = 0; j < tile_size_x; ++j) {
			tile_C += tile_A[threadIdx.y * tile_size_x + j] * tile_B[j * tile_size_x + threadIdx.x];
		}
		// synchronize threads to ensure previous calculation has completed
		__syncthreads();
	}
	// write the result to global memory if within the bounds of C
	if (row < m && col < p) {
		C[row * p + col] = tile_C;
	}
}

__global__ void tiled_matrix_multiply_noBoundaries(float *A, float *B, float *C, int n)
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
	const int M_rows = 350;
	const int M_cols = 400;
	const int N_rows = 400;
	const int N_cols = 500;

	printf("Matrix M size is %d by %d\n\n", M_rows, M_cols);
	printf("Matrix N size is %d by %d\n\n", N_rows, N_cols);

	// Allocate memory for matrices on host
	float* M = (float*)malloc(M_rows * M_cols * sizeof(float));
	float* N = (float*)malloc(N_rows * N_cols * sizeof(float));
	float* P = (float*)malloc(M_rows * N_cols * sizeof(float));
	float* P_CPU = (float*)malloc(M_rows * N_cols * sizeof(float));

	// Initialize matrices with random values
	srand(time(NULL));
	for (int i = 0; i < M_rows; i++) {
		for (int j = 0; j < M_cols; j++) {
			M[i * M_cols + j] = (float)rand() / RAND_MAX;
		}
	}
	for (int i = 0; i < N_rows; i++) {
		for (int j = 0; j < N_cols; j++) {
			N[i * N_cols + j] = (float)rand() / RAND_MAX;
		}
	}

	// Matrix dimensions on device
	const int M_rows_d = M_rows;
	const int M_cols_d = M_cols;
	const int N_rows_d = N_rows;
	const int N_cols_d = N_cols;
	const int P_rows_d = M_rows_d;
	const int P_cols_d = N_cols_d;

	// Allocate memory for matrices on device
	float* M_d, *N_d, *P_d;
	cudaMalloc(&M_d, M_rows_d * M_cols_d * sizeof(float));
	cudaMalloc(&N_d, N_rows_d * N_cols_d * sizeof(float));
	cudaMalloc(&P_d, P_rows_d * P_cols_d * sizeof(float));

	// Copy matrices from host to device
	cudaMemcpy(M_d, M, M_rows_d * M_cols_d * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(N_d, N, N_rows_d * N_cols_d * sizeof(float), cudaMemcpyHostToDevice);

	// Kernel configuration
	const int tile_width = 8;
	const int tile_height = 15;
	dim3 gridDim((N_cols + tile_width - 1) / tile_width, (M_rows + tile_height - 1) / tile_height, 1);
	dim3 blockDim(tile_width, tile_height, 1);
	int sharedMemSize = 2 * tile_width * tile_height * sizeof(float);

	float gpu_time1 = 0.0f;
	float gpu_time2 = 0.0f;

	// Create events to measure time
	cudaEvent_t start1, stop1, start2, stop2;
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	cudaEventCreate(&start2);
	cudaEventCreate(&stop2);

	//HOST TEST
	cudaEventRecord(start1, 0);
	matrix_multiply(M, N, P_CPU, M_rows, M_cols, N_cols);
	cudaEventRecord(stop1, 0);

	cudaEventElapsedTime(&gpu_time1, start1, stop1);
	printf("Host Multiplication time: %0.2f\n", gpu_time1);

	//---------------------------------------------------------------
	cudaEventRecord(start2, 0);
	tiled_matrix_multiply << <gridDim, blockDim, sharedMemSize >> > (M_d, N_d, P_d, M_rows_d, M_cols_d, N_cols_d, tile_width, tile_height);
	cudaDeviceSynchronize();
	cudaEventRecord(stop2, 0);
	cudaEventSynchronize(stop2);
	cudaEventElapsedTime(&gpu_time2, start2, stop2);
	cudaMemcpy(P, P_d, P_rows_d * P_cols_d * sizeof(float), cudaMemcpyDeviceToHost);
	printf("Tiled Boundary Multiplication time: %0.2f\n", gpu_time2);

	int passedFlag = 1;
	for (int i = 0; i < M_rows * N_cols; i++) {
		if (abs(P[i] - P_CPU[i]) > 0.001) {
			passedFlag = 0;
			break;
		}
	}
	if (passedFlag) {
		printf("Test PASSED\n\n");
	}
	else {
		printf("Test FAILED\n");
	}

	// Free memory
	free(M);
	free(N);
	free(P);
	cudaFree(M_d);
	cudaFree(N_d);
	cudaFree(P_d);

	return 0;
}