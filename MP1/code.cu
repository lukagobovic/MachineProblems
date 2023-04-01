#include <stdio.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

int _ConvertSMVer2Cores(int major, int minor);

int main() {
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	printf("Number of CUDA devices: %d\n", deviceCount);

	for (int i = 0; i < deviceCount; ++i) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, i);

		printf("Device %d: %s\n", i, deviceProp.name);

		printf("\tClock rate: %d MHz\n", deviceProp.clockRate / 1000);
		printf("\tNumber of streaming multiprocessors: %d\n", deviceProp.multiProcessorCount);
		printf("\tNumber of cores: %d\n", deviceProp.multiProcessorCount * _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor));
		printf("\tWarp size: %d\n", deviceProp.warpSize);
		printf("\tAmount of global memory: %lu bytes\n", (unsigned long)deviceProp.totalGlobalMem);
		printf("\tAmount of constant memory: %lu bytes\n", (unsigned long)deviceProp.totalConstMem);
		printf("\tAmount of shared memory per block: %lu bytes\n", (unsigned long)deviceProp.sharedMemPerBlock);
		printf("\tNumber of registers available per block: %d\n", deviceProp.regsPerBlock);
		printf("\tMaximum number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);
		printf("\tMaximum size of each dimension of a block: (%d, %d, %d)\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
		printf("\tMaximum size of each dimension of a grid: (%d, %d, %d)\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
	}

	return 0;
}

int _ConvertSMVer2Cores(int major, int minor) {
	// Returns the number of CUDA cores per SM for a given architecture version.
	// Refer to the CUDA documentation for the compute capability of each GPU.
	switch (major) {
	case 2:
		return (minor == 0) ? 32 : 48;
	case 3:
		return 192;
	case 5:
		return 128;
	case 6:
		return 64;
	case 7:
		return (minor == 0) ? 64 : 128;
	default:
		return 0;
	}
}
