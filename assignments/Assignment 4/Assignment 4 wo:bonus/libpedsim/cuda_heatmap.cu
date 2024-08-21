#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ped_model.h"

#include <iostream>
#include <cmath>
#include <cstdlib>

#define CHECK(code) check_cuda_operation(code, __FILE__, __LINE__, false)
#define MUST(code) check_cuda_operation(code, __FILE__, __LINE__, true)

inline void check_cuda_operation(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) {
		fprintf(stderr, "CUDA Error: %s (%s:%d)\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

#define CHKLAST() check_cuda_last()
inline void check_cuda_last()
{
	cudaDeviceSynchronize();
	cudaError_t code = cudaGetLastError();
	if (code != cudaSuccess) {
		fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(code));
	}
}

void Ped::Model::setupHeatmapCUDA()
{
	std::cout << "setting up heatmap" << std::endl;

	MUST(cudaMalloc((void**)&d_heatmap, SIZE * SIZE * sizeof(int)));
	MUST(cudaMalloc((void**)&d_scaled_heatmap, SCALED_SIZE * SCALED_SIZE * sizeof(int)));
	MUST(cudaMalloc((void**)&d_blurred_heatmap, SCALED_SIZE * SCALED_SIZE * sizeof(int)));

	// Allocate arrays for desired positions
	const size_t NUM_AGENTS = agents.size();
	MUST(cudaMalloc((void**)&d_desired_xs, NUM_AGENTS * sizeof(int)));
	MUST(cudaMalloc((void**)&d_desired_ys, NUM_AGENTS * sizeof(int)));
	desired_xs = (int*)malloc(NUM_AGENTS * sizeof(int));
	desired_ys = (int*)malloc(NUM_AGENTS * sizeof(int));

	// Allocate blurred_heatmap array for CPU
	bhm = (int*)malloc(SCALED_SIZE * SCALED_SIZE * sizeof(int));
	blurred_heatmap = (int**)malloc(SCALED_SIZE * sizeof(int*));
	for (int i = 0; i < SCALED_SIZE; i++) {
		blurred_heatmap[i] = bhm + SCALED_SIZE * i;
	}
}

__global__ void fadeHeatmap(int *heatmap)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < SIZE && y < SIZE) {
		heatmap[y * SIZE + x] = __float2int_rn(heatmap[y * SIZE + x] * 0.80f);
	}
}

__global__ void intensifyHeat(int *heatmap, int *desired_xs, int *desired_ys, const int NUM_AGENTS)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < NUM_AGENTS) {
		int x = desired_xs[idx];
		int y = desired_ys[idx];

		if (x >= 0 && x < SIZE && y >= 0 && y < SIZE) {
			atomicAdd(&heatmap[y * SIZE + x], 40);
		}
	}
}

__global__ void postIntensify(int *heatmap)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < SIZE && y < SIZE) {
		atomicMin(&heatmap[y * SIZE + x], 255);
	}
}

__global__ void scaleHeatmap(int *heatmap, int *scaled_heatmap)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < SIZE && y < SIZE) {
		int value = heatmap[y * SIZE + x];
		for (int cellY = 0; cellY < CELLSIZE; cellY++) {
			for (int cellX = 0; cellX < CELLSIZE; cellX++) {
				int scaled_index = ((y * CELLSIZE + cellY) * SIZE * CELLSIZE) + (x * CELLSIZE + cellX);
				scaled_heatmap[scaled_index] = value;
			}
		}
	}
}


#define WEIGHTSUM 273
__constant__ int w[5][5] = {
	{ 1, 4, 7, 4, 1 },
	{ 4, 16, 26, 16, 4 },
	{ 7, 26, 41, 26, 7 },
	{ 4, 16, 26, 16, 4 },
	{ 1, 4, 7, 4, 1 }
};

__global__ void blurHeatmap(int *scaled_heatmap, int *blurred_heatmap)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x + 2;
	int j = blockIdx.y * blockDim.y + threadIdx.y + 2;

	if (i < SCALED_SIZE - 2 && j < SCALED_SIZE - 2) {
		int sum = 0;
		for (int k = -2; k <= 2; k++) {
			for (int l = -2; l <= 2; l++) {
				sum += w[k + 2][l + 2] * scaled_heatmap[(j + k) * SCALED_SIZE + (i + l)];
			}
		}
		int value = sum / WEIGHTSUM;
		blurred_heatmap[j * SCALED_SIZE + i] = 0x00FF0000 | value << 24;
	}
}

void Ped::Model::updateHeatmapCUDA()
{
	// std::cout << "updating heatmap" << std::endl;

	dim3 dimBlock(16, 16);
	dim3 dimGrid(SIZE/dimBlock.x, SIZE/dimBlock.y);
	fadeHeatmap<<<dimGrid, dimBlock>>>(d_heatmap);
	CHKLAST();

	const size_t NUM_AGENTS = agents.size();
	for (size_t i = 0; i < NUM_AGENTS; ++i) {
		auto agent = agents[i];
		desired_xs[i] = agent->getDesiredX();
		desired_ys[i] = agent->getDesiredY();
		// printf("Agent %d desires (%d, %d)\n", i, desired_xs[i], desired_ys[i]);
	}

	CHECK(cudaMemcpy(d_desired_xs, desired_xs, NUM_AGENTS * sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_desired_ys, desired_ys, NUM_AGENTS * sizeof(int), cudaMemcpyHostToDevice));

	dimGrid = dim3(NUM_AGENTS/dimBlock.x, NUM_AGENTS/dimBlock.y);
	intensifyHeat<<<dimGrid, dimBlock>>>(d_heatmap, d_desired_xs, d_desired_ys, NUM_AGENTS);
	CHKLAST();

	dimGrid = dim3(SIZE/dimBlock.x, SIZE/dimBlock.y);
	postIntensify<<<dimGrid, dimBlock>>>(d_heatmap);
	CHKLAST();

	dimGrid = dim3(SIZE/dimBlock.x, SIZE/dimBlock.y);
	scaleHeatmap<<<dimGrid, dimBlock>>>(d_heatmap, d_scaled_heatmap);
	CHKLAST();

	dimGrid = dim3(SCALED_SIZE/dimBlock.x, SCALED_SIZE/dimBlock.y);
	blurHeatmap<<<dimGrid, dimBlock>>>(d_scaled_heatmap, d_blurred_heatmap);
	CHKLAST();

	cudaMemcpy(bhm, d_blurred_heatmap, SCALED_SIZE * SCALED_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
}

void Ped::Model::cleanupCUDA()
{
	cudaFree(d_heatmap);
	cudaFree(d_scaled_heatmap);
	cudaFree(d_blurred_heatmap);
	cudaFree(d_desired_xs);
	cudaFree(d_desired_ys);
}
