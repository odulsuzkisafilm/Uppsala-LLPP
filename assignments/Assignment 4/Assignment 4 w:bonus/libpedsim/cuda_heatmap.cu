#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ped_model.h"

#include <iostream>
#include <cmath>
#include <cstdlib>
#include "ped_agent.h"
#include "ped_waypoint.h"
#include <math.h>
#include <stdlib.h>

#define THREADSPERBLOCK 1024
#define MAX_WAYPOINTS_PER_AGENT 100

float *d_x, *d_y, *d_dstX, *d_dstY, *d_dstR;
Ped::Twaypoint** d_flattened_waypoints;

__global__ void floatToIntKernel(float *input, int *output, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        output[idx] = static_cast<int>(round(input[idx]));
    }
}

// Call this function from your host code where you need the conversion.
void convertFloatToIntAndCopy(float *d_input, int *h_output, int num_elements) {
    int *d_output;
    cudaMalloc(&d_output, num_elements * sizeof(int));

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    floatToIntKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, num_elements);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        // Handle the error (e.g., by returning or exiting)
    }

    // Synchronize to wait for the kernel to finish
    cudaDeviceSynchronize();

    // Step 2: Copy the converted array back to the host
    cudaMemcpy(h_output, d_output, num_elements * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_output);
}


void initt(int num_agents) {
    // Allocate device memory for agent positions and destinations
    cudaMalloc((void**)&d_x, num_agents * sizeof(float));
    cudaMalloc((void**)&d_y, num_agents * sizeof(float));
    cudaMalloc((void**)&d_dstX, num_agents * sizeof(float));
    cudaMalloc((void**)&d_dstY, num_agents * sizeof(float));
    cudaMalloc((void**)&d_dstR, num_agents * sizeof(float));
}

__global__ void computeNextDesiredPositionCU(float *x, float *y, float *dstX, float *dstY, float *dstR, float* waypointX, float* waypointY, float* waypointR, int* numWaypointsPerAgent, int maxWaypointsPerAgent, int num_agents) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_agents) {
        // Load data from global memory
        float _x = x[tid];
        float _y = y[tid];
        float _dstX = dstX[tid];
        float _dstY = dstY[tid];
        float _dstR = dstR[tid];

        // Access waypoints for the agent
        int numWaypoints = numWaypointsPerAgent[tid];
        float* agentWaypointX = &waypointX[tid * maxWaypointsPerAgent];
        float* agentWaypointY = &waypointY[tid * maxWaypointsPerAgent];
        float* agentWaypointR = &waypointR[tid * maxWaypointsPerAgent];

        // Compute agentReachedDestination and update waypoints if needed
        float _diffX = _dstX - _x;
        float _diffY = _dstY - _y;
        float _len = sqrt(_diffX * _diffX + _diffY * _diffY);

        bool agentReachedDestination = (_len < _dstR) || (numWaypoints > 0);
        if (agentReachedDestination) {
            // Update destination
            dstX[tid] = numWaypoints > 0 ? agentWaypointX[0] : 0;
            dstY[tid] = numWaypoints > 0 ? agentWaypointY[0] : 0;
            dstR[tid] = numWaypoints > 0 ? agentWaypointR[0] : 0;

            // Update waypoints
            if (numWaypoints > 0) {
                // Shift waypoints
                for (int i = 0; i < numWaypoints - 1; ++i) {
                    agentWaypointX[i] = agentWaypointX[i + 1];
                    agentWaypointY[i] = agentWaypointY[i + 1];
                    agentWaypointR[i] = agentWaypointR[i + 1];
                }
                // Set last waypoint to zero
                agentWaypointX[numWaypoints - 1] = 0;
                agentWaypointY[numWaypoints - 1] = 0;
                agentWaypointR[numWaypoints - 1] = 0;

                // Decrement number of waypoints
                numWaypointsPerAgent[tid]--;
            }
        }

        // Compute desired position
        float _desiredX = _x + _diffX / _len;
        float _desiredY = _y + _diffY / _len;
        // Round to nearest integer
        _desiredX = round(_desiredX);
        _desiredY = round(_desiredY);

        // Store the result back to global memory
        x[tid] = _desiredX;
        y[tid] = _desiredY;
    }
}

void flattenWaypoints(std::deque<Ped::Twaypoint*>* waypoints, float* flattened_waypoints_x, float* flattened_waypoints_y, float* flattened_waypoints_r) {
    int num_agents = waypoints->size();
    for (int i = 0; i < num_agents; ++i) {
        int j = 0;
        for (auto it = waypoints[i].begin(); it != waypoints[i].end(); ++it) {
            Ped::Twaypoint* waypoint = *it;
            flattened_waypoints_x[i * MAX_WAYPOINTS_PER_AGENT + j] = waypoint->getx();
            flattened_waypoints_y[i * MAX_WAYPOINTS_PER_AGENT + j] = waypoint->gety();
            flattened_waypoints_r[i * MAX_WAYPOINTS_PER_AGENT + j] = waypoint->getr();
            ++j;
        }
        // Set remaining coordinates and radii to zero
        for (; j < MAX_WAYPOINTS_PER_AGENT; ++j) {
            flattened_waypoints_x[i * MAX_WAYPOINTS_PER_AGENT + j] = 0;
            flattened_waypoints_y[i * MAX_WAYPOINTS_PER_AGENT + j] = 0;
            flattened_waypoints_r[i * MAX_WAYPOINTS_PER_AGENT + j] = 0;
        }
    }
}



// interactive -A uppmax2024-2-5 -M snowy -p core -n 1 -c 1 -t 1:00:01 --gres=gpu:1 --gres=mps:25

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

#define TILE_SIZE 16
#define RADIUS 2

__global__ void blurHeatmap(int *scaled_heatmap, int *blurred_heatmap)
{
	__shared__ int tile[TILE_SIZE + 2 * RADIUS][TILE_SIZE + 2 * RADIUS];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x * (TILE_SIZE - 2 * RADIUS) + tx - RADIUS;
	int by = blockIdx.y * (TILE_SIZE - 2 * RADIUS) + ty - RADIUS;

	if (bx >= 0 && bx < SCALED_SIZE && by >= 0 && by < SCALED_SIZE) {
		tile[ty][tx] = scaled_heatmap[by * SCALED_SIZE + bx];
	} else {
		tile[ty][tx] = 0;
	}

	__syncthreads();

	int sum = 0;
	if (tx >= RADIUS && tx < TILE_SIZE + RADIUS && ty >= RADIUS && ty < TILE_SIZE + RADIUS) {
		for (int k = -RADIUS; k <= RADIUS; k++) {
			for (int l = -RADIUS; l <= RADIUS; l++) {
				sum += w[k + RADIUS][l + RADIUS] * tile[ty + k][tx + l];
			}
		}
		int value = sum / WEIGHTSUM;
		if (bx < SCALED_SIZE && by < SCALED_SIZE) {
			blurred_heatmap[by * SCALED_SIZE + bx] = 0x00FF0000 | value << 24;
		}
	}

	/*
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
	 */
}

void Ped::Model::updateHeatmapCUDA(std::vector<Tagent*> &agents, std::vector<Tagent*> &SOA_agents)
{
	size_t num_agents = SOA_agents[0]->num_agents;
	float *xs = SOA_agents[0]->xs;
	float *ys = SOA_agents[0]->ys;
	float *dstXs = SOA_agents[0]->dstXs;
	float *dstYs = SOA_agents[0]->dstYs;
	float *dstRs = SOA_agents[0]->dstRs;
	deque<Twaypoint*> *waypointss = SOA_agents[0]->waypointss;

	int* numWaypointsPerAgent = new int[num_agents];
    int* d_numWaypointsPerAgent;
    cudaMalloc((void**)&d_numWaypointsPerAgent, num_agents * sizeof(int));
    cudaMemcpy(d_numWaypointsPerAgent, numWaypointsPerAgent, num_agents * sizeof(int), cudaMemcpyHostToDevice);
    initt(num_agents);

    // Copy agent positions and destinations to device
    cudaMemcpy(d_x, xs, num_agents * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, ys, num_agents * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dstX, dstXs, num_agents * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dstY, dstYs, num_agents * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dstR, dstRs, num_agents * sizeof(float), cudaMemcpyHostToDevice);

    float flattened_waypoints_x[(num_agents+1)*MAX_WAYPOINTS_PER_AGENT], flattened_waypoints_y[(num_agents+1)*MAX_WAYPOINTS_PER_AGENT], flattened_waypoints_r[(num_agents+1)*MAX_WAYPOINTS_PER_AGENT];
    flattenWaypoints(waypointss, flattened_waypoints_x, flattened_waypoints_y, flattened_waypoints_r);
    // Launch kernel
    int numBlocks = (num_agents + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
    float* d_flattened_waypointX;
    float* d_flattened_waypointY;
    float* d_flattened_waypointR;

    cudaMalloc((void**)&d_flattened_waypointX, num_agents * MAX_WAYPOINTS_PER_AGENT * sizeof(float));
    cudaMalloc((void**)&d_flattened_waypointY, num_agents * MAX_WAYPOINTS_PER_AGENT * sizeof(float));
    cudaMalloc((void**)&d_flattened_waypointR, num_agents * MAX_WAYPOINTS_PER_AGENT * sizeof(float));

    // Copy flattened waypoints data to device
    cudaMemcpy(d_flattened_waypointX, flattened_waypoints_x, num_agents * MAX_WAYPOINTS_PER_AGENT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flattened_waypointY, flattened_waypoints_y, num_agents * MAX_WAYPOINTS_PER_AGENT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flattened_waypointR, flattened_waypoints_r, num_agents * MAX_WAYPOINTS_PER_AGENT * sizeof(float), cudaMemcpyHostToDevice);
    computeNextDesiredPositionCU<<<numBlocks, THREADSPERBLOCK>>>(d_x, d_y, d_dstX, d_dstY, d_dstR, d_flattened_waypointX, d_flattened_waypointY, d_flattened_waypointR, d_numWaypointsPerAgent, MAX_WAYPOINTS_PER_AGENT, num_agents);

    // Copy result back to host
    convertFloatToIntAndCopy(d_x, desired_xs, num_agents);
	convertFloatToIntAndCopy(d_y, desired_ys, num_agents);

	cudaDeviceSynchronize();
	for (auto & agent: agents){
		move(agent);
	}
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

	cudaMemcpy(d_desired_xs, desired_xs, NUM_AGENTS * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_desired_ys, desired_ys, NUM_AGENTS * sizeof(int), cudaMemcpyHostToDevice);

	dimGrid = dim3(NUM_AGENTS/dimBlock.x, NUM_AGENTS/dimBlock.y);
	intensifyHeat<<<dimGrid, dimBlock>>>(d_heatmap, d_desired_xs, d_desired_ys, NUM_AGENTS);
	CHKLAST();

	dimGrid = dim3(SIZE/dimBlock.x, SIZE/dimBlock.y);
	postIntensify<<<dimGrid, dimBlock>>>(d_heatmap);
	CHKLAST();

	dimGrid = dim3(SIZE/dimBlock.x, SIZE/dimBlock.y);
	scaleHeatmap<<<dimGrid, dimBlock>>>(d_heatmap, d_scaled_heatmap);
	CHKLAST();

	dimBlock = dim3(TILE_SIZE, TILE_SIZE);
	dimGrid = dim3(SCALED_SIZE/dimBlock.x, SCALED_SIZE/dimBlock.y);
	blurHeatmap<<<dimGrid, dimBlock>>>(d_scaled_heatmap, d_blurred_heatmap);
	CHKLAST();

	cudaMemcpy(bhm, d_blurred_heatmap, SCALED_SIZE * SCALED_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

	// Free memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_dstX);
    cudaFree(d_dstY);
    cudaFree(d_dstR);
    cudaFree(d_numWaypointsPerAgent);
    cudaFree(d_flattened_waypointX);
    cudaFree(d_flattened_waypointY);
    cudaFree(d_flattened_waypointR);
    delete[] numWaypointsPerAgent;
}

void Ped::Model::cleanupCUDA()
{
	cudaFree(d_heatmap);
	cudaFree(d_scaled_heatmap);
	cudaFree(d_blurred_heatmap);
	cudaFree(d_desired_xs);
	cudaFree(d_desired_ys);
}
