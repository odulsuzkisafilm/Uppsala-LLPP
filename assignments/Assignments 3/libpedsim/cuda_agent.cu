#include "ped_agent.h"
#include "ped_waypoint.h"
#include <math.h>
#include <stdlib.h>
#include <iostream>

#define THREADSPERBLOCK 1024
#define MAX_WAYPOINTS_PER_AGENT 100

float *d_x, *d_y, *d_dstX, *d_dstY, *d_dstR;
Ped::Twaypoint** d_flattened_waypoints;

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


void Ped::Tagent::computeNextDesiredPositionCUDA(){
    int* numWaypointsPerAgent = new int[num_agents];
    int* d_numWaypointsPerAgent;
    cudaMalloc((void**)&d_numWaypointsPerAgent, num_agents * sizeof(int));
    cudaMemcpy(d_numWaypointsPerAgent, numWaypointsPerAgent, num_agents * sizeof(int), cudaMemcpyHostToDevice);
    initt(num_agents);

    // Copy agent positions and destinations to device
    cudaMemcpy(d_x, x, num_agents * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, num_agents * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dstX, dstX, num_agents * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dstY, dstY, num_agents * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dstR, dstR, num_agents * sizeof(float), cudaMemcpyHostToDevice);

    float flattened_waypoints_x[(num_agents+1)*MAX_WAYPOINTS_PER_AGENT], flattened_waypoints_y[(num_agents+1)*MAX_WAYPOINTS_PER_AGENT], flattened_waypoints_r[(num_agents+1)*MAX_WAYPOINTS_PER_AGENT];
    flattenWaypoints(waypoints, flattened_waypoints_x, flattened_waypoints_y, flattened_waypoints_r);
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
    cudaMemcpy(x, d_x, num_agents * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(y, d_y, num_agents * sizeof(float), cudaMemcpyDeviceToHost);

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
// interactive -A uppmax2024-2-5 -M snowy -p core -n 1 -c 1 -t 1:00:01 --gres=gpu:1 --gres=mps:25




