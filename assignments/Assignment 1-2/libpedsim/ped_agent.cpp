//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_agent.h"
#include "ped_waypoint.h"
#include <math.h>

#include <stdlib.h>

#include <iostream>
#include <immintrin.h>

Ped::Tagent::Tagent(int posX, int posY) : num_agents(0) {
	Ped::Tagent::init(posX, posY);
}

Ped::Tagent::Tagent(double posX, double posY) : num_agents(0) {
	Ped::Tagent::init((int)round(posX), (int)round(posY));
}

Ped::Tagent::~Tagent()
{
	if (num_agents > 0) {
		_mm_free(x);
		_mm_free(y);
		_mm_free(dstX);
		_mm_free(dstY);
		_mm_free(dstR);
	} 
	else {
		delete x;
		delete y;
		delete desiredPositionX;
		delete desiredPositionY;
	}

	delete[] destination;
	delete[] waypoints;
}

// Constructor that transforms from AoS to SoA
Ped::Tagent::Tagent(std::vector<Tagent*>& agents) : num_agents(agents.size())
{
	size_t mem_size = (num_agents + 3) & ~3;
	x = (float*) _mm_malloc(mem_size * sizeof(float), 16);
	y = (float*) _mm_malloc(mem_size * sizeof(float), 16);
	dstX = (float*) _mm_malloc(mem_size * sizeof(float), 16);
	dstY = (float*) _mm_malloc(mem_size * sizeof(float), 16);
	dstR = (float*) _mm_malloc(mem_size * sizeof(float), 16);

	destination = new Twaypoint*[num_agents];
	waypoints = new std::deque<Twaypoint*>[num_agents];
	for (size_t i = 0; i < num_agents; i++) {
		x[i] = agents[i]->getX();
		y[i] = agents[i]->getY();
		dstX[i] = dstY[i] = dstR[i] = 0;
		destination[i] = nullptr;
		waypoints[i] = agents[i]->waypoints[0];
		// TODO: delete agents after?
	}
}

void Ped::Tagent::init(int posX, int posY) {
	x = new float(posX);
	y = new float(posY);
	desiredPositionX = new float();
	desiredPositionY = new float();

	destination = new Twaypoint*[1];
	destination[0] = nullptr;

	waypoints = new std::deque<Twaypoint*>[1];
	waypoints[0] = std::deque<Twaypoint*>();

	lastDestination = nullptr;
}

void Ped::Tagent::computeNextDesiredPosition() {
	destination[0] = getNextDestination();
	if (destination[0] == NULL) {
		// no destination, no need to
		// compute where to move to
		return;
	}

	double diffX = destination[0]->getx() - getX();
	double diffY = destination[0]->gety() - getY();
	double len = sqrt(diffX * diffX + diffY * diffY);
	desiredPositionX[0] = (int)round(getX() + diffX / len);
	desiredPositionY[0] = (int)round(getY() + diffY / len);
}

// SIMD version
void Ped::Tagent::computeNextDesiredPositionSIMD() 
{
	#pragma omp parallel for
	for (int i = 0; i < num_agents; i += 4) {
		// agent->getNextDestination()
		__m128 _x, _y, _r, _dstX, _dstY, _dstR;
		_x = _mm_load_ps(&x[i]);
		_y = _mm_load_ps(&y[i]);
		_dstX = _mm_load_ps(&dstX[i]);
		_dstY = _mm_load_ps(&dstY[i]);
		_dstR = _mm_load_ps(&dstR[i]);

		__m128 _diffX, _diffY, _len;
		_diffX = _mm_sub_ps(_dstX, _x);
		_diffY = _mm_sub_ps(_dstY, _y);
		_len = _mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(_diffX, _diffX), _mm_mul_ps(_diffY, _diffY)));

		__m128 _cmpRes;
		_cmpRes = _mm_cmplt_ps(_len, _dstR);
		int mask = _mm_movemask_ps(_cmpRes);

		int remaining = num_agents - i;
		int count = (remaining < 4) ? remaining : 4;
		for (int j = 0; j < count; j++) {
			int k = i + j;
			bool agentReachedDestination = (mask & (1 << j)) != 0;
			if ((agentReachedDestination || !destination[k]) && waypoints[k].size() > 0) 
			{
				waypoints[k].push_back(destination[k]);
				destination[k] = waypoints[k].front();
				if (destination[k]) {
					dstX[k] = destination[k]->getx();
					dstY[k] = destination[k]->gety();
					dstR[k] = destination[k]->getr();
				} else {
					dstX[k] = dstY[k] = dstR[k] = 0;
				}
				waypoints[k].pop_front();
			}
		}

		// agent->computeNextDesiredPosition()
		_x = _mm_load_ps(&x[i]);
		_y = _mm_load_ps(&y[i]);
		_dstX = _mm_load_ps(&dstX[i]);
		_dstY = _mm_load_ps(&dstY[i]);
		_dstR = _mm_load_ps(&dstR[i]);

		_diffX = _mm_sub_ps(_dstX, _x);
		_diffY = _mm_sub_ps(_dstY, _y);
		_len = _mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(_diffX, _diffX), _mm_mul_ps(_diffY, _diffY)));

		__m128 _desiredX, _desiredY;
		_desiredX = _mm_add_ps(_x, _mm_div_ps(_diffX, _len));
		_desiredY = _mm_add_ps(_y, _mm_div_ps(_diffY, _len));

		_desiredX = _mm_round_ps(_desiredX, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
		_desiredY = _mm_round_ps(_desiredY, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	
		_mm_store_ps(&x[i], _desiredX);
		_mm_store_ps(&y[i], _desiredY);

		// _mm_store_ps(&desiredPositionX[i], _desiredX);
		// _mm_store_ps(&desiredPositionY[i], _desiredY);
	}
}

void Ped::Tagent::addWaypoint(Twaypoint* wp) {
	waypoints[0].push_back(wp);
}

Ped::Twaypoint* Ped::Tagent::getNextDestination() {
	Ped::Twaypoint* nextDestination = NULL;
	bool agentReachedDestination = false;

	if (destination[0] != NULL) {
		// compute if agent reached its current destination
		double diffX = destination[0]->getx() - getX();
		double diffY = destination[0]->gety() - getY();
		double length = sqrt(diffX * diffX + diffY * diffY);
		agentReachedDestination = length < destination[0]->getr();
	}

	if ((agentReachedDestination || destination[0] == NULL) && !waypoints[0].empty()) {
		// Case 1: agent has reached destination (or has no current destination);
		// get next destination if available
		waypoints[0].push_back(destination[0]);
		nextDestination = waypoints[0].front();
		waypoints[0].pop_front();
	}
	else {
		// Case 2: agent has not yet reached destination, continue to move towards
		// current destination
		nextDestination = destination[0];
	}

	return nextDestination;
}


