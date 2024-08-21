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

Ped::Tagent::Tagent(int posX, int posY) {
	Ped::Tagent::init(posX, posY);
}

Ped::Tagent::Tagent(double posX, double posY) {
	Ped::Tagent::init((int)round(posX), (int)round(posY));
}

Ped::Tagent::~Tagent()
{
	if (num_agents > 0) {
		free(xs);
		free(ys);
		free(dstXs);
		free(dstYs);
		free(dstRs);
		free(desiredPositionXs);
		free(desiredPositionYs);
	} 
	else {
		delete xs;
		delete ys;
		delete desiredPositionXs;
		delete desiredPositionYs;
	}

	delete[] destinations;
	delete[] waypointss;
}

Ped::Tagent::Tagent(std::vector<Tagent*>& agents) : num_agents(agents.size())
{
	size_t mem_size = (num_agents + 3) & ~3;
	xs = (float*) malloc(mem_size * sizeof(float));
	ys = (float*) malloc(mem_size * sizeof(float));
	dstXs = (float*) malloc(mem_size * sizeof(float));
	dstYs = (float*) malloc(mem_size * sizeof(float));
	dstRs = (float*) malloc(mem_size * sizeof(float));
	desiredPositionXs = (float*) malloc(mem_size * sizeof(float));
	desiredPositionYs = (float*) malloc(mem_size * sizeof(float));

	destinations = new Twaypoint*[num_agents];
	waypointss = new std::deque<Twaypoint*>[num_agents];

	for (size_t i = 0; i < num_agents; i++) {
		xs[i] = agents[i]->getX();
		ys[i] = agents[i]->getY();
		dstXs[i] = dstYs[i] = dstRs[i] = 0;
		destinations[i] = nullptr;
		waypointss[i] = agents[i]->waypointss[0];
		// TODO: delete agents after?
	}


	lastDestination = nullptr;
}

void Ped::Tagent::init(int posX, int posY) {
	x = posX;
	y = posY;
	destination = NULL;
	lastDestination = NULL;

	xs = new float(posX);
	ys = new float(posY);
	desiredPositionXs = new float();
	desiredPositionYs = new float();

	destinations = new Twaypoint*[1];
	destinations[0] = nullptr;

	waypointss = new std::deque<Twaypoint*>[1];
	waypointss[0] = std::deque<Twaypoint*>();
}


void Ped::Tagent::computeNextDesiredPosition() {
	destination = getNextDestination();
	if (destination == NULL) {
		// no destination, no need to
		// compute where to move to
		return;
	}

	double diffX = destination->getx() - x;
	double diffY = destination->gety() - y;
	double len = sqrt(diffX * diffX + diffY * diffY);
	desiredPositionX = (int)round(x + diffX / len);
	desiredPositionY = (int)round(y + diffY / len);
}

void Ped::Tagent::addWaypoint(Twaypoint* wp) {
	waypoints.push_back(wp);
}


Ped::Twaypoint* Ped::Tagent::getNextDestination() {
	Ped::Twaypoint* nextDestination = NULL;
	bool agentReachedDestination = false;

	if (destination != NULL) {
		// compute if agent reached its current destination
		double diffX = destination->getx() - x;
		double diffY = destination->gety() - y;
		double length = sqrt(diffX * diffX + diffY * diffY);
		agentReachedDestination = length < destination->getr();
	}

	if ((agentReachedDestination || destination == NULL) && !waypoints.empty()) {
		// Case 1: agent has reached destination (or has no current destination);
		// get next destination if available
		waypoints.push_back(destination);
		nextDestination = waypoints.front();
		waypoints.pop_front();
	}
	else {
		// Case 2: agent has not yet reached destination, continue to move towards
		// current destination
		nextDestination = destination;
	}

	return nextDestination;
}

