//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
// Adapted for Low Level Parallel Programming 2017
//
// TAgent represents an agent in the scenario. Each
// agent has a position (x,y) and a number of destinations
// it wants to visit (waypoints). The desired next position
// represents the position it would like to visit next as it
// will bring it closer to its destination.
// Note: the agent will not move by itself, but the movement
// is handled in ped_model.cpp. 
//

#ifndef _ped_agent_h_
#define _ped_agent_h_ 1

#include <vector>
#include <deque>

namespace Ped {
	class Twaypoint;

	struct Tagent {
		Tagent(int posX, int posY);
		Tagent(double posX, double posY);

		// Returns the coordinates of the desired position
		int getDesiredX() const { return static_cast<int>(desiredPositionX[0]); }
		int getDesiredY() const { return static_cast<int>(desiredPositionY[0]); }
		// int getDesiredX(size_t i) const { return desiredPositionX[i]; }
		// int getDesiredY(size_t i) const { return desiredPositionY[i]; }

		// Sets the agent's position
		void setX(int newX) { x[0] = static_cast<float>(newX); }
		void setY(int newY) { y[0] = static_cast<float>(newY); }
		// void setX(size_t i, int newX) { x[i] = static_cast<float>(newX); }
		// void setY(size_t i, int newY) { y[i] = static_cast<float>(newY); }

		// Update the position according to get closer
		// to the current destination
		void computeNextDesiredPosition();
		void computeNextDesiredPositionSIMD();
		void computeNextDesiredPositionCUDA();

		// Position of agent defined by x and y
		int getX() const { return x[0]; };
		int getY() const { return y[0]; };
		// int getX(size_t i) const { return x[i]; };
		// int getY(size_t i) const { return y[i]; };

		// Adds a new waypoint to reach for this agent
		void addWaypoint(Twaypoint* wp);

		size_t num_agents;

		Tagent() {};
		Tagent(std::vector<Tagent*>& agents);
		~Tagent();

		// The agent's current position
		float *x;
		float *y;

		// The agent's desired next position
		float *desiredPositionX;
		float *desiredPositionY;

		float *dstX;
		float *dstY;
		float *dstR;

		// The current destination (may require several steps to reach)
		// Twaypoint* destination;
		Twaypoint** destination;

		// The last destination
		Twaypoint* lastDestination;

		// The queue of all destinations that this agent still has to visit
		// deque<Twaypoint*> waypoints;
		std::deque<Twaypoint*> *waypoints;

		// Internal init function 
		void init(int posX, int posY);

		// Returns the next destination to visit
		Twaypoint* getNextDestination();
	};
}

#endif
