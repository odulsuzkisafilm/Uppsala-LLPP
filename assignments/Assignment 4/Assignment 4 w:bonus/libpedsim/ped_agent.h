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

using namespace std;

namespace Ped {
	class Twaypoint;

	class Tagent {
	public:
		Tagent(int posX, int posY);
		Tagent(double posX, double posY);

		// Returns the coordinates of the desired position
		int getDesiredX() const { return desiredPositionX; }
		int getDesiredY() const { return desiredPositionY; }

		// Sets the agent's position
		void setX(int newX) { x = newX; }
		void setY(int newY) { y = newY; }

		int getDesiredXs() const { return static_cast<int>(desiredPositionXs[0]); }
		int getDesiredYs() const { return static_cast<int>(desiredPositionYs[0]); }
		int getDesiredXs(size_t i) const { return desiredPositionXs[i]; }
		int getDesiredYs(size_t i) const { return desiredPositionYs[i]; }

		// Sets the agent's position
		void setXs(int newX) { xs[0] = static_cast<float>(newX); }
		void setYs(int newY) { ys[0] = static_cast<float>(newY); }
		void setXs(size_t i, int newX) { xs[i] = static_cast<float>(newX); }
		void setYs(size_t i, int newY) { ys[i] = static_cast<float>(newY); }


		// Update the position according to get closer
		// to the current destination
		void computeNextDesiredPosition();

		// Position of agent defined by x and y
		int getX() const { return x; };
		int getY() const { return y; };
		int getXs() const { return xs[0]; };
		int getYs() const { return ys[0]; };
		int getXs(size_t i) const { return xs[i]; };
		int getYs(size_t i) const { return ys[i]; };

		// Adds a new waypoint to reach for this agent
		void addWaypoint(Twaypoint* wp);

		~Tagent();
		Tagent(std::vector<Tagent*>& agents);

		float *xs;
		float *ys;
		float *desiredPositionXs;
		float *desiredPositionYs;

		float *dstXs;
		float *dstYs;
		float *dstRs;
		Twaypoint** destinations;
		size_t num_agents;
		deque<Twaypoint*> * waypointss;

	private:
		Tagent() {};

		// The agent's current position
		int x;
		int y;

		// The agent's desired next position
		int desiredPositionX;
		int desiredPositionY;

		// The current destination (may require several steps to reach)
		Twaypoint* destination;
		

		// The last destination
		Twaypoint* lastDestination;		

		// The queue of all destinations that this agent still has to visit
		deque<Twaypoint*> waypoints;
		
		// Internal init function 
		void init(int posX, int posY);

		// Returns the next destination to visit
		Twaypoint* getNextDestination();
	};
}

#endif