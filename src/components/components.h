#pragma once
#include <Magic!/core/math.hpp>
#include <Magic!/components/physics.hpp>

#include <vector>
#include <functional>
using namespace Magic;


struct FreeCam
{

};

struct Pathing
{
    std::vector<Vec3> waypoints;
    size_t current_waypoint = 0;
    float min_distance = 1.0f;
    bool loop = false;
    bool active = true;

    std::function<void(Pathing&)> on_arrive = nullptr;

    void add_waypoint(const Vec3 &point)
    {
        waypoints.push_back(point);
    }
    void reset()
    {
        active = true;
        current_waypoint = 0;
    }

    Vec3 &get_current_waypoint()
    {
        assert(current_waypoint < waypoints.size());
        return waypoints[current_waypoint];
    }

    Vec3 get_current_waypoint() const
    {
        assert(current_waypoint < waypoints.size());
        return waypoints[current_waypoint];
    }

    void update(double dt, Transform &t)
    {
        if (!active)
            return;

        const Vec3 &target = waypoints[current_waypoint];
        // switch target
        if (glm::distance(t.pos, target) <= min_distance * dt)
        {
            t.pos = target;
            if (on_arrive)
                on_arrive(*this);


            current_waypoint++;
            if (loop)
            {
                current_waypoint %= waypoints.size();
            }
            else if (current_waypoint >= waypoints.size())
            {
                active = false;
                current_waypoint = waypoints.size();
            }
        }
    }
};
