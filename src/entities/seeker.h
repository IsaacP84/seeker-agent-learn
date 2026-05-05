#pragma once
#include <Magic!/core/entity.hpp>
#include <Magic!/core/entity_manager.hpp>
#include <Jolt/Jolt.h>
#include <Jolt/Physics/Body/BodyID.h>

// #include <torch/extension.h>

// ---------------------------------------------------------------------------
// Seeker
// ---------------------------------------------------------------------------

class Seeker
{
public:
    enum Action
    {
        NONE,
        MOVE_FORWARD,
        MOVE_BACKWARD,
        STRAFE_LEFT,
        STRAFE_RIGHT,
        TURN_LEFT,
        TURN_RIGHT,
        JUMP,
        LAND,
        SHOOT
    };
    struct Data
    {
        bool is_jumping = false;
        bool in_air = false;
        bool jumped = false;
        bool shoot = false;
        float jump_force;
        float move_speed;
        float turn_speed; // degrees per second
        float looking_angle = 0;  // in degrees

        // Optional override: if set, replaces the SeekerUpdateHandler on_move callback.
        std::function<Action(Magic::EntityManager &, Magic::Entity, double)> on_move_override;
    };
    Seeker() {}
    Seeker(Magic::EntityManager &em);
    Magic::Entity operator()() const
    {
        return m_e;
    }

    void shoot(Magic::EntityManager &em, Magic::Camera &cam);
    static void LoadModel();
    static void ClearBodyIds();


    // get state
    // new_state, reward, terminated, _, info = env.step(action.item())
    // get random action
private:
    Magic::Entity m_e = entt::null;
};

