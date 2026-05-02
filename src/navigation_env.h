#pragma once
#include <vector>
#include <array>
#include <random>
#include <string>
#include <unordered_map>
#include <variant>
#include <glm/glm.hpp>
#include <Magic!/core/entity.hpp>
#include <Magic!/core/entity_manager.hpp>
#include "components/components.h"
#include "entities/seeker.h"


struct Goal
{
    glm::vec3 pos;
    Magic::Entity entity = entt::null;
};

class NavigationEnv
{
public:
    struct Sizes
    {
        static constexpr int num_states = 199;
        static constexpr int num_actions = 6;
        static constexpr int num_goals   = 3;
    };

    struct Raycast
    {
        static constexpr int   num_rays        = 13;
        static constexpr int   num_ground_rays = 5; // downward edge-detection rays
        static constexpr float ray_len         = 8.0f;
        static constexpr float ground_pitch    = 45.f; // degrees below horizontal
        static constexpr float fov_deg         = 120.f;
    };

    struct SightingConstants
    {
        static constexpr int   history   = 3;
        static constexpr float max_age   = 5.0f;
        static constexpr float max_stale = 5.0f;
    };

    struct World
    {
        static constexpr float min             = -20.0f;
        static constexpr float max             = 20.0f;
        static constexpr float death_height    = -10.f;
    };

    struct Episode
    {
        static constexpr int   max_steps               = 2000;
        static constexpr float goal_radius             = 0.5f;
        static constexpr float max_seconds             = 180.f;
        static constexpr float max_goal_search_seconds = 60.f;
        static constexpr float min_goal_search_seconds = 10.f;
        static constexpr float search_time_fall_rate   = 0.003f;
    };

    struct Reward
    {
        static constexpr float fall_penalty   = -400.f;
        static constexpr float look_weight    = 0.01f;
        static constexpr float action_penalty  = 0.01f;
        static constexpr float strafe_penalty = 0.005f;
    };

    struct Angles
    {
        static constexpr float full_circle_deg = 360.f;
        static constexpr float half_circle_deg = 180.f;
        static constexpr float deg_to_rad      = 3.14159265358979323846f / 180.f;
    };

    using SightingBodyID = std::variant<JPH::BodyID, int>;

    void bind(Magic::EntityManager &em, Magic::Entity seeker, std::vector<Magic::Entity> goals);

    std::vector<float>                         reset();
    std::vector<float>                         get_observation(float dt);
    float                                      compute_reward();
    bool                                       is_done() const;
    void                                       apply_action(int action);
    Seeker::Action                             pending_action() const;
    std::unordered_map<std::string, float>     get_env_data() const;
    void                                       set_env_data(const std::unordered_map<std::string, float> &data);

private:
    Magic::EntityManager *m_em     = nullptr;
    Magic::Entity         m_seeker = entt::null;
    std::vector<Goal>     m_goals;

    glm::vec3 m_last_known_goal_pos{0.f}; // last position where goal had line-of-sight
    float     m_time_since_goal_visible = 0.f; // seconds since goal was last in LOS
    float     m_prev_distance           = 0.f;
    float     m_elapsed_seconds         = 0.f;
    float     m_time_since_goal         = 0.f;
    float     m_current_goal_time_limit = Episode::max_goal_search_seconds;
    bool      m_done             = false;
    int       m_step_count      = 0;
    int       m_pending_action  = 0;

    struct Sighting
    {
        glm::vec3      offset{0.f}; // XYZ offset from seeker at time of hit
        float          age      = SightingConstants::max_age;
        SightingBodyID body_id{};
        bool           valid    = false;
        float          interest = 0.f; // -0.1 = wall obstacle, 0.9 = goal
    };

    std::array<std::array<Sighting, SightingConstants::history>, Raycast::num_rays> m_sightings{};
    std::array<int, Raycast::num_rays>                                               m_sight_head{};

    std::mt19937 m_rng{std::random_device{}()};
};

