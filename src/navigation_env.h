#pragma once
#include <vector>
#include <array>
#include <random>
#include <limits>
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
    bool collected = false; // for bookkeeping in case you want to give partial rewards for proximity to the goal or something like that
};

class NavigationEnv
{
public:
    struct Raycast
    {
        static constexpr int num_rays = 13;
        static constexpr int num_ground_rays = 5; // downward edge-detection rays
        static constexpr float ray_len = 8.0f;
        static constexpr float ground_pitch = 45.f; // degrees below horizontal
        static constexpr float fov_deg = 120.f;
    };

    struct SightingConstants
    {
        static constexpr int history = 3;
        static constexpr float max_age = 5.0f;
        static constexpr float max_stale = 5.0f;
        static constexpr float wall_interest = -0.1f; // sighting interest for a wall hit
        static constexpr float goal_interest = 0.9f;  // sighting interest for a visible goal
    };

    struct World
    {
        static constexpr float min = -20.0f;
        static constexpr float max = 20.0f;
        static constexpr float death_height = -10.f;
        static constexpr float spawn_margin = 0.8f; // fraction of world bounds used for goal spawn

        static constexpr int num_concurrent_goals = 1;
        static constexpr float goal_radius = 1.25f;
    };

    struct Episode
    {
        static constexpr int max_steps = 100000;
        static constexpr float max_seconds = 300.f;
    };

    struct Curriculum
    {
        static constexpr float max_goal_search_seconds = 60.f;
        static constexpr float min_goal_search_seconds = 20.f;
        static constexpr float search_time_fall_rate = 0.05f;
        // Episodes to keep boundary walls active (curriculum learning).
        // LearnScene creates the walls; after this many resets it removes them.
        // Set to 0 to disable the curriculum (no walls ever created).
        static constexpr int boundary_wall_episodes = 200;
    };

    struct Reward
    {
        static constexpr float fall_penalty = -50.f;
        static constexpr float goal_reward = 10.f;
        static constexpr float look_weight = 0.003f;
        static constexpr float action_penalty = 0.01f;
        static constexpr float strafe_penalty = 0.02f;
        static constexpr float forward_reward = 0.02f;       // bonus per step for MOVE_FORWARD on ground
        static constexpr float goal_speed_bonus = 2.f;       // extra added to goal_reward if reached quickly
        static constexpr float goal_quick_threshold = 15.f;  // seconds — full bonus if goal reached within this
        static constexpr float max_speed = 10.f;             // normalisation divisor for linear velocity inputs
        static constexpr float max_angular_speed = 10.f;     // normalisation divisor for angular velocity input (rad/s)
        static constexpr float edge_danger_dist = 4.f;       // world units from map boundary where penalty begins
        static constexpr float edge_danger_penalty = 1.0f;   // max penalty per step at the boundary itself
        static constexpr float stuck_speed_threshold = 0.9f; // m/s — below this while moving = stuck against wall
        static constexpr float stuck_penalty = 0.05f;        // per-step penalty when stuck
    };

    struct Angles
    {
        static constexpr float full_circle_deg = 360.f;
        static constexpr float half_circle_deg = 180.f;
        static constexpr float deg_to_rad = 3.14159265358979323846f / 180.f;
    };

    struct Sizes
    {
        static constexpr int num_actions = 8;
        static constexpr int action_history = 4;
        static constexpr int num_states =
            Raycast::num_rays * 2 +
            Raycast::num_rays * SightingConstants::history * 4 +
            Raycast::num_ground_rays +
            15 +
            num_actions * action_history;
    };

    using SightingBodyID = std::variant<JPH::BodyID, int>;

    void bind(Magic::EntityManager &em, Magic::Entity seeker, std::vector<Magic::Entity> goals);
    void set_agent_id(int id) { m_agent_id = id; }
    void set_ignored_bodies(std::vector<JPH::BodyID> ids) { m_ignored_bodies = std::move(ids); }
    void set_pending_none() { m_pending_action = -1; } // maps to Seeker::NONE via default case
    void seed_rng(uint32_t seed) { m_rng.seed(seed); }

    std::vector<float> reset();
    std::vector<float> get_observation(float dt);
    float compute_reward();
    bool is_done() const;     // true on any termination (terminal or truncated)
    bool is_terminal() const; // true only for true terminal (fell off map)
    void apply_action(int action);
    Seeker::Action pending_action() const;
    std::unordered_map<std::string, float> get_env_data() const;
    void set_env_data(const std::unordered_map<std::string, float> &data);
    std::unordered_map<std::string, std::unordered_map<std::string, float>> get_config_data() const;
    void set_config_data(const std::string &category, const std::string &name, float value);
    static void load_goals(const std::vector<Goal> &goals);
    static std::vector<Goal> get_goals()
    {
        return s_goals;
    }
    // you can use this function to set the position for the next goal before the agent reaches it, but be careful not to call it after the episode is done since that can cause a crash (since the goal entities get destroyed on reset).
    static std::function<void()> on_next_goal() { s_goals.clear(); }
    static Goal get_next_goal(int idx);
    int episode_count() const { return m_episode_count; }
    void disable_goal_timeout() { m_current_goal_time_limit = std::numeric_limits<float>::max(); }

private:
    struct RuntimeConfig
    {
        struct Raycast
        {
            float ray_len = NavigationEnv::Raycast::ray_len;
            float ground_pitch = NavigationEnv::Raycast::ground_pitch;
            float fov_deg = NavigationEnv::Raycast::fov_deg;
        } raycast;

        struct Sighting
        {
            float wall_interest = NavigationEnv::SightingConstants::wall_interest;
            float goal_interest = NavigationEnv::SightingConstants::goal_interest;
        } sighting;

        struct World
        {
            float min = NavigationEnv::World::min;
            float max = NavigationEnv::World::max;
            float death_height = NavigationEnv::World::death_height;
            int num_concurrent_goals = NavigationEnv::World::num_concurrent_goals;
            float goal_radius = NavigationEnv::World::goal_radius;
        } world;

        struct Episode
        {
            int max_steps = NavigationEnv::Episode::max_steps;
            float max_seconds = NavigationEnv::Episode::max_seconds;
        } episode;

        struct Curriculum
        {
            float max_goal_search_seconds = NavigationEnv::Curriculum::max_goal_search_seconds;
            float min_goal_search_seconds = NavigationEnv::Curriculum::min_goal_search_seconds;
            float search_time_fall_rate = NavigationEnv::Curriculum::search_time_fall_rate;
        } curriculum;

        struct Reward
        {
            float fall_penalty = NavigationEnv::Reward::fall_penalty;
            float goal_reward = NavigationEnv::Reward::goal_reward;
            float look_weight = NavigationEnv::Reward::look_weight;
            float action_penalty = NavigationEnv::Reward::action_penalty;
            float strafe_penalty = NavigationEnv::Reward::strafe_penalty;
            float forward_reward = NavigationEnv::Reward::forward_reward;
            float goal_speed_bonus = NavigationEnv::Reward::goal_speed_bonus;
            float goal_quick_threshold = NavigationEnv::Reward::goal_quick_threshold;
            float max_speed = NavigationEnv::Reward::max_speed;
            float max_angular_speed = NavigationEnv::Reward::max_angular_speed;
            float edge_danger_dist = NavigationEnv::Reward::edge_danger_dist;
            float edge_danger_penalty = NavigationEnv::Reward::edge_danger_penalty;
            float stuck_speed_threshold = NavigationEnv::Reward::stuck_speed_threshold;
            float stuck_penalty = NavigationEnv::Reward::stuck_penalty;
        } reward;
    };

    RuntimeConfig m_config;
    Magic::EntityManager *m_em = nullptr;
    int m_agent_id = -1;
    Magic::Entity m_seeker = entt::null;
    static std::vector<Goal> s_goals;
    std::vector<Goal> m_goals;

    glm::vec3 m_last_known_goal_pos{0.f};  // last position where goal had line-of-sight
    float m_time_since_goal_visible = 0.f; // seconds since goal was last in LOS
    float m_prev_distance = 0.f;
    int m_nearest_goal_idx = 0; // cached by get_observation, consumed by compute_reward
    float m_nearest_goal_dist = 0.f;
    bool m_in_air = false;           // cached by get_observation, consumed by compute_reward
    bool m_was_goal_visible = false; // edge-detection: log only on first visible frame
    float m_speed_xz = 0.f;          // cached by get_observation, consumed by compute_reward
    int m_episode_count = 0;         // total resets; used by LearnScene for curriculum
    std::array<int, Sizes::action_history> m_prev_actions{};
    float m_elapsed_seconds = 0.f;
    float m_time_since_goal = 0.f;
    float m_current_goal_time_limit = Curriculum::max_goal_search_seconds;
    bool m_done = false;
    int m_step_count = 0;
    int m_goals_this_episode = 0;
    int m_pending_action = 0;
    int m_prev_action = -1; // -1 = no previous action (start of episode)

    struct Sighting
    {
        glm::vec3 offset{0.f}; // XYZ offset from seeker at time of hit
        float age = SightingConstants::max_age;
        SightingBodyID body_id{};
        bool valid = false;
        float interest = 0.f; // -0.1 = wall obstacle, 0.9 = goal
    };

    std::array<std::array<Sighting, SightingConstants::history>, Raycast::num_rays> m_sightings{};
    std::array<int, Raycast::num_rays> m_sight_head{};

    std::vector<JPH::BodyID> m_ignored_bodies; // other agents' seeker bodies — ignored by raycasts

    std::mt19937 m_rng{std::random_device{}()};
};
