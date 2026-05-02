#include "navigation_env.h"

#include <Magic!/app/application.hpp>
#include <Magic!/components/physics.hpp>
#include <Magic!/components/renderable.hpp>
#include <Magic!/core/math.hpp>
#include <Magic!/debug/debug.hpp>

#include <Jolt/Physics/Collision/RayCast.h>
#include <Jolt/Physics/Collision/CastResult.h>

#include <cmath>
#include <algorithm>
#include <limits>

using namespace Magic;

// Ignore the seeker body when raycasting (same pattern as seeker.cpp).
class NavIgnoreSelfFilter : public JPH::BodyFilter
{
public:
    NavIgnoreSelfFilter(JPH::BodyID id) : m_id(id) {}
    bool ShouldCollide(const JPH::BodyID &id) const override { return id != m_id; }

private:
    JPH::BodyID m_id;
};

void NavigationEnv::bind(EntityManager &em, Entity seeker, std::vector<Entity> goals)
{
    m_em = &em;
    m_seeker = seeker;

    m_goals.clear();
    {
        auto [lock, reg] = ((const EntityManager&)em).get_registry();
        for (Entity e : goals)
        {
            Goal g;
            g.entity = e;
            if (e != entt::null && reg.all_of<Transform>(e))
                g.pos = reg.get<Transform>(e).pos;
            m_goals.push_back(g);
        }
    }
}

std::vector<float> NavigationEnv::reset()
{
    Debug::Log("NavigationEnv reset");
    m_step_count       = 0;
    m_elapsed_seconds  = 0.f;
    m_time_since_goal  = 0.f;
    m_done             = false;
    m_pending_action   = 0;

    m_current_goal_time_limit = std::max(Episode::min_goal_search_seconds,
                                         m_current_goal_time_limit - Episode::search_time_fall_rate);

    // Clear sighting buffers.
    for (auto &ray_buf : m_sightings)
        for (auto &sg : ray_buf)
            sg = Sighting{};

    m_sight_head.fill(0);

    // Randomize all goal positions.
    if (m_em && !m_goals.empty())
    {
        auto [lock, reg] = m_em->get_registry();
        std::uniform_real_distribution<float> rdist(World::min * 0.8f, World::max * 0.8f);
        for (auto &g : m_goals)
        {
            g.pos.x = rdist(m_rng);
            g.pos.z = rdist(m_rng);
            if (g.entity != entt::null && reg.all_of<Transform>(g.entity))
                reg.get<Transform>(g.entity).pos = g.pos;
        }
    }

    // Return a zero observation to keep Python happy.
    m_prev_distance           = 0.f;
    m_last_known_goal_pos     = m_goals.empty() ? glm::vec3{0.f} : m_goals[0].pos;
    m_time_since_goal_visible = 0.f;
    return std::vector<float>(NavigationEnv::Sizes::num_states, 0.f);
}

std::vector<float> NavigationEnv::get_observation(float dt)
{
    std::vector<float> obs(NavigationEnv::Sizes::num_states, 0.f);

    if (!m_em || m_seeker == entt::null)
        return obs;

    // Copy all needed component data under the shared lock, then release it
    // before raycasting so the registry mutex is not held across Jolt calls.
    glm::vec3 pos;
    float looking_angle;
    JPH::BodyID body_id;
    {
        auto [lock, reg] = ((const EntityManager&)*m_em).get_registry();
        if (!reg.all_of<Transform, Seeker::Data, RigidBody>(m_seeker))
            return obs;

        pos          = reg.get<Transform>(m_seeker).pos;
        looking_angle = reg.get<Seeker::Data>(m_seeker).looking_angle;
        body_id      = reg.get<RigidBody>(m_seeker).id;

        for (auto &g : m_goals)
            if (g.entity != entt::null && reg.all_of<Transform>(g.entity))
                g.pos = reg.get<Transform>(g.entity).pos;
    } // shared lock released here

    // Fell off the map — end the episode immediately.
    if (pos.y < World::death_height)
        m_done = true;

    // Accumulate elapsed time for the episode timeout check.
    m_elapsed_seconds  += dt;
    m_time_since_goal  += dt;

    Physics &physics = GetEngine().get_physics();
    JPH::BodyInterface &bi = physics.physics_system.GetBodyInterface();

    // Find nearest goal.
    glm::vec3 goal_pos{0.f};
    if (!m_goals.empty())
    {
        int nearest_idx = 0;
        float nearest_dist = std::numeric_limits<float>::max();
        for (int i = 0; i < (int)m_goals.size(); ++i)
        {
            glm::vec2 d = {m_goals[i].pos.x - pos.x, m_goals[i].pos.z - pos.z};
            float d_len = glm::length(d);
            if (d_len < nearest_dist) { nearest_dist = d_len; nearest_idx = i; }
        }
        goal_pos = m_goals[nearest_idx].pos;
    }

    glm::vec2 to_goal_world = {goal_pos.x - pos.x, goal_pos.z - pos.z};
    float distance = glm::length(to_goal_world);
    m_prev_distance = distance;

    // --- Pass 1: age-and-blank (no registry needed) ---
    for (int i = 0; i < Raycast::num_rays; ++i)
    {
        for (int s = 0; s < SightingConstants::history; ++s)
        {
            Sighting &sg = m_sightings[i][s];
            if (!sg.valid)
                continue;
            sg.age += dt;
            if (sg.age >= SightingConstants::max_age)
                sg = Sighting{};
        }
    }

    // --- LOS check: ray from seeker straight to goal position ---
    // If anything (wall) is hit before we reach the goal, the goal is occluded.
    NavIgnoreSelfFilter self_filter(body_id);
    bool goal_visible = false;
    {
        JPH::Vec3 los_dir(to_goal_world.x, 0.f, to_goal_world.y);
        JPH::RRayCast los_ray{JPH::Vec3(pos.x, pos.y, pos.z), los_dir};
        JPH::RayCastResult los_hit;
        goal_visible = !physics.physics_system.GetNarrowPhaseQuery()
                           .CastRay(los_ray, los_hit, {}, {}, self_filter);
    }
    if (goal_visible)
    {
        m_last_known_goal_pos     = goal_pos;
        m_time_since_goal_visible = 0.f;
    }
    else
    {
        m_time_since_goal_visible += dt;
    }

    // --- Precompute goal visibility + bearing (one LOS cast per goal) ---
    struct GoalInfo
    {
        bool              visible;
        float             bearing; // world-space degrees
        glm::vec3         offset;  // from seeker
        NavigationEnv::SightingBodyID body_id;
    };
    std::vector<GoalInfo> goal_infos;
    goal_infos.reserve(m_goals.size());
    for (int gi = 0; gi < (int)m_goals.size(); ++gi)
    {
        glm::vec2 to_g = {m_goals[gi].pos.x - pos.x, m_goals[gi].pos.z - pos.z};
        JPH::Vec3 g_los_dir(to_g.x, 0.f, to_g.y);
        JPH::RRayCast g_los_ray{JPH::Vec3(pos.x, pos.y, pos.z), g_los_dir};
        JPH::RayCastResult g_los_hit;
        bool g_vis = !physics.physics_system.GetNarrowPhaseQuery()
                          .CastRay(g_los_ray, g_los_hit, {}, {}, self_filter);
        goal_infos.push_back({
            g_vis,
            glm::degrees(std::atan2(to_g.x, to_g.y)),
            {to_g.x, m_goals[gi].pos.y - pos.y, to_g.y},
            gi
        });
    }

    // --- Pass 2: raycast + record + build obs (registry lock not held) ---
    // 120° FOV: 13 rays spanning -60° to +60° relative to looking direction.
    const float SPREAD = Raycast::fov_deg / (Raycast::num_rays - 1);

    for (int i = 0; i < Raycast::num_rays; ++i)
    {
        float angle_deg = looking_angle + (i - Raycast::num_rays / 2) * SPREAD;
        float rad = glm::radians(angle_deg);
        JPH::Vec3 dir(std::sin(rad) * Raycast::ray_len, 0.f, std::cos(rad) * Raycast::ray_len);

        JPH::RRayCast ray{JPH::Vec3(pos.x, pos.y, pos.z), dir};
        JPH::RayCastResult hit;
        bool had_hit = physics.physics_system.GetNarrowPhaseQuery()
                           .CastRay(ray, hit, {}, {}, self_filter);

        // Section A — current hit fraction.
        obs[i] = had_hit ? hit.mFraction : 1.0f;

        // Track the most significant interest written to this ray this frame.
        float ray_interest = 0.f;

        auto write_sighting = [&](glm::vec3 offset, NavigationEnv::SightingBodyID bid, float interest)
        {
            int write_slot = -1;
            for (int s = 0; s < SightingConstants::history; ++s)
            {
                if (m_sightings[i][s].valid && m_sightings[i][s].body_id == bid)
                {
                    write_slot = s;
                    break;
                }
            }
            if (write_slot < 0)
                write_slot = m_sight_head[i];

            m_sightings[i][write_slot] = {offset, 0.f, bid, true, interest};

            if (write_slot == m_sight_head[i])
                m_sight_head[i] = (m_sight_head[i] + 1) % SightingConstants::history;

            ray_interest = interest;
        };

        // Record wall sighting (interest = -0.1).
        if (had_hit)
        {
            glm::vec3 hit_offset = {
                dir.GetX() * hit.mFraction,
                dir.GetY() * hit.mFraction,
                dir.GetZ() * hit.mFraction};
            write_sighting(hit_offset, hit.mBodyID, -0.1f);
        }

        // Record goal sightings (interest = 0.9) for any visible goal whose
        // bearing falls within this ray bin. Goals overwrite ray_interest last,
        // so Section B reflects the goal over any wall on the same ray.
        for (const auto &gi : goal_infos)
        {
            if (!gi.visible)
                continue;
            float diff = std::abs(gi.bearing - angle_deg);
            if (diff > Angles::half_circle_deg) diff = Angles::full_circle_deg - diff;
            if (diff <= SPREAD * 0.5f)
                write_sighting(gi.offset, gi.body_id, 0.9f);
        }

        // Section B — interest of the most significant sighting this frame.
        obs[Raycast::num_rays + i] = ray_interest;

        // Section C — write ring buffer (read from oldest → newest).
        // 4 floats per sighting: offset.x, offset.y, offset.z, age.
        for (int s = 0; s < SightingConstants::history; ++s)
        {
            int idx = Raycast::num_rays * 2 + i * (SightingConstants::history * 4) + s * 4;
            int slot = (m_sight_head[i] + s) % SightingConstants::history;
            const Sighting &sg = m_sightings[i][slot];
            obs[idx + 0] = sg.valid ? sg.offset.x : 0.f;
            obs[idx + 1] = sg.valid ? sg.offset.y : 0.f;
            obs[idx + 2] = sg.valid ? sg.offset.z : 0.f;
            obs[idx + 3] = sg.valid ? sg.age / SightingConstants::max_age : 1.f;
        }
    }

    // --- Section E (indices 182–186): ground / edge-detection rays ---
    // NUM_GROUND_RAYS downward-pitched rays spanning the forward 120° FOV.
    // 1.0 = no ground hit within RAY_LEN = edge of map nearby.
    {
        const float gSpread    = Raycast::fov_deg / (Raycast::num_ground_rays - 1);
        const float pitchRad   = glm::radians(Raycast::ground_pitch);
        const float cosPitch   = std::cos(pitchRad);
        const float sinPitch   = std::sin(pitchRad); // positive, applied as -y
        for (int i = 0; i < Raycast::num_ground_rays; ++i)
        {
            float angle_deg = looking_angle + (i - Raycast::num_ground_rays / 2) * gSpread;
            float rad = glm::radians(angle_deg);
            JPH::Vec3 dir(
                std::sin(rad) * cosPitch * Raycast::ray_len,
                -sinPitch * Raycast::ray_len,
                std::cos(rad) * cosPitch * Raycast::ray_len);
            JPH::RRayCast ray{JPH::Vec3(pos.x, pos.y, pos.z), dir};
            JPH::RayCastResult hit;
            bool had_hit = physics.physics_system.GetNarrowPhaseQuery()
                               .CastRay(ray, hit, {}, {}, self_filter);
            obs[182 + i] = had_hit ? hit.mFraction : 1.0f;
        }
    }

    // --- Section D (indices 187–198) ---
    const float diag = std::sqrt(2.f) * (World::max - World::min);

    // 187–188: normalised direction to LAST KNOWN goal in seeker-local space.
    // Uses m_last_known_goal_pos so the agent retains a search target when
    // the goal is behind a wall. Updated to m_goal_pos whenever goal is visible.
    // Seeker forward = {sin(a), 0, cos(a)}, right = {cos(a), 0, -sin(a)}.
    // World-to-local: local_x = cos(a)*dx - sin(a)*dz,  local_z = sin(a)*dx + cos(a)*dz.
    float look_rad = glm::radians(looking_angle);
    float cos_l = glm::cos(look_rad);
    float sin_l = glm::sin(look_rad);
    glm::vec2 to_last_known = {m_last_known_goal_pos.x - pos.x, m_last_known_goal_pos.z - pos.z};
    float last_known_dist = std::max(glm::length(to_last_known), 1e-6f);
    glm::vec2 local_goal = {
        cos_l * to_last_known.x - sin_l * to_last_known.y,
        sin_l * to_last_known.x + cos_l * to_last_known.y};
    float local_len = std::max(glm::length(local_goal), 1e-6f);
    obs[187] = local_goal.x / local_len;
    obs[188] = local_goal.y / local_len;

    // 189–190: seeker linear velocity (xz).
    JPH::Vec3 vel = bi.GetLinearVelocity(body_id);
    obs[189] = vel.GetX();
    obs[190] = vel.GetZ();

    // 191: looking angle normalised to [-1, 1].
    {
        float norm_angle = std::fmod(looking_angle, Angles::full_circle_deg);
        if (norm_angle < 0.f) norm_angle += Angles::full_circle_deg;
        obs[191] = norm_angle / Angles::half_circle_deg - 1.f;
    }

    // 192: distance to last known goal, normalised.
    obs[192] = std::min(last_known_dist / diag, 1.f);

    // 193–194: seeker world position normalised.
    obs[193] = (pos.x - World::min) / (World::max - World::min);
    obs[194] = (pos.z - World::min) / (World::max - World::min);

    // 195–196: last known goal world position normalised.
    obs[195] = (m_last_known_goal_pos.x - World::min) / (World::max - World::min);
    obs[196] = (m_last_known_goal_pos.z - World::min) / (World::max - World::min);

    // 197: goal currently visible (1.0 = yes, 0.0 = occluded by wall).
    obs[197] = goal_visible ? 1.f : 0.f;

    // 198: staleness of last known position, normalised [0, 1] over 5 s.
    //      0 = just saw it, 1 = not seen for 5+ seconds.
    obs[198] = std::min(m_time_since_goal_visible / SightingConstants::max_stale, 1.f);

    return obs;
}

float NavigationEnv::compute_reward()
{
    if (!m_em || m_seeker == entt::null || m_goals.empty())
        return 0.f;

    auto [lock, reg] = m_em->get_registry();
    if (!reg.all_of<Transform>(m_seeker))
        return 0.f;

    glm::vec3 pos = reg.get<Transform>(m_seeker).pos;
    float looking_angle = 0.f;
    if (reg.all_of<Seeker::Data>(m_seeker))
        looking_angle = reg.get<Seeker::Data>(m_seeker).looking_angle;

    // Fell off the map — heavy penalty so the agent strongly avoids the edges.
    if (pos.y < World::death_height)
        return Reward::fall_penalty;

    // TODO: edge danger penalty

    // Find nearest goal.
    int nearest_idx = 0;
    float nearest_dist = std::numeric_limits<float>::max();
    for (int i = 0; i < (int)m_goals.size(); ++i)
    {
        glm::vec2 d = {m_goals[i].pos.x - pos.x, m_goals[i].pos.z - pos.z};
        float d_len = glm::length(d);
        if (d_len < nearest_dist) { nearest_dist = d_len; nearest_idx = i; }
    }

    glm::vec2 to_goal = {m_goals[nearest_idx].pos.x - pos.x, m_goals[nearest_idx].pos.z - pos.z};
    float dist = nearest_dist;

    // Reached goal — big reward, relocate that goal, keep episode alive.
    if (dist <= Episode::goal_radius)
    {
        std::uniform_real_distribution<float> rdist(World::min * 0.8f, World::max * 0.8f);
        m_goals[nearest_idx].pos.x = rdist(m_rng);
        m_goals[nearest_idx].pos.z = rdist(m_rng);
        if (m_goals[nearest_idx].entity != entt::null && reg.all_of<Transform>(m_goals[nearest_idx].entity))
            reg.get<Transform>(m_goals[nearest_idx].entity).pos = m_goals[nearest_idx].pos;

        // Update prev_distance to nearest remaining goal after relocation.
        float new_nearest = std::numeric_limits<float>::max();
        for (const auto &g : m_goals)
        {
            glm::vec2 d = {g.pos.x - pos.x, g.pos.z - pos.z};
            new_nearest = std::min(new_nearest, glm::length(d));
        }
        m_prev_distance   = new_nearest;
        m_time_since_goal = 0.f;

        // Invalidate sightings after goal relocation.
        for (auto &ray_buf : m_sightings)
            for (auto &sg : ray_buf)
                sg = Sighting{};
        m_sight_head.fill(0);

        return 10.f;
    }

    // TODO: add a penalty for early episode termination (e.g. falling off the map) to encourage the agent to learn to avoid that.

    // Small shaping reward: delta distance (positive when moving closer).
    float shaping = m_prev_distance - dist;
    m_prev_distance = dist;

    // Look-at reward: cos(angle between facing direction and bearing to goal).
    float goal_world_angle = glm::degrees(std::atan2(to_goal.x, to_goal.y));
    float angle_diff = goal_world_angle - looking_angle;
    float look_alignment = std::cos(glm::radians(angle_diff));
    float strafe_pen = (m_pending_action == 2 || m_pending_action == 3) ? Reward::strafe_penalty : 0.f;

    return shaping + Reward::look_weight * look_alignment - Reward::action_penalty - strafe_pen;
}

bool NavigationEnv::is_done() const
{
    return m_done
        || m_step_count      >= Episode::max_steps
        || m_elapsed_seconds >= Episode::max_seconds
        || m_time_since_goal >= m_current_goal_time_limit;
}

void NavigationEnv::apply_action(int action)
{
    m_pending_action = action;
    ++m_step_count;
}

// void NavigationEnv::move_goal_random()
// {
//     std::uniform_real_distribution<float> dist(MAP_MIN * 0.8f, MAP_MAX * 0.8f);
//     float gx = dist(m_rng);
//     float gz = dist(m_rng);
//     m_goal_pos = {gx, m_goal_pos.y, gz};
//     // NOTE: the caller is responsible for writing m_goal_pos to the goal
//     // entity's Transform while holding the registry lock (to avoid deadlock).

//     // Invalidate all sightings — their positions are relative to the old
//     // world state and are no longer useful context for the new goal.
//     for (auto &ray_buf : m_sightings)
//         for (auto &sg : ray_buf)
//             sg = Sighting{};
//     m_sight_head.fill(0);
// }

Seeker::Action NavigationEnv::pending_action() const
{
    // Map 0-5 → Seeker::Action enum values.
    switch (m_pending_action)
    {
    case 0:
        return Seeker::MOVE_FORWARD;
    case 1:
        return Seeker::MOVE_BACKWARD;
    case 2:
        return Seeker::STRAFE_LEFT;
    case 3:
        return Seeker::STRAFE_RIGHT;
    case 4:
        return Seeker::TURN_LEFT;
    case 5:
        return Seeker::TURN_RIGHT;
    default:
        return Seeker::NONE;
    }
}

std::unordered_map<std::string, float> NavigationEnv::get_env_data() const
{
    return {
        {"current_goal_time_limit", m_current_goal_time_limit},
    };
}

void NavigationEnv::set_env_data(const std::unordered_map<std::string, float> &data)
{
    auto it = data.find("current_goal_time_limit");
    if (it != data.end())
        m_current_goal_time_limit = it->second;
}

