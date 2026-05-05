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
    ++m_episode_count;
    m_step_count       = 0;
    m_elapsed_seconds  = 0.f;
    m_time_since_goal  = 0.f;
    m_done             = false;
    m_pending_action   = 0;
    m_prev_action      = -1;

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
        std::uniform_real_distribution<float> rdist(World::min * World::spawn_margin, World::max * World::spawn_margin);
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
    m_was_goal_visible        = false;
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
    bool in_air;
    bool is_jumping;
    {
        auto [lock, reg] = ((const EntityManager&)*m_em).get_registry();
        if (!reg.all_of<Transform, Seeker::Data, RigidBody>(m_seeker))
            return obs;

        pos           = reg.get<Transform>(m_seeker).pos;
        looking_angle = reg.get<Seeker::Data>(m_seeker).looking_angle;
        body_id       = reg.get<RigidBody>(m_seeker).id;
        in_air        = reg.get<Seeker::Data>(m_seeker).in_air;
        is_jumping    = reg.get<Seeker::Data>(m_seeker).is_jumping;
        m_in_air      = in_air;

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

    // Find nearest goal — cached in m_nearest_goal_idx/dist for reuse in compute_reward.
    glm::vec3 goal_pos{0.f};
    m_nearest_goal_idx  = 0;
    m_nearest_goal_dist = std::numeric_limits<float>::max();
    for (int i = 0; i < (int)m_goals.size(); ++i)
    {
        glm::vec2 d = {m_goals[i].pos.x - pos.x, m_goals[i].pos.z - pos.z};
        float d_len = glm::length(d);
        if (d_len < m_nearest_goal_dist) { m_nearest_goal_dist = d_len; m_nearest_goal_idx = i; }
    }
    if (!m_goals.empty())
        goal_pos = m_goals[m_nearest_goal_idx].pos;

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

    // --- LOS check + goal info: one ray per goal, derive nearest-goal visibility from results ---
    NavIgnoreSelfFilter self_filter(body_id);
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

    // Derive nearest-goal visibility from goal_infos (reuses already-cast rays,
    // avoids a duplicate LOS cast for the nearest goal).
    bool goal_visible = !m_goals.empty() && goal_infos[m_nearest_goal_idx].visible;
    if (goal_visible)
    {
        if (!m_was_goal_visible)
            Debug::Log("Goal spotted (idx=" + std::to_string(m_nearest_goal_idx)
                + ", dist=" + std::to_string(m_nearest_goal_dist) + ")");
        m_last_known_goal_pos     = goal_pos;
        m_time_since_goal_visible = 0.f;
    }
    else
    {
        m_time_since_goal_visible += dt;
    }
    m_was_goal_visible = goal_visible;

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
            write_sighting(hit_offset, hit.mBodyID, SightingConstants::wall_interest);
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
                write_sighting(gi.offset, gi.body_id, SightingConstants::goal_interest);
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

    // Observation layout — derived from structure constants so indices
    // stay in sync automatically if any section size changes.
    constexpr int section_E_base   = Raycast::num_rays * 2 + Raycast::num_rays * SightingConstants::history * 4;
    constexpr int section_D_base   = section_E_base + Raycast::num_ground_rays;
    constexpr int idx_goal_dir_x   = section_D_base + 0;
    constexpr int idx_goal_dir_z   = section_D_base + 1;
    constexpr int idx_vel_x        = section_D_base + 2;
    constexpr int idx_vel_z        = section_D_base + 3;
    constexpr int idx_vel_y        = section_D_base + 4;
    constexpr int idx_ang_vel_y    = section_D_base + 5;
    constexpr int idx_angle_sin    = section_D_base + 6;
    constexpr int idx_angle_cos    = section_D_base + 7;
    constexpr int idx_goal_dist    = section_D_base + 8;
    constexpr int idx_last_known_x = section_D_base + 9;
    constexpr int idx_last_known_z = section_D_base + 10;
    constexpr int idx_goal_visible = section_D_base + 11;
    constexpr int idx_staleness    = section_D_base + 12;
    constexpr int idx_in_air       = section_D_base + 13;
    constexpr int idx_is_jumping   = section_D_base + 14;
    constexpr int idx_prev_action  = section_D_base + 15;

    // --- Section E: ground / edge-detection rays (section_E_base … section_E_base + num_ground_rays - 1) ---
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
            obs[section_E_base + i] = had_hit ? hit.mFraction : 1.0f;
        }
    }

    // --- Section D (section_D_base … section_D_base + num_actions + 12) ---
    const float diag = std::sqrt(2.f) * (World::max - World::min);

    // idx_goal_dir_x/z: normalised direction to LAST KNOWN goal in seeker-local space.
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
    obs[idx_goal_dir_x] = local_goal.x / local_len;
    obs[idx_goal_dir_z] = local_goal.y / local_len;

    // idx_vel_x/z/y, idx_ang_vel_y: seeker velocity, normalised by max_speed / max_angular_speed.
    JPH::Vec3 vel = bi.GetLinearVelocity(body_id);
    obs[idx_vel_x]     = vel.GetX() / Reward::max_speed;
    obs[idx_vel_z]     = vel.GetZ() / Reward::max_speed;
    obs[idx_vel_y]     = vel.GetY() / Reward::max_speed;
    obs[idx_ang_vel_y] = bi.GetAngularVelocity(body_id).GetY() / Reward::max_angular_speed;
    m_speed_xz = std::sqrt(vel.GetX() * vel.GetX() + vel.GetZ() * vel.GetZ());

    // idx_angle_sin/cos: looking angle encoded as (sin, cos) to avoid the ±180° discontinuity.
    obs[idx_angle_sin] = std::sin(look_rad);
    obs[idx_angle_cos] = std::cos(look_rad);

    // idx_goal_dist: distance to last known goal, normalised.
    obs[idx_goal_dist] = std::min(last_known_dist / diag, 1.f);

    // idx_last_known_x/z: last known goal world position normalised.
    obs[idx_last_known_x] = (m_last_known_goal_pos.x - World::min) / (World::max - World::min);
    obs[idx_last_known_z] = (m_last_known_goal_pos.z - World::min) / (World::max - World::min);

    // idx_goal_visible: goal currently visible (1.0 = yes, 0.0 = occluded by wall).
    obs[idx_goal_visible] = goal_visible ? 1.f : 0.f;

    // idx_staleness: staleness of last known position, normalised [0, 1] over max_stale seconds.
    //      0 = just saw it, 1 = not seen for max_stale+ seconds.
    obs[idx_staleness] = std::min(m_time_since_goal_visible / SightingConstants::max_stale, 1.f);

    // idx_in_air / idx_is_jumping: binary flags so the agent knows when JUMP is valid.
    obs[idx_in_air]     = in_air     ? 1.f : 0.f;
    obs[idx_is_jumping] = is_jumping ? 1.f : 0.f;

    // idx_prev_action + action index: previous action one-hot (num_actions floats).
    // All zeros at the start of an episode (m_prev_action == -1).
    if (m_prev_action >= 0 && m_prev_action < Sizes::num_actions)
        obs[idx_prev_action + m_prev_action] = 1.f;

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
    // Subtract the action penalty for all remaining steps so that falling early
    // is never cheaper than falling late: early termination forfeits potential
    // goal rewards AND is penalised for the steps it won't take.
    if (pos.y < World::death_height)
    {
        int remaining = Episode::max_steps - m_step_count;
        return Reward::fall_penalty - static_cast<float>(remaining) * Reward::action_penalty;
    }

    // Edge danger penalty: linear ramp from 0 at edge_danger_dist to
    // edge_danger_penalty at the boundary, applied every step the agent
    // is in the danger zone. Stacks with shaping rewards so the agent
    // is still incentivised to reach goals near the edge.
    float edge_penalty = 0.f;
    {
        float dist_to_edge = std::min({
            pos.x - World::min,
            World::max - pos.x,
            pos.z - World::min,
            World::max - pos.z
        });
        if (dist_to_edge < Reward::edge_danger_dist)
        {
            float t = 1.f - dist_to_edge / Reward::edge_danger_dist; // 0 at threshold, 1 at boundary
            edge_penalty = -Reward::edge_danger_penalty * t;
        }
    }

    // Use nearest goal cached by get_observation this step.
    int   nearest_idx  = m_nearest_goal_idx;
    float dist         = m_nearest_goal_dist;
    glm::vec2 to_goal  = {m_goals[nearest_idx].pos.x - pos.x, m_goals[nearest_idx].pos.z - pos.z};

    // Reached goal — big reward, relocate that goal, keep episode alive.
    if (dist <= Episode::goal_radius)
    {
        std::uniform_real_distribution<float> rdist(World::min * World::spawn_margin, World::max * World::spawn_margin);
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
        // Speed bonus: full bonus if reached within quick_threshold, linear ramp to 0 beyond it.
        float speed_bonus = Reward::goal_speed_bonus
            * std::max(0.f, 1.f - m_time_since_goal / Reward::goal_quick_threshold);

        m_time_since_goal = 0.f;

        // Invalidate sightings after goal relocation.
        for (auto &ray_buf : m_sightings)
            for (auto &sg : ray_buf)
                sg = Sighting{};
        m_sight_head.fill(0);

        return Reward::goal_reward + speed_bonus;
    }

    // Small shaping reward: delta distance (positive when moving closer).
    // Skip on step 1 (m_prev_distance is 0 — would produce a spurious large negative).
    // Skip while airborne — the agent can gain positive shaping by falling toward a goal,
    // which would make jumping off edges appear attractive.
    float shaping = 0.f;
    if (m_step_count > 1 && !m_in_air)
        shaping = m_prev_distance - dist;
    m_prev_distance = dist;

    // Look-at reward: cos(angle between facing direction and bearing to goal).
    float goal_world_angle = glm::degrees(std::atan2(to_goal.x, to_goal.y));
    float angle_diff = goal_world_angle - looking_angle;
    float look_alignment = std::cos(glm::radians(angle_diff));
    float strafe_pen = (pending_action() == Seeker::STRAFE_LEFT || pending_action() == Seeker::STRAFE_RIGHT) ? Reward::strafe_penalty : 0.f;
    float fwd_bonus   = (pending_action() == Seeker::MOVE_FORWARD && !m_in_air) ? Reward::forward_reward : 0.f;

    // Stuck penalty: if the agent is trying to move but barely going anywhere,
    // it's probably wedged against a wall. Penalise to encourage going around.
    float stuck_pen = 0.f;
    if (m_step_count > 1 && !m_in_air && m_speed_xz < Reward::stuck_speed_threshold)
    {
        Seeker::Action a = pending_action();
        if (a == Seeker::MOVE_FORWARD || a == Seeker::MOVE_BACKWARD
            || a == Seeker::STRAFE_LEFT || a == Seeker::STRAFE_RIGHT)
            stuck_pen = -Reward::stuck_penalty;
    }

    return shaping + Reward::look_weight * look_alignment - Reward::action_penalty - strafe_pen + edge_penalty + stuck_pen + fwd_bonus;
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
    m_prev_action    = m_pending_action;
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
    case 6:
        return Seeker::JUMP;
    default:
        return Seeker::NONE;
    }
}

std::unordered_map<std::string, float> NavigationEnv::get_env_data() const
{
    return {
        {"current_goal_time_limit", m_current_goal_time_limit},
        {"episode_count",           static_cast<float>(m_episode_count)},
    };
}

std::unordered_map<std::string, float> NavigationEnv::get_config_data() const
{
    return {
        {"min_goal_search_seconds", Episode::min_goal_search_seconds},
        {"max_goal_search_seconds", Episode::max_goal_search_seconds},
    };
}

void NavigationEnv::set_env_data(const std::unordered_map<std::string, float> &data)
{
    auto it = data.find("current_goal_time_limit");
    if (it != data.end())
        m_current_goal_time_limit = it->second;

    auto it2 = data.find("episode_count");
    if (it2 != data.end())
        m_episode_count = static_cast<int>(it2->second);
}

