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

#ifdef TRACY_ENABLE
#include <tracy/Tracy.hpp>
#else
#define ZoneScoped
#endif

using namespace Magic;

std::vector<Goal> NavigationEnv::s_goals;

Goal NavigationEnv::get_next_goal(int idx)
{
    on_next_goal(); // callback for user code to update the static pattern if desired (e.g. to make it dynamic or procedurally generated), but default is to just loop through the originally loaded pattern
    Goal goal;
    goal.pos = glm::vec3{0.f};
    goal.entity = entt::null;
    
    if (!NavigationEnv::get_goals().empty())
        goal.pos = NavigationEnv::get_goals()[idx % NavigationEnv::get_goals().size()].pos;
    return goal;
}

// Ignore the seeker body when raycasting (same pattern as seeker.cpp).
class NavIgnoreSelfFilter : public JPH::BodyFilter
{
public:
    NavIgnoreSelfFilter(JPH::BodyID self, const std::vector<JPH::BodyID> &extra)
        : m_self(self), m_extra(extra) {}
    bool ShouldCollide(const JPH::BodyID &id) const override
    {
        if (id == m_self)
            return false;
        for (const auto &x : m_extra)
            if (id == x)
                return false;
        return true;
    }

private:
    JPH::BodyID m_self;
    const std::vector<JPH::BodyID> &m_extra;
};

void NavigationEnv::bind(EntityManager &em, Entity seeker, std::vector<Entity> goals)
{
    m_em = &em;
    m_seeker = seeker;

    m_goals.clear();
    {
        auto [lock, reg] = em.get_registry();

        if (s_goals.empty())
        {
            s_goals.reserve(goals.size());
            for (Entity e : goals)
            {
                Goal g;
                g.entity = entt::null;
                if (e != entt::null && reg.all_of<Transform>(e))
                    g.pos = reg.get<Transform>(e).pos;
                s_goals.push_back(g);
            }
        }

        for (int i = 0; i < (int)goals.size(); ++i)
        {
            Goal g = get_next_goal(i);
            g.entity = goals[i];
            if (g.entity != entt::null && reg.all_of<Transform>(g.entity))
            {
                auto &transform = reg.get<Transform>(g.entity);
                transform.pos = g.pos;
            }
            m_goals.push_back(g);
        }
    }
}

std::vector<float> NavigationEnv::reset()
{
    ZoneScoped;
    Debug::Log("NavigationEnv reset");
    ++m_episode_count;
    m_step_count = 0;
    m_goals_this_episode = 0;
    m_elapsed_seconds = 0.f;
    m_time_since_goal = 0.f;
    m_done = false;
    m_pending_action = -1;
    m_prev_action = -1;

    if (m_episode_count == 1)
        m_current_goal_time_limit = m_config.curriculum.max_goal_search_seconds;
    else
        m_current_goal_time_limit = std::max(m_config.curriculum.min_goal_search_seconds,
                                             m_current_goal_time_limit - m_config.curriculum.search_time_fall_rate);

    // Clear sighting buffers.
    for (auto &ray_buf : m_sightings)
        for (auto &sg : ray_buf)
            sg = Sighting{};

    m_sight_head.fill(0);
    m_prev_actions.fill(-1);

    // Randomize all goal positions.
    if (m_em && !m_goals.empty())
    {
        auto [lock, reg] = m_em->get_registry();
        m_goals.clear();
        for (int i = 0; i < (int)m_config.world.num_concurrent_goals; ++i)
        {
            m_goals.push_back(get_next_goal(i));
            if (m_goals[i].entity != entt::null && reg.all_of<Transform>(m_goals[i].entity))
            {
                auto &transform = reg.get<Transform>(m_goals[i].entity);
                transform.pos = m_goals[i].pos;
            }
        }
        // std::uniform_real_distribution<float> rdist(m_config.world.min * m_config.world.spawn_margin, m_config.world.max * m_config.world.spawn_margin);
        // for (int i = 0; i < (int)m_goals.size(); ++i)
        // {
            // glm::vec3 template_pos = get_next_goal(i).pos;
            // m_goals[i].pos = {rdist(m_rng), template_pos.y, rdist(m_rng)};
            // if (m_goals[i].entity != entt::null && reg.all_of<Transform>(m_goals[i].entity))
            // {
            //     auto &transform = reg.get<Transform>(m_goals[i].entity);
            //     transform.pos = m_goals[i].pos;
            // }
        // }
    }

    // Return a zero observation to keep Python happy.
    m_prev_distance = 0.f;
    m_last_known_goal_pos = glm::vec3{0.f};
    m_time_since_goal_visible = SightingConstants::max_stale;
    m_was_goal_visible = false;
    return std::vector<float>(NavigationEnv::Sizes::num_states, 0.f);
}

std::vector<float> NavigationEnv::get_observation(float dt)
{
    ZoneScoped;
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
        auto [lock, reg] = ((const EntityManager &)*m_em).get_registry();
        if (!reg.all_of<Transform, Seeker::Data, RigidBody>(m_seeker))
            return obs;

        pos = reg.get<Transform>(m_seeker).pos;
        looking_angle = reg.get<Seeker::Data>(m_seeker).looking_angle;
        body_id = reg.get<RigidBody>(m_seeker).id;
        in_air = reg.get<Seeker::Data>(m_seeker).in_air;
        is_jumping = reg.get<Seeker::Data>(m_seeker).is_jumping;
        m_in_air = in_air;

        for (auto &g : m_goals)
            if (g.entity != entt::null && reg.all_of<Transform>(g.entity))
                g.pos = reg.get<Transform>(g.entity).pos;
    } // shared lock released here

    // Fell off the map — end the episode immediately.
    if (pos.y < m_config.world.death_height)
        m_done = true;

    // Accumulate elapsed time for the episode timeout check.
    m_elapsed_seconds += dt;
    m_time_since_goal += dt;

    Physics &physics = GetEngine().get_physics();
    JPH::BodyInterface &bi = physics.physics_system.GetBodyInterface();

    // Find nearest goal — cached in m_nearest_goal_idx/dist for reuse in compute_reward.
    glm::vec3 goal_pos{0.f};
    m_nearest_goal_idx = 0;
    m_nearest_goal_dist = std::numeric_limits<float>::max();
    for (int i = 0; i < (int)m_goals.size(); ++i)
    {
        glm::vec2 d = {m_goals[i].pos.x - pos.x, m_goals[i].pos.z - pos.z};
        float d_len = glm::length(d);
        if (d_len < m_nearest_goal_dist)
        {
            m_nearest_goal_dist = d_len;
            m_nearest_goal_idx = i;
        }
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
            if (sg.age >= m_config.sighting.max_age)
                sg = Sighting{};
        }
    }

    // --- LOS check + goal info: one ray per goal, derive nearest-goal visibility from results ---
    NavIgnoreSelfFilter self_filter(body_id, m_ignored_bodies);
    struct GoalInfo
    {
        bool visible;
        float bearing;    // world-space degrees
        glm::vec3 offset; // from seeker
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
        goal_infos.push_back({g_vis,
                              glm::degrees(std::atan2(to_g.x, to_g.y)),
                              {to_g.x, m_goals[gi].pos.y - pos.y, to_g.y},
                              gi});
    }

    // Derive nearest-goal visibility from goal_infos (reuses already-cast rays,
    // avoids a duplicate LOS cast for the nearest goal).
    bool goal_visible = !m_goals.empty() && goal_infos[m_nearest_goal_idx].visible;
    if (goal_visible)
    {
        // if (!m_was_goal_visible)
        //     Debug::Log("Goal spotted (idx=" + std::to_string(m_nearest_goal_idx)
        //         + ", dist=" + std::to_string(m_nearest_goal_dist) + ")");
        m_last_known_goal_pos = goal_pos;
        m_time_since_goal_visible = 0.f;
    }
    else
    {
        m_time_since_goal_visible += dt;
    }
    m_was_goal_visible = goal_visible;

    // --- Pass 2: raycast + record + build obs (registry lock not held) ---
    // 120° FOV: 13 rays spanning -60° to +60° relative to looking direction.
    const float SPREAD = m_config.raycast.fov_deg / (Raycast::num_rays - 1);

    for (int i = 0; i < Raycast::num_rays; ++i)
    {
        float angle_deg = looking_angle + (i - Raycast::num_rays / 2) * SPREAD;
        float rad = glm::radians(angle_deg);
        JPH::Vec3 dir(std::sin(rad) * m_config.raycast.ray_len, 0.f, std::cos(rad) * m_config.raycast.ray_len);

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
            write_sighting(hit_offset, hit.mBodyID, m_config.sighting.wall_interest);
        }

        // Record goal sightings (interest = 0.9) for any visible goal whose
        // bearing falls within this ray bin. Goals overwrite ray_interest last,
        // so Section B reflects the goal over any wall on the same ray.
        for (const auto &gi : goal_infos)
        {
            if (!gi.visible)
                continue;
            float diff = std::abs(gi.bearing - angle_deg);
            if (diff > Angles::half_circle_deg)
                diff = Angles::full_circle_deg - diff;
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
    constexpr int section_E_base = Raycast::num_rays * 2 + Raycast::num_rays * SightingConstants::history * 4;
    constexpr int section_D_base = section_E_base + Raycast::num_ground_rays;
    constexpr int idx_goal_dir_x = section_D_base + 0;
    constexpr int idx_goal_dir_z = section_D_base + 1;
    constexpr int idx_vel_x = section_D_base + 2;
    constexpr int idx_vel_z = section_D_base + 3;
    constexpr int idx_vel_y = section_D_base + 4;
    constexpr int idx_ang_vel_y = section_D_base + 5;
    constexpr int idx_angle_sin = section_D_base + 6;
    constexpr int idx_angle_cos = section_D_base + 7;
    constexpr int idx_goal_dist = section_D_base + 8;
    constexpr int idx_last_known_x = section_D_base + 9;
    constexpr int idx_last_known_z = section_D_base + 10;
    constexpr int idx_goal_visible = section_D_base + 11;
    constexpr int idx_staleness = section_D_base + 12;
    constexpr int idx_in_air = section_D_base + 13;
    constexpr int idx_is_jumping = section_D_base + 14;
    constexpr int idx_prev_action = section_D_base + 15;

    // --- Section E: ground / edge-detection rays (section_E_base … section_E_base + num_ground_rays - 1) ---
    // NUM_GROUND_RAYS downward-pitched rays spanning the forward 120° FOV.
    // 1.0 = no ground hit within RAY_LEN = edge of map nearby.
    {
        const float gSpread = m_config.raycast.fov_deg / (Raycast::num_ground_rays - 1);
        const float pitchRad = glm::radians(m_config.raycast.ground_pitch);
        const float cosPitch = std::cos(pitchRad);
        const float sinPitch = std::sin(pitchRad); // positive, applied as -y
        for (int i = 0; i < Raycast::num_ground_rays; ++i)
        {
            float angle_deg = looking_angle + (i - Raycast::num_ground_rays / 2) * gSpread;
            float rad = glm::radians(angle_deg);
            JPH::Vec3 dir(
                std::sin(rad) * cosPitch * m_config.raycast.ray_len,
                -sinPitch * m_config.raycast.ray_len,
                std::cos(rad) * cosPitch * m_config.raycast.ray_len);
            JPH::RRayCast ray{JPH::Vec3(pos.x, pos.y, pos.z), dir};
            JPH::RayCastResult hit;
            bool had_hit = physics.physics_system.GetNarrowPhaseQuery()
                               .CastRay(ray, hit, {}, {}, self_filter);
            obs[section_E_base + i] = had_hit ? hit.mFraction : 1.0f;
        }
    }

    // --- Section D (section_D_base … section_D_base + num_actions + 12) ---
    const float diag = std::sqrt(2.f) * (m_config.world.max - m_config.world.min);

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
    obs[idx_vel_x] = vel.GetX() / m_config.reward.max_speed;
    obs[idx_vel_z] = vel.GetZ() / m_config.reward.max_speed;
    obs[idx_vel_y] = vel.GetY() / m_config.reward.max_speed;
    obs[idx_ang_vel_y] = bi.GetAngularVelocity(body_id).GetY() / m_config.reward.max_angular_speed;
    m_speed_xz = std::sqrt(vel.GetX() * vel.GetX() + vel.GetZ() * vel.GetZ());

    // idx_angle_sin/cos: looking angle encoded as (sin, cos) to avoid the ±180° discontinuity.
    obs[idx_angle_sin] = std::sin(look_rad);
    obs[idx_angle_cos] = std::cos(look_rad);

    // idx_goal_dist: distance to last known goal, normalised.
    obs[idx_goal_dist] = std::min(last_known_dist / diag, 1.f);

    // idx_last_known_x/z: last known goal world position normalised.
    obs[idx_last_known_x] = (m_last_known_goal_pos.x - m_config.world.min) / (m_config.world.max - m_config.world.min);
    obs[idx_last_known_z] = (m_last_known_goal_pos.z - m_config.world.min) / (m_config.world.max - m_config.world.min);

    // idx_goal_visible: goal currently visible (1.0 = yes, 0.0 = occluded by wall).
    obs[idx_goal_visible] = goal_visible ? 1.f : 0.f;

    // idx_staleness: staleness of last known position, normalised [0, 1] over max_stale seconds.
    //      0 = just saw it, 1 = not seen for max_stale+ seconds.
    obs[idx_staleness] = std::min(m_time_since_goal_visible / SightingConstants::max_stale, 1.f);

    // idx_in_air / idx_is_jumping: binary flags so the agent knows when JUMP is valid.
    obs[idx_in_air] = in_air ? 1.f : 0.f;
    obs[idx_is_jumping] = is_jumping ? 1.f : 0.f;

    // Previous action history: one-hot blocks for the last N actions.
    for (int h = 0; h < Sizes::action_history; ++h)
    {
        int action = m_prev_actions[h];
        if (action >= 0 && action < Sizes::num_actions)
            obs[idx_prev_action + h * Sizes::num_actions + action] = 1.f;
    }

    return obs;
}

float NavigationEnv::compute_reward()
{
    ZoneScoped;
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
    if (pos.y < m_config.world.death_height)
    {
        int remaining = m_config.episode.max_steps - m_step_count;
        return m_config.reward.fall_penalty - static_cast<float>(remaining) * m_config.reward.action_penalty;
    }

    // Edge danger penalty: linear ramp from 0 at edge_danger_dist to
    // edge_danger_penalty at the boundary, applied every step the agent
    // is in the danger zone. Stacks with shaping rewards so the agent
    // is still incentivised to reach goals near the edge.
    float edge_penalty = 0.f;
    {
        float dist_to_edge = std::min({pos.x - m_config.world.min,
                                       m_config.world.max - pos.x,
                                       pos.z - m_config.world.min,
                                       m_config.world.max - pos.z});
        if (dist_to_edge < m_config.reward.edge_danger_dist)
        {
            float t = 1.f - dist_to_edge / m_config.reward.edge_danger_dist; // 0 at threshold, 1 at boundary
            edge_penalty = -m_config.reward.edge_danger_penalty * t;
        }
    }

    // Use nearest goal cached by get_observation this step.
    int nearest_idx = m_nearest_goal_idx;
    float dist = m_nearest_goal_dist;
    glm::vec2 to_goal = {m_goals[nearest_idx].pos.x - pos.x, m_goals[nearest_idx].pos.z - pos.z};

    // Reached goal — big reward, relocate that goal, keep episode alive.
    if (dist <= m_config.world.goal_radius)
    {
        // TODO: make sure that this pulls from the last local goal from the static pattern, not just the next one in the pattern, so that if the pattern is dynamic / procedural the same goal doesn't just respawn in the same place but actually moves to a new location. This also means that the static pattern can be smaller than num_concurrent_goals if desired, since get_next_goal will advance it on every collection.
        get_next_goal(nearest_idx); // advance the static pattern so the same goal doesn't respawn in the same place

        Debug::Log("Seeker_" + std::to_string(m_agent_id) + " collected goal " + std::to_string(nearest_idx) +
                   " (goals this episode: " + std::to_string(++m_goals_this_episode) + ")");

        // Update prev_distance and last-known goal to the nearest current goal after relocation.
        float new_nearest = std::numeric_limits<float>::max();
        glm::vec3 new_nearest_goal_pos = m_goals[nearest_idx].pos;
        for (const auto &g : m_goals)
        {
            glm::vec2 d = {g.pos.x - pos.x, g.pos.z - pos.z};
            float len = glm::length(d);
            if (len < new_nearest)
            {
                new_nearest = len;
                new_nearest_goal_pos = g.pos;
            }
        }
        m_prev_distance = new_nearest;
        m_last_known_goal_pos = new_nearest_goal_pos;
        m_time_since_goal_visible = SightingConstants::max_stale;
        m_was_goal_visible = false;

        // Speed bonus: full bonus if reached within quick_threshold, linear ramp to 0 beyond it.
        float speed_bonus = m_config.reward.goal_speed_bonus * std::max(0.f, 1.f - m_time_since_goal / m_config.reward.goal_quick_threshold);

        m_time_since_goal = 0.f;

        // Invalidate only sightings for the collected goal.
        for (auto &ray_buf : m_sightings)
        {
            for (auto &sg : ray_buf)
            {
                if (sg.valid && std::holds_alternative<int>(sg.body_id) && std::get<int>(sg.body_id) == nearest_idx)
                {
                    sg = Sighting{};
                    break; // only one sighting per ray can match the collected goal, so stop searching this ray after invalidating
                }
            }
        }

        return m_config.reward.goal_reward + speed_bonus;
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
    float strafe_pen = (pending_action() == Seeker::STRAFE_LEFT || pending_action() == Seeker::STRAFE_RIGHT) ? m_config.reward.strafe_penalty : 0.f;
    float fwd_bonus = (pending_action() == Seeker::MOVE_FORWARD && !m_in_air) ? m_config.reward.forward_reward : 0.f;

    // Stuck penalty: if the agent is trying to move but barely going anywhere,
    // it's probably wedged against a wall. Penalize to encourage going around.
    float stuck_pen = 0.f;
    if (m_step_count > 1 && !m_in_air && m_speed_xz < m_config.reward.stuck_speed_threshold)
    {
        Seeker::Action a = pending_action();
        if (a == Seeker::MOVE_FORWARD || a == Seeker::MOVE_BACKWARD || a == Seeker::STRAFE_LEFT || a == Seeker::STRAFE_RIGHT)
            stuck_pen = -m_config.reward.stuck_penalty;
    }

    return shaping + m_config.reward.look_weight * look_alignment - m_config.reward.action_penalty - strafe_pen + edge_penalty + stuck_pen + fwd_bonus;
}

bool NavigationEnv::is_done() const
{
    return m_done || m_step_count >= m_config.episode.max_steps || m_elapsed_seconds >= m_config.episode.max_seconds || m_time_since_goal >= m_current_goal_time_limit;
}

bool NavigationEnv::is_terminal() const
{
    // Only true when the agent fell off the map — a true terminal state
    // with no continuation value. Step/time/goal-timeout truncations are
    // NOT terminal: the episode ends but the world continues, so V(s_last)
    // should be used as the GAE bootstrap, not 0.
    return m_done;
}

void NavigationEnv::apply_action(int action)
{
    for (int i = Sizes::action_history - 1; i > 0; --i)
        m_prev_actions[i] = m_prev_actions[i - 1];
    m_prev_actions[0] = m_pending_action;

    m_prev_action = m_pending_action;
    m_pending_action = action;
    ++m_step_count;
}

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
    case 7:
        return Seeker::NONE;
    default:
        return Seeker::NONE;
    }
}

std::unordered_map<std::string, float> NavigationEnv::get_env_data() const
{
    return {
        {"current_goal_time_limit", m_current_goal_time_limit},
        {"episode_count", static_cast<float>(m_episode_count)},
    };
}

std::unordered_map<std::string, std::unordered_map<std::string, float>> NavigationEnv::get_config_data() const
{
    return {
        {"sizes", {{"num_states", Sizes::num_states}, {"num_actions", Sizes::num_actions}, {"action_history", Sizes::action_history}}},
        {"raycast", {{"num_rays", Raycast::num_rays}, {"num_ground_rays", Raycast::num_ground_rays}, {"ray_len", m_config.raycast.ray_len}, {"ground_pitch", m_config.raycast.ground_pitch}, {"fov_deg", m_config.raycast.fov_deg}}},
        {"sighting", {{"history", SightingConstants::history}, {"max_age", SightingConstants::max_age}, {"max_stale", SightingConstants::max_stale}, {"wall_interest", m_config.sighting.wall_interest}, {"goal_interest", m_config.sighting.goal_interest}}},
        {"world", {{"min", m_config.world.min}, {"max", m_config.world.max}, {"death_height", m_config.world.death_height}, {"num_concurrent_goals", m_config.world.num_concurrent_goals}, {"goal_radius", m_config.world.goal_radius}}},
        {"episode", {{"max_steps", static_cast<float>(m_config.episode.max_steps)}, {"max_seconds", m_config.episode.max_seconds}}},
        {"curriculum", {{"max_goal_search_seconds", m_config.curriculum.max_goal_search_seconds}, {"min_goal_search_seconds", m_config.curriculum.min_goal_search_seconds}, {"search_time_fall_rate", m_config.curriculum.search_time_fall_rate}, {"boundary_wall_episodes", static_cast<float>(m_config.curriculum.boundary_wall_episodes)}}},
        {"reward", {{"goal_reward", m_config.reward.goal_reward}, {"goal_speed_bonus", m_config.reward.goal_speed_bonus}, {"goal_quick_threshold", m_config.reward.goal_quick_threshold}, {"look_weight", m_config.reward.look_weight}, {"action_penalty", m_config.reward.action_penalty}, {"strafe_penalty", m_config.reward.strafe_penalty}, {"edge_danger_dist", m_config.reward.edge_danger_dist}, {"edge_danger_penalty", m_config.reward.edge_danger_penalty}, {"stuck_speed_threshold", m_config.reward.stuck_speed_threshold}, {"stuck_penalty", m_config.reward.stuck_penalty}, {"forward_reward", m_config.reward.forward_reward}, {"max_speed", m_config.reward.max_speed}, {"max_angular_speed", m_config.reward.max_angular_speed}, {"fall_penalty", m_config.reward.fall_penalty}}},
        {"angles", {{"full_circle_deg", Angles::full_circle_deg}, {"half_circle_deg", Angles::half_circle_deg}, {"deg_to_rad", Angles::deg_to_rad}}},
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

void NavigationEnv::set_config_data(const std::string &category, const std::string &name, float value)
{
    if (category == "raycast")
    {
        if (name == "ray_len")
            m_config.raycast.ray_len = value;
        else if (name == "ground_pitch")
            m_config.raycast.ground_pitch = value;
        else if (name == "fov_deg")
            m_config.raycast.fov_deg = value;
    }
    else if (category == "sighting")
    {
        if (name == "wall_interest")
            m_config.sighting.wall_interest = value;
        else if (name == "goal_interest")
            m_config.sighting.goal_interest = value;
    }
    else if (category == "world")
    {
        if (name == "min")
            m_config.world.min = value;
        else if (name == "max")
            m_config.world.max = value;
        else if (name == "death_height")
            m_config.world.death_height = value;
        else if (name == "num_concurrent_goals")
            m_config.world.num_concurrent_goals = static_cast<int>(value);
        else if (name == "goal_radius")
            m_config.world.goal_radius = value;
    }
    else if (category == "episode")
    {
        if (name == "max_steps")
            m_config.episode.max_steps = static_cast<int>(value);
        else if (name == "max_seconds")
            m_config.episode.max_seconds = value;
    }
    else if (category == "curriculum")
    {
        if (name == "max_goal_search_seconds")
            m_config.curriculum.max_goal_search_seconds = value;
        else if (name == "min_goal_search_seconds")
            m_config.curriculum.min_goal_search_seconds = value;
        else if (name == "search_time_fall_rate")
            m_config.curriculum.search_time_fall_rate = value;
    }
    else if (category == "reward")
    {
        if (name == "fall_penalty")
            m_config.reward.fall_penalty = value;
        else if (name == "goal_reward")
            m_config.reward.goal_reward = value;
        else if (name == "look_weight")
            m_config.reward.look_weight = value;
        else if (name == "action_penalty")
            m_config.reward.action_penalty = value;
        else if (name == "strafe_penalty")
            m_config.reward.strafe_penalty = value;
        else if (name == "forward_reward")
            m_config.reward.forward_reward = value;
        else if (name == "goal_speed_bonus")
            m_config.reward.goal_speed_bonus = value;
        else if (name == "goal_quick_threshold")
            m_config.reward.goal_quick_threshold = value;
        else if (name == "max_speed")
            m_config.reward.max_speed = value;
        else if (name == "max_angular_speed")
            m_config.reward.max_angular_speed = value;
        else if (name == "edge_danger_dist")
            m_config.reward.edge_danger_dist = value;
        else if (name == "edge_danger_penalty")
            m_config.reward.edge_danger_penalty = value;
        else if (name == "stuck_speed_threshold")
            m_config.reward.stuck_speed_threshold = value;
        else if (name == "stuck_penalty")
            m_config.reward.stuck_penalty = value;
    }
}
