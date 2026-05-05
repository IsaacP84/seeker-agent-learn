#include "LearnScene.hpp"

#include "../entities/seeker.h"
#include "../navigation_env.h"
#include "../components/components.h"
#include "../util.h"

#include "factory.h"

#include <Magic!/app/application.hpp>
#include <Magic!/debug/debug.hpp>
#include <Magic!/globals.hpp>

#include <Magic!/components/physics.hpp>
#include <Magic!/components/renderable.hpp>
#include <Magic!/components/events.hpp>
#include <Magic!/components/tags.hpp>

#include <Magic!/render/meshes/cube.hpp>

#include <Jolt/Jolt.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Math/Real.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/list.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/unique_ptr.h>

#include <random>

namespace nb = nanobind;
using namespace Magic;

bool in_background = false;

LearnScene::LearnScene(Magic::SceneManager &sm)
    : Magic::Scene(sm)
{
}

LearnScene::~LearnScene() = default;

void LearnScene::onCreate()
{
    LOG("CREATING LEARN SCENE");
    Seeker::LoadModel();
}

void LearnScene::_clear_python_env_refs()
{
    if (!m_python_scene)
        return;
    nb::gil_scoped_acquire acquire;
    try
    {
        m_python_scene.attr("scene_env")  = nb::none();
        m_python_scene.attr("scene_envs") = nb::list();
        if (nb::hasattr(m_python_scene, "scene_agent"))
        {
            auto agent = m_python_scene.attr("scene_agent");
            agent.attr("_env") = nb::none();
        }
        // Force a GC pass so cyclic references don't keep wrappers alive.
        nb::module_::import_("gc").attr("collect")();
    }
    catch (...) {}
}

void LearnScene::onDestroy()
{
    _clear_python_env_refs();
}

void LearnScene::onActivate()
{
    // Bind Python functions now — the module is already in sys.modules so this is instant.
    if (!m_step_func)
    {
        nb::gil_scoped_acquire acquire;
        try
        {
            m_python_scene = nb::module_::import_("scenes.learn_scene");
            m_step_func    = m_python_scene.attr("step");
            if (nb::hasattr(m_python_scene, "feedback"))
            {
                m_feedback_func = m_python_scene.attr("feedback");
                m_has_feedback  = true;
            }
            if (nb::hasattr(m_python_scene, "round_complete"))
            {
                m_round_complete_func = m_python_scene.attr("round_complete");
                m_has_round_complete  = true;
            }
            if (nb::hasattr(m_python_scene, "IS_TRAINING"))
                m_is_training = nb::cast<bool>(m_python_scene.attr("IS_TRAINING"));
            if (nb::hasattr(m_python_scene, "NUM_AGENTS"))
                m_num_agents = nb::cast<int>(m_python_scene.attr("NUM_AGENTS"));
            // Run the logic thread as fast as possible during training.
            Application::get().use_locked_simulation_speed(!m_is_training);
            LOG("LEARN SCENE READY");
        }
        catch (nb::python_error &e)
        {
            Debug::Log(std::string("Failed to bind Python scene: ") + e.what());
        }
    }

    Application::get().SetRelativeMouseMode(true);
    EntityManager &em = *m_entity_manager;
    ResourceManager &rm = GetResourceManager();
    {
        // setup camera
        Camera &cam = GetApplication().CurrentCamera();
        cam.pos = {0, 2, 0};
    }
    makeGround(em);
    makeLight(em, "sun", {0, 10, 0}, Magic::DirectionLight{hexToRgbStoi("#ffffff"), {0.5, -1, 0.5}});

    // Walls — static obstacles the agent must navigate around.
    // pos is the base (floor-level) centre; half_extents defines size from there.
    // Map is now ±20 so all positions scaled ~2× from the old ±10 layout.
    makeWall(em, {0.f,    0.f,  0.f},  {1.5f,   3.f, 0.25f}); // smaller centre barrier
    makeWall(em, {-8.f,   0.f,  6.f},  {0.25f,  3.f, 6.f});   // perpendicular arm (left)
    makeWall(em, {8.f,    0.f, -6.f},  {0.25f,  3.f, 6.f});   // perpendicular arm (right)
    makeWall(em, {-14.f,  0.f, -8.f},  {0.25f,  3.f, 8.f});   // far left vertical
    makeWall(em, {14.f,   0.f,  8.f},  {0.25f,  3.f, 8.f});   // far right vertical

    // Curriculum boundary walls — keep the agent from falling off early in training.
    // Removed after Curriculum::boundary_wall_episodes resets by remove_boundary_walls().
    if (NavigationEnv::Curriculum::boundary_wall_episodes > 0)
    {
        const float e  = NavigationEnv::World::max;          // 20
        const float bt = 1.5f;                               // thick — robust collision
        const float hw = e + bt + 2.f;                      // half-width: span + corner overlap
        const float bh = 10.f;                              // tall enough to block the seeker
        // Centre each wall at (e + bt) so the inner face sits exactly at ±e.
        m_boundary_walls = {
            makeWall(em, {0.f, 0.f, -(e + bt)}, {hw, bh, bt}), // north
            makeWall(em, {0.f, 0.f,  (e + bt)}, {hw, bh, bt}), // south
            makeWall(em, {-(e + bt), 0.f, 0.f}, {bt, bh, hw}), // west
            makeWall(em, { (e + bt), 0.f, 0.f}, {bt, bh, hw}), // east
        };
    }
    m_walls_removed = false;

    // ---- Per-agent slots: goals + seeker + env ----
    // Each agent has its own goal entities but they are placed at the same
    // initial positions so all agents start with an equivalent layout.
    m_agents.clear();
    m_agents.reserve(m_num_agents);
    for (int a = 0; a < m_num_agents; ++a)
    {
        AgentSlot &slot = m_agents.emplace_back();
        slot.env = std::make_unique<NavigationEnv>();

        // Goals for this agent — same initial positions for every agent.
        for (int g = 0; g < NavigationEnv::Sizes::num_goals; ++g)
        {
            std::string name = "Goal_a" + std::to_string(a) + "_g" + std::to_string(g);
            Entity goal = em.create(name, {Vec3{3.f * (g + 1), 0.5f, 3.f}});
            RenderObject goal_ro = rm.copy("cool_box");
            goal_ro.model(rm.copy(goal_ro.handle()));
            em.addComponent<Magic::RenderObject>(goal, goal_ro);
            slot.goals.push_back(goal);
        }

        slot.seeker = Seeker(*m_entity_manager);
        slot.env->bind(*m_entity_manager, slot.seeker(), slot.goals);
        slot.env->set_agent_id(a);

        // Wire the seeker callback to this slot's env.
        {
            auto [lock, reg] = m_entity_manager->get_registry();
            auto &data = reg.get<Seeker::Data>(slot.seeker());
            NavigationEnv *env_ptr = slot.env.get();
            data.on_move_override = [env_ptr](EntityManager &, Entity, double) -> Seeker::Action
            {
                return env_ptr->pending_action();
            };
        }
    }

    // Give each env the body IDs of all OTHER seekers so raycasts ignore them.
    {
        auto [lock, reg] = m_entity_manager->get_registry();
        std::vector<JPH::BodyID> all_seeker_ids;
        all_seeker_ids.reserve(m_agents.size());
        for (auto &s : m_agents)
            if (s.seeker() != entt::null && reg.all_of<Magic::RigidBody>(s.seeker()))
                all_seeker_ids.push_back(reg.get<Magic::RigidBody>(s.seeker()).id);

        for (int a = 0; a < (int)m_agents.size(); ++a)
        {
            std::vector<JPH::BodyID> others;
            others.reserve(all_seeker_ids.size() - 1);
            for (int b = 0; b < (int)all_seeker_ids.size(); ++b)
                if (b != a) others.push_back(all_seeker_ids[b]);
            m_agents[a].env->set_ignored_bodies(std::move(others));
        }
    }

    // Expose envs to Python: primary (agent 0) as scene_env + full list as scene_envs.
    if (m_python_scene && !m_agents.empty())
    {
        nb::gil_scoped_acquire acquire;
        m_python_scene.attr("scene_env") = nb::cast(m_agents[0].env.get(), nb::rv_policy::reference);

        nb::list env_list;
        for (auto &s : m_agents)
            env_list.append(nb::cast(s.env.get(), nb::rv_policy::reference));
        m_python_scene.attr("scene_envs") = env_list;
    }
}

void LearnScene::onDeactivate()
{
    Application::get().use_locked_simulation_speed(true);
    Application::get().SetRelativeMouseMode(false);

    _clear_python_env_refs();

    m_entity_manager->clear();
    m_agents.clear();
    m_boundary_walls.clear(); // already freed by clear() above
    m_walls_removed = false;
    Seeker::ClearBodyIds();
}

// doesn't get called when the engine is paused
void LearnScene::update(double dt)
{
    Scene::update(dt);
    Camera &cam = Application::get().CurrentCamera();
    free_cam_motion_update_handler(dt, cam);

    // Ask Python for the next action and apply it — once per agent slot.
    {
        nb::gil_scoped_acquire acquire;

        // Static RNG for seeker spawn positions — shared across all agents.
        static std::mt19937 rng{std::random_device{}()};
        static std::uniform_real_distribution<float> dist(
            NavigationEnv::World::min * NavigationEnv::World::spawn_margin,
            NavigationEnv::World::max * NavigationEnv::World::spawn_margin);

        if (!m_step_func)
            return;

        try
        {
            for (int i = 0; i < (int)m_agents.size(); ++i)
            {
                AgentSlot &slot = m_agents[i];

                // This agent has finished its episode and is waiting for the
                // others to finish before the round resets synchronously.
                if (slot.episode_done)
                    continue;

                std::vector<float> obs;
                if (slot.has_last_obs)
                {
                    float reward    = slot.env->compute_reward();
                    bool  episode_over = slot.env->is_done();
                    // Pass true terminal (fell off map) vs truncation (timeout/step-limit)
                    // separately so the rollout buffer bootstraps truncations with V(s_last)
                    // rather than 0, avoiding systematic value underestimation.
                    bool  terminal  = slot.env->is_terminal();
                    obs = slot.env->get_observation(static_cast<float>(dt));

                    if (m_is_training && m_has_feedback && m_feedback_func)
                        m_feedback_func(slot.last_obs, slot.last_action, reward, terminal, obs, i);

                    if (episode_over)
                    {
                        // Mark done but do NOT reset yet — wait for all agents
                        // to finish so the round resets synchronously.
                        // Freeze the seeker immediately so it doesn't keep
                        // executing its last action and colliding with the
                        // still-running agents (raycasts ignore it, physics doesn't).
                        slot.env->set_pending_none();
                        {
                            JPH::BodyID seeker_body{};
                            {
                                auto [lock, reg] = m_entity_manager->get_registry();
                                if (reg.all_of<Magic::RigidBody>(slot.seeker()))
                                    seeker_body = reg.get<Magic::RigidBody>(slot.seeker()).id;
                            }
                            if (!seeker_body.IsInvalid())
                            {
                                Physics &physics = Magic::GetEngine().get_physics();
                                JPH::BodyInterface &bi = physics.physics_system.GetBodyInterface();
                                bi.SetLinearAndAngularVelocity(seeker_body, JPH::Vec3::sZero(), JPH::Vec3::sZero());
                                // Switch to Kinematic so active seekers cannot push this
                                // finished body around while waiting for the round to end.
                                bi.SetMotionType(seeker_body, JPH::EMotionType::Kinematic, JPH::EActivation::DontActivate);
                            }
                        }
                        // Hide goals for this finished agent while others are still running.
                        {
                            auto [lock, reg] = m_entity_manager->get_registry();
                            for (Entity goal : slot.goals)
                                if (!reg.all_of<Magic::Hidden>(goal))
                                    reg.emplace<Magic::Hidden>(goal);
                        }
                        slot.episode_done = true;
                        slot.has_last_obs = false;
                        continue;
                    }
                }
                else
                {
                    obs = slot.env->get_observation(static_cast<float>(dt));
                }

                nb::object action_obj = m_step_func(obs, i);
                int action = nb::cast<int>(action_obj);
                slot.env->apply_action(action);

                slot.last_obs    = std::move(obs);
                slot.last_action = action;
                slot.has_last_obs = true;
            }

            // Round complete: all agents have finished their episode.
            // Reset everyone simultaneously so they all start the next
            // episode together on the same frame.
            bool all_done = !m_agents.empty() &&
                std::all_of(m_agents.begin(), m_agents.end(),
                    [](const AgentSlot &s){ return s.episode_done; });

            if (all_done)
            {
                // Notify Python first so it can flush the PPO buffer before
                // env state is overwritten by reset().
                if (m_has_round_complete && m_round_complete_func)
                    m_round_complete_func();

                {
                    int round = m_agents.empty() ? 1 : m_agents[0].env->episode_count() + 1;
                    std::string msg = "Round " + std::to_string(round) + " complete";
                    for (int a = 0; a < (int)m_agents.size(); ++a)
                        msg += " | agent" + std::to_string(a) + "_eps=" + std::to_string(m_agents[a].env->episode_count());
                    Debug::Log(msg);
                }

                // Seed all envs from the same value so goal positions are
                // identical across agents each round.
                uint32_t round_seed = rng();
                for (auto &slot : m_agents)
                    slot.env->seed_rng(round_seed);

                for (auto &slot : m_agents)
                {
                    slot.env->reset();
                    slot.episode_done = false;
                    slot.has_last_obs = false;

                    // Unhide goals for the new episode.
                    {
                        auto [lock, reg] = m_entity_manager->get_registry();
                        for (Entity goal : slot.goals)
                            reg.remove<Magic::Hidden>(goal);
                    }

                    JPH::BodyID seeker_body{};
                    {
                        auto [lock, reg] = m_entity_manager->get_registry();
                        if (reg.all_of<Magic::RigidBody>(slot.seeker()))
                            seeker_body = reg.get<Magic::RigidBody>(slot.seeker()).id;
                    }
                    if (!seeker_body.IsInvalid())
                    {
                        Physics &physics = Magic::GetEngine().get_physics();
                        JPH::BodyInterface &bi = physics.physics_system.GetBodyInterface();
                        // Restore Dynamic motion before repositioning so the seeker
                        // responds to physics (gravity, collisions) in the new episode.
                        bi.SetMotionType(seeker_body, JPH::EMotionType::Dynamic, JPH::EActivation::DontActivate);
                        float sx = dist(rng), sz = dist(rng);
                        bi.SetPositionAndRotation(seeker_body, JPH::RVec3(sx, 0.f, sz), JPH::Quat::sIdentity(), JPH::EActivation::Activate);
                        bi.SetLinearAndAngularVelocity(seeker_body, JPH::Vec3::sZero(), JPH::Vec3::sZero());
                    }
                }

                // Curriculum: remove boundary walls once enough total experience collected.
                if (!m_walls_removed)
                {
                    int total_eps = 0;
                    for (auto &s : m_agents) total_eps += s.env->episode_count();
                    if (total_eps >= NavigationEnv::Curriculum::boundary_wall_episodes)
                        remove_boundary_walls();
                }
            }
        }
        catch (nb::python_error &e)
        {
            Debug::Log(e.what());
        }
    }
}

void LearnScene::handle_input(const SDL_Event &e)
{
    Camera &cam = Application::get().CurrentCamera();
    if (!Magic::GetEngine().is_paused() && !in_background)
        free_cam_motion_input_handler(e, cam);

    switch (e.type)
    {
    case SDL_EVENT_WINDOW_FOCUS_LOST:
    {
        in_background = true;
        Application::get().SetRelativeMouseMode(false);
        break;
    }
    case SDL_EVENT_WINDOW_FOCUS_GAINED:
    {
        in_background = false;
        Application::get().SetRelativeMouseMode(true);
        break;
    }
    }

    PauseHelper::handle_input(e);
}

void LearnScene::remove_boundary_walls()
{
    if (m_boundary_walls.empty())
        return;

    // Collect body IDs under the registry lock, then release before calling Jolt.
    std::vector<JPH::BodyID> body_ids;
    {
        auto [lock, reg] = m_entity_manager->get_registry();
        for (Entity e : m_boundary_walls)
            if (e != entt::null && reg.valid(e) && reg.all_of<Magic::RigidBody>(e))
                body_ids.push_back(reg.get<Magic::RigidBody>(e).id);
    }

    Physics &physics = Magic::GetEngine().get_physics();
    JPH::BodyInterface &bi = physics.physics_system.GetBodyInterface();
    for (const auto &id : body_ids)
    {
        bi.RemoveBody(id);
        bi.DestroyBody(id);
    }

    // Destroy ECS entities.
    {
        auto [lock, reg] = m_entity_manager->get_registry();
        for (Entity e : m_boundary_walls)
            if (e != entt::null && reg.valid(e))
                reg.destroy(e);
    }

    m_boundary_walls.clear();
    m_walls_removed = true;
    int total_eps = 0;
    for (auto &s : m_agents) total_eps += s.env->episode_count();
    Debug::Log("Curriculum: boundary walls removed after " + std::to_string(total_eps) + " total episodes");
}

