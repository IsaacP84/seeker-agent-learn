#include "ShowCase.hpp"

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

#ifdef TRACY_ENABLE
#include <tracy/Tracy.hpp>
#else
#define ZoneScoped
#define FrameMark
#define ZoneScopedN(...)
#endif

namespace nb = nanobind;
using namespace Magic;

static bool test_in_background = false;

ShowCase::ShowCase(Magic::SceneManager &sm)
    : Magic::Scene(sm)
{
}

ShowCase::~ShowCase() = default;

void ShowCase::onCreate()
{
    LOG("CREATING TEST SCENE");
    Seeker::LoadModel();
}

void ShowCase::onDestroy()
{
}

void ShowCase::onActivate()
{
    if (!m_step_func)
    {
        nb::gil_scoped_acquire acquire;
        try
        {
            m_python_scene = nb::module_::import_("scenes.learn_scene");
            m_step_func = m_python_scene.attr("step");
            if (nb::hasattr(m_python_scene, "NUM_AGENTS"))
                m_num_agents = nb::cast<int>(m_python_scene.attr("NUM_AGENTS"));
            // Force inference regardless of config.json — ShowCase never calls feedback().
            m_python_scene.attr("IS_TRAINING") = nb::bool_(false);
            // Always inference — lock to real-time speed.
            Application::get().use_locked_simulation_speed(true);
            LOG("TEST SCENE READY");
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
        Camera &cam = GetApplication().CurrentCamera();
        cam.pos = {0, 2, 0};
    }

    makeGround(em);
    makeLight(em, "sun", {0, 10, 0}, Magic::DirectionLight{hexToRgbStoi("#ffffff"), {0.5, -1, 0.5}});

    // ---- Test layout — different from training to probe generalisation ----
    // Diagonal corridor pair across the map.
    makeWall(em, {-10.f, 0.f, -10.f}, {0.25f, 3.f, 8.f}); // far SW vertical
    makeWall(em, {10.f, 0.f, 10.f}, {0.25f, 3.f, 8.f});   // far NE vertical
    // Horizontal blockers near centre
    makeWall(em, {0.f, 0.f, -4.f}, {6.f, 3.f, 0.25f}); // centre-south horizontal
    makeWall(em, {0.f, 0.f, 4.f}, {6.f, 3.f, 0.25f});  // centre-north horizontal
    // Isolated pillars
    makeWall(em, {-6.f, 0.f, 0.f}, {0.75f, 3.f, 0.75f}); // left pillar
    makeWall(em, {6.f, 0.f, 0.f}, {0.75f, 3.f, 0.75f});  // right pillar
    // L-shaped corner blocker
    makeWall(em, {14.f, 0.f, -14.f}, {4.f, 3.f, 0.25f}); // L horizontal arm
    makeWall(em, {14.f, 0.f, -10.f}, {0.25f, 3.f, 4.f}); // L vertical arm

    // No boundary walls — this is inference; agent should already know the edges.

    m_agents.clear();
    m_agents.reserve(m_num_agents);
    for (int a = 0; a < m_num_agents; ++a)
    {
        AgentSlot &slot = m_agents.emplace_back();
        slot.env = std::make_unique<NavigationEnv>();

        for (int g = 0; g < NavigationEnv::Sizes::num_goals; ++g)
        {
            std::string name = "Agent" + std::to_string(a) + "Goal" + std::to_string(g);
            Entity goal = em.create(name, {Vec3{3.f * (g + 1), 0.5f, 3.f}});
            RenderObject goal_ro = rm.copy("cool_box");
            goal_ro.model(rm.copy(goal_ro.handle()));
            em.addComponent<Magic::RenderObject>(goal, goal_ro);
            slot.goals.push_back(goal);
        }

        slot.seeker = Seeker(*m_entity_manager);
        slot.env->bind(*m_entity_manager, slot.seeker(), slot.goals);
        slot.env->set_agent_id(a);
        slot.env->disable_goal_timeout();

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
                if (b != a)
                    others.push_back(all_seeker_ids[b]);
            m_agents[a].env->set_ignored_bodies(std::move(others));
        }
    }

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

void ShowCase::onDeactivate()
{
    Application::get().use_locked_simulation_speed(true);
    Application::get().SetRelativeMouseMode(false);
    _clear_python_env_refs();

    m_entity_manager->clear();
    m_agents.clear();
    m_python_scene = nb::object();
    m_step_func = nb::object();
    Seeker::ClearBodyIds();
}

void ShowCase::update(double dt)
{
    FrameMark;
    ZoneScoped;
    Scene::update(dt);
    Camera &cam = Application::get().CurrentCamera();
    free_cam_motion_update_handler(dt, cam);

    {
        nb::gil_scoped_acquire acquire;
        try
        {
            if (!m_step_func)
                throw std::runtime_error("Python step function not loaded");

            for (int i = 0; i < (int)m_agents.size(); ++i)
            {
                ZoneScopedN("Agent step");
                AgentSlot &slot = m_agents[i];
                std::vector<float> obs;

                if (slot.has_last_obs)
                {
                    bool done = slot.env->is_done();
                    slot.env->compute_reward();
                    obs = slot.env->get_observation(static_cast<float>(dt));

                    if (done)
                    {
                        slot.env->reset();

                        static std::mt19937 rng{std::random_device{}()};
                        static std::uniform_real_distribution<float> dist(
                            NavigationEnv::World::min * NavigationEnv::World::spawn_margin,
                            NavigationEnv::World::max * NavigationEnv::World::spawn_margin);

                        float sx = dist(rng), sz = dist(rng);

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
                            bi.SetPositionAndRotation(seeker_body, JPH::RVec3(sx, 0.f, sz), JPH::Quat::sIdentity(), JPH::EActivation::Activate);
                            bi.SetLinearAndAngularVelocity(seeker_body, JPH::Vec3::sZero(), JPH::Vec3::sZero());
                        }

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

                slot.last_obs = std::move(obs);
                slot.last_action = action;
                slot.has_last_obs = true;
            }
        }
        catch (nb::python_error &e)
        {
            Debug::Log(e.what());
        }
    }
}

void ShowCase::handle_input(const SDL_Event &e)
{
    Camera &cam = Application::get().CurrentCamera();
    if (!Magic::GetEngine().is_paused() && !test_in_background)
        free_cam_motion_input_handler(e, cam);

    switch (e.type)
    {
    case SDL_EVENT_WINDOW_FOCUS_LOST:
        test_in_background = true;
        Application::get().SetRelativeMouseMode(false);
        break;
    case SDL_EVENT_WINDOW_FOCUS_GAINED:
        test_in_background = false;
        Application::get().SetRelativeMouseMode(true);
        break;
    }

    PauseHelper::handle_input(e);
}

void ShowCase::_clear_python_env_refs()
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