#include "TestScene.hpp"

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

namespace nb = nanobind;
using namespace Magic;

static bool test_in_background = false;

TestScene::TestScene(Magic::SceneManager &sm)
    : Magic::Scene(sm), m_env(std::make_unique<NavigationEnv>())
{
}

TestScene::~TestScene() = default;

void TestScene::onCreate()
{
    LOG("CREATING TEST SCENE");
    Seeker::LoadModel();
}

void TestScene::onDestroy()
{
}

void TestScene::onActivate()
{
    if (!m_step_func)
    {
        nb::gil_scoped_acquire acquire;
        try
        {
            m_python_scene = nb::module_::import_("scenes.learn_scene");
            m_step_func    = m_python_scene.attr("step");
            // Force inference regardless of config.json — TestScene never calls feedback().
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
    makeWall(em, {-10.f,  0.f, -10.f}, {0.25f, 3.f, 8.f});   // far SW vertical
    makeWall(em, { 10.f,  0.f,  10.f}, {0.25f, 3.f, 8.f});   // far NE vertical
    // Horizontal blockers near centre
    makeWall(em, {  0.f,  0.f,  -4.f}, {6.f,   3.f, 0.25f}); // centre-south horizontal
    makeWall(em, {  0.f,  0.f,   4.f}, {6.f,   3.f, 0.25f}); // centre-north horizontal
    // Isolated pillars
    makeWall(em, { -6.f,  0.f,   0.f}, {0.75f, 3.f, 0.75f}); // left pillar
    makeWall(em, {  6.f,  0.f,   0.f}, {0.75f, 3.f, 0.75f}); // right pillar
    // L-shaped corner blocker
    makeWall(em, { 14.f,  0.f, -14.f}, {4.f,   3.f, 0.25f}); // L horizontal arm
    makeWall(em, { 14.f,  0.f, -10.f}, {0.25f, 3.f, 4.f});   // L vertical arm

    // No boundary walls — this is inference; agent should already know the edges.

    // ---- Goal entities ----
    m_goals.clear();
    for (int i = 0; i < NavigationEnv::Sizes::num_goals; ++i)
    {
        Entity goal = em.create("Goal" + std::to_string(i), {Vec3{3.f * (i + 1), 0.5f, 3.f}});
        RenderObject goal_ro = rm.copy("cool_box");
        goal_ro.model(rm.copy(goal_ro.handle()));
        em.addComponent<Magic::RenderObject>(goal, goal_ro);
        m_goals.push_back(goal);
    }

    m_seeker = Seeker(*m_entity_manager);
    m_env->bind(*m_entity_manager, m_seeker(), m_goals);
    m_env->disable_goal_timeout();

    if (m_python_scene)
    {
        nb::gil_scoped_acquire acquire;
        m_python_scene.attr("scene_env") = nb::cast(m_env.get(), nb::rv_policy::reference);
    }

    {
        auto [lock, reg] = m_entity_manager->get_registry();
        auto &data = reg.get<Seeker::Data>(m_seeker());
        NavigationEnv *env_ptr = m_env.get();
        data.on_move_override = [env_ptr](EntityManager &, Entity, double) -> Seeker::Action
        {
            return env_ptr->pending_action();
        };
    }
}

void TestScene::onDeactivate()
{
    Application::get().use_locked_simulation_speed(true);
    Application::get().SetRelativeMouseMode(false);
    m_entity_manager->clear();
    m_has_last_obs = false;
    m_last_obs.clear();
    m_last_action = -1;
}

void TestScene::update(double dt)
{
    Scene::update(dt);
    Camera &cam = Application::get().CurrentCamera();
    free_cam_motion_update_handler(dt, cam);

    {
        nb::gil_scoped_acquire acquire;
        try
        {
            std::vector<float> obs;
            if (m_has_last_obs)
            {
                bool done = m_env->is_done();
                obs = m_env->get_observation(static_cast<float>(dt));

                if (done)
                {
                    m_env->reset();

                    static std::mt19937 rng{std::random_device{}()};
                    static std::uniform_real_distribution<float> dist(
                        NavigationEnv::World::min * NavigationEnv::World::spawn_margin,
                        NavigationEnv::World::max * NavigationEnv::World::spawn_margin);

                    float sx = dist(rng), sz = dist(rng);

                    JPH::BodyID seeker_body{};
                    {
                        auto [lock, reg] = m_entity_manager->get_registry();
                        if (reg.all_of<Magic::RigidBody>(m_seeker()))
                            seeker_body = reg.get<Magic::RigidBody>(m_seeker()).id;
                    }
                    if (!seeker_body.IsInvalid())
                    {
                        Physics &physics = Magic::GetEngine().get_physics();
                        JPH::BodyInterface &bi = physics.physics_system.GetBodyInterface();
                        bi.SetPositionAndRotation(seeker_body, JPH::RVec3(sx, 0.f, sz), JPH::Quat::sIdentity(), JPH::EActivation::Activate);
                        bi.SetLinearAndAngularVelocity(seeker_body, JPH::Vec3::sZero(), JPH::Vec3::sZero());
                    }

                    m_has_last_obs = false;
                    return;
                }
            }
            else
            {
                obs = m_env->get_observation(static_cast<float>(dt));
            }

            if (!m_step_func)
                throw std::runtime_error("Python step function not loaded");

            // compute_reward() handles goal collection and relocation even in
            // inference — without it goals are never detected as reached.
            m_env->compute_reward();

            nb::object action_obj = m_step_func(obs);
            int action = nb::cast<int>(action_obj);
            m_env->apply_action(action);

            m_last_obs    = std::move(obs);
            m_last_action = action;
            m_has_last_obs = true;
        }
        catch (nb::python_error &e)
        {
            Debug::Log(e.what());
        }
    }
}

void TestScene::handle_input(const SDL_Event &e)
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
