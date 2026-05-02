#include "LearnScene.hpp"
#include "LoadingScene.hpp"

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

#include <future>
#include <random>

namespace nb = nanobind;
using namespace Magic;

bool in_background = false;

LearnScene::LearnScene(Magic::SceneManager &sm)
    : Magic::Scene(sm), m_env(std::make_unique<NavigationEnv>())
{
}

LearnScene::~LearnScene() = default;

void LearnScene::onCreate()
{
    LOG("CREATING LEARN SCENE");

    // Start Python import asynchronously. LoadingScene will poll this future
    // and switch back here once it completes.
    m_python_future = std::async(std::launch::async, []()
    {
        nb::gil_scoped_acquire acquire;
        nb::module_::import_("scenes.learn_scene");
    }).share();

    Seeker::LoadModel();

    m_sm.add_scene<LoadingScene>("loading", m_sm, m_python_future, "learn");
    m_loading_scene_registered = true;
    m_sm.switch_to("loading");
}

void LearnScene::onDestroy()
{
    // Safe to remove here: onDestroy is never called from within LoadingScene::update(),
    // so there is no dangling-this risk.
    if (m_loading_scene_registered)
    {
        m_sm.remove_scene("loading");
        m_loading_scene_registered = false;
    }
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
            if (nb::hasattr(m_python_scene, "IS_TRAINING"))
                m_is_training = nb::cast<bool>(m_python_scene.attr("IS_TRAINING"));
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

    // Bind env to the freshly constructed seeker.
    m_env->bind(*m_entity_manager, m_seeker(), m_goals);

    // Expose the env instance to the Python scene so it can save/restore env data.
    if (m_python_scene)
    {
        nb::gil_scoped_acquire acquire;
        m_python_scene.attr("scene_env") = nb::cast(m_env.get(), nb::rv_policy::reference);
    }

    // Wire the seeker callback to the env.
    // Scope the registry lock tightly — holding it past this block causes a deadlock
    // when anything else in onActivate() (or the callback itself) tries to acquire it.
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

void LearnScene::onDeactivate()
{
    Application::get().use_locked_simulation_speed(true);
    Application::get().SetRelativeMouseMode(false);
    m_entity_manager->clear();
    m_has_last_obs = false;
    m_last_obs.clear();
    m_last_action = -1;
    m_boundary_walls.clear(); // already freed by clear() above
    m_walls_removed = false;
}

// doesn't get called when the engine is paused
void LearnScene::update(double dt)
{
    Scene::update(dt);
    Camera &cam = Application::get().CurrentCamera();
    free_cam_motion_update_handler(dt, cam);

    // Ask Python for the next action and apply it.
    {
        nb::gil_scoped_acquire acquire;
        try
        {
            std::vector<float> obs;
            bool prev_done = false;
            if (m_has_last_obs)
            {
                float reward = m_env->compute_reward();
                bool done = m_env->is_done();
                obs = m_env->get_observation(static_cast<float>(dt));
                if (m_is_training && m_has_feedback && m_feedback_func)
                {
                    m_feedback_func(m_last_obs, m_last_action, reward, done, obs);
                }
                prev_done = done;

                if (prev_done)
                {
                    m_env->reset();

                    // Curriculum: remove boundary walls once the agent has enough experience.
                    if (!m_walls_removed && m_env->episode_count() >= NavigationEnv::Curriculum::boundary_wall_episodes)
                        remove_boundary_walls();

                    // Random spawn position for the seeker.
                    // Goals are randomized internally by NavigationEnv::reset().
                    static std::mt19937 rng{std::random_device{}()};
                    static std::uniform_real_distribution<float> dist(
                        NavigationEnv::World::min * NavigationEnv::World::spawn_margin,
                        NavigationEnv::World::max * NavigationEnv::World::spawn_margin);

                    float sx = dist(rng), sz = dist(rng);

                    // Teleport the seeker (Jolt body) and zero its velocity.
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

                    // Defer the post-reset observation to the next frame so the
                    // teleport cost isn't stacked on top of feedback/optimize.
                    // The (1-done) term in the Bellman update means the stale
                    // next_state on the terminal transition doesn't affect training.
                    m_has_last_obs = false;
                    return;
                }
            }
            else
            {
                obs = m_env->get_observation(static_cast<float>(dt));
            }

            if (!m_step_func)
            {
                throw std::runtime_error("Python step function not loaded");
            }
            nb::object action_obj = m_step_func(obs);
            int action = nb::cast<int>(action_obj);
            m_env->apply_action(action);

            // Always record the obs/action we just acted on, including the
            // fresh post-reset obs so the next frame can call feedback correctly.
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
    Debug::Log("Curriculum: boundary walls removed after episode " + std::to_string(m_env->episode_count()));
}

