#include "LoadingScene.hpp"

#include <Magic!/app/application.hpp>
#include <Magic!/core/scene_manager.hpp>
#include <Magic!/debug/debug.hpp>
#include <Magic!/globals.hpp>

#include <nanobind/nanobind.h>

#include <chrono>
#include <string>

namespace nb = nanobind;
using namespace Magic;

LoadingScene::LoadingScene(SceneManager &sm, std::shared_future<void> future, std::string target)
    : Scene(sm), m_load_future(std::move(future)), m_target_scene(std::move(target))
{}

LoadingScene::~LoadingScene() = default;

void LoadingScene::onCreate() {}

void LoadingScene::onDestroy() {}

void LoadingScene::onActivate()
{
    m_elapsed         = 0.0;
    m_last_dot_second = -1;
    LOG("LoadingScene: waiting for Python import...");
}

void LoadingScene::onDeactivate() {}

void LoadingScene::update(double dt)
{
    m_elapsed += dt;

    int current_second = static_cast<int>(m_elapsed);
    if (current_second != m_last_dot_second)
    {
        m_last_dot_second = current_second;
        LOG("Loading" + std::string(current_second % 3 + 1, '.'));
    }

    if (!m_load_future.valid())
        return;

    // Briefly release the GIL so the background import thread can make progress.
    std::future_status status;
    {
        nb::gil_scoped_release release;
        status = m_load_future.wait_for(std::chrono::seconds(0));
    }

    if (status != std::future_status::ready)
        return;

    try
    {
        m_load_future.get();
    }
    catch (const std::exception &e)
    {
        Debug::Log(std::string("LoadingScene: Python import failed: ") + e.what());
        // Prevent repeatedly rethrowing/logging the same exception every frame.
        m_load_future = {};
        return;
    }

    LOG("LoadingScene: done, switching to " + m_target_scene);
    m_load_future = {};  // prevent any accidental re-trigger
    // switch_to() calls onActivate() which runs factory/GL code;
    // defer it to the main (render) thread where the GL context is current.
    std::string target = m_target_scene;
    Magic::Application::get().run_on_main_thread([&sm = m_sm, target]()
    {
        sm.switch_to(target);
    });
}


