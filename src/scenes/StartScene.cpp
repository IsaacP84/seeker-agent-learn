#include "StartScene.hpp"
#include "LoadingScene.hpp"

#include <Magic!/app/application.hpp>
#include <Magic!/debug/debug.hpp>
#include <Magic!/globals.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <fstream>
#include <future>
#include <string>

namespace nb = nanobind;
using namespace Magic;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::string read_target_scene()
{
    // Read "scene" key from assets/learning/config.json.
    // Falls back to "learn" if the key is absent or the file is unreadable.
    std::filesystem::path cfg = ASSETS_FOLDER / "learning" / "config.json";
    std::ifstream f(cfg);
    if (!f.is_open())
    {
        Debug::Log("StartScene: could not open config.json, defaulting to 'learn'");
        return "learn";
    }

    std::string contents((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());

    // Look for "scene": "value" — simple enough not to need a JSON library.
    auto key_pos = contents.find("\"scene\"");
    if (key_pos == std::string::npos)
        return "learn";

    auto colon = contents.find(':', key_pos);
    if (colon == std::string::npos)
        return "learn";

    auto q1 = contents.find('"', colon + 1);
    if (q1 == std::string::npos)
        return "learn";

    auto q2 = contents.find('"', q1 + 1);
    if (q2 == std::string::npos)
        return "learn";

    return contents.substr(q1 + 1, q2 - q1 - 1);
}

// ---------------------------------------------------------------------------

StartScene::StartScene(Magic::SceneManager &sm)
    : Magic::Scene(sm)
{}

StartScene::~StartScene() = default;

void StartScene::onCreate()
{
    LOG("START SCENE: booting Python...");

    m_target_scene = read_target_scene();
    Debug::Log("StartScene: target scene is '" + m_target_scene + "'");

    m_python_future = std::async(std::launch::async, []()
    {
        nb::gil_scoped_acquire acquire;
        nb::module_::import_("scenes.learn_scene");
    }).share();

    m_sm.add_scene<LoadingScene>("loading", m_sm, m_python_future, m_target_scene);
    m_loading_scene_registered = true;
    m_sm.switch_to("loading");
}

void StartScene::onDestroy()
{
    if (m_loading_scene_registered)
    {
        m_sm.remove_scene("loading");
        m_loading_scene_registered = false;
    }
}

void StartScene::onActivate()  {}
void StartScene::onDeactivate() {}
void StartScene::update(double) {}
