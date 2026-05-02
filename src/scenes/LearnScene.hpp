#pragma once
#include "../entities/seeker.h"
#include "../util.h"

#include <Magic!/core/scene.hpp>
#include <Magic!/render/object.hpp>
#include <Magic!/core/entity_manager.hpp>

#include <nanobind/nanobind.h>

namespace nb = nanobind;

#include <memory>
#include <future>
#include <vector>

class NavigationEnv;  // forward declaration — full definition in navigation_env.h

class LearnScene : public Magic::Scene, PauseHelper
{
public:
    LearnScene(Magic::SceneManager &sm);
    ~LearnScene();

    void onCreate() override;
    void onDestroy() override;
    void onActivate() override;
    void onDeactivate() override;
    void update(double dt) override;
    void handle_input(const SDL_Event &) override;

private:
    Seeker           m_seeker;
    std::vector<Magic::Entity> m_goals;
    std::vector<Magic::Entity> m_boundary_walls;
    bool m_walls_removed = false;
    // Use unique_ptr to avoid requiring the full definition in the header.
    std::unique_ptr<NavigationEnv> m_env;
    std::vector<float> m_last_obs;
    int m_last_action = -1;
    bool m_has_last_obs = false;

    nb::object m_python_scene;
    nb::object m_step_func;
    nb::object m_feedback_func;
    bool m_has_feedback = false;
    bool m_loading_scene_registered = false;
    bool m_is_training = true;
    std::shared_future<void> m_python_future;

    void remove_boundary_walls();
};

