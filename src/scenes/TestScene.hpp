#pragma once
#include "../entities/seeker.h"
#include "../util.h"

#include <Magic!/core/scene.hpp>
#include <Magic!/render/object.hpp>
#include <Magic!/core/entity_manager.hpp>

#include <nanobind/nanobind.h>

namespace nb = nanobind;

#include <memory>
#include <vector>

class NavigationEnv;

// Inference-only scene with a different wall layout for testing trained models.
// Never calls feedback() — the agent always runs greedy (IS_TRAINING=false).
class TestScene : public Magic::Scene, PauseHelper
{
public:
    TestScene(Magic::SceneManager &sm);
    ~TestScene();

    void onCreate() override;
    void onDestroy() override;
    void onActivate() override;
    void onDeactivate() override;
    void update(double dt) override;
    void handle_input(const SDL_Event &) override;

private:
    Seeker                     m_seeker;
    std::vector<Magic::Entity> m_goals;
    std::unique_ptr<NavigationEnv> m_env;
    std::vector<float>         m_last_obs;
    int                        m_last_action  = -1;
    bool                       m_has_last_obs = false;

    nb::object m_python_scene;
    nb::object m_step_func;
};
