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
    struct AgentSlot {
        Seeker                         seeker;
        std::vector<Magic::Entity>     goals;
        std::unique_ptr<NavigationEnv> env;
        std::vector<float>             last_obs;
        int                            last_action  = -1;
        bool                           has_last_obs = false;
        bool                           episode_done = false; // waiting for other agents to finish the round
    };

    int                        m_num_agents = 1;
    std::vector<AgentSlot>     m_agents;
    std::vector<Magic::Entity> m_boundary_walls;
    bool                       m_walls_removed = false;

    nb::object m_python_scene;
    nb::object m_step_func;
    nb::object m_feedback_func;
    nb::object m_round_complete_func;
    bool m_has_feedback       = false;
    bool m_has_round_complete = false;
    bool m_is_training = true;

    void remove_boundary_walls();
};

