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
class ShowCase : public Magic::Scene, PauseHelper
{
public:
    ShowCase(Magic::SceneManager &sm);
    ~ShowCase();

    void onCreate() override;
    void onDestroy() override;
    void onActivate() override;
    void onDeactivate() override;
    void update(double dt) override;
    void handle_input(const SDL_Event &) override;

private:
    nb::object m_python_scene;
    nb::object m_step_func;

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

    
    void _clear_python_env_refs();
};
