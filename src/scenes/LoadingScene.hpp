#pragma once

#include <Magic!/core/scene.hpp>

#include <future>
#include <string>

class LoadingScene : public Magic::Scene
{
public:
    LoadingScene(Magic::SceneManager &sm, std::shared_future<void> future, std::string target);
    ~LoadingScene();

    void onCreate() override;
    void onDestroy() override;
    void onActivate() override;
    void onDeactivate() override;
    void update(double dt) override;

private:
    std::shared_future<void> m_load_future;
    std::string              m_target_scene;
    double                   m_elapsed         = 0.0;
    int                      m_last_dot_second = -1;
};

