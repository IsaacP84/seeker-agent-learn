#pragma once
#include <Magic!/core/scene.hpp>
#include <future>
#include <string>

// Entry-point scene. Boots Python asynchronously, reads config.json to
// determine the target scene ("learn" or "test"), then switches to it
// via LoadingScene once the import completes.
class StartScene : public Magic::Scene
{
public:
    StartScene(Magic::SceneManager &sm);
    ~StartScene();

    void onCreate()    override;
    void onDestroy()   override;
    void onActivate()  override;
    void onDeactivate() override;
    void update(double dt) override;

private:
    std::shared_future<void> m_python_future;
    std::string              m_target_scene;
    bool                     m_loading_scene_registered = false;
};
