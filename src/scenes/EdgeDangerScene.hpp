#pragma once    
#include "LearnScene.hpp"

class EdgeDangerScene : public LearnScene
{
public:
    EdgeDangerScene(Magic::SceneManager &sm);
    ~EdgeDangerScene() override;

protected:
    void make_map() override;
    void implement_curriculum(int episode_count) override;
};
