#include "EdgeDangerScene.hpp"

#include "factory.h"

#include <Magic!/app/application.hpp>
#include <Magic!/debug/debug.hpp>
#include <Magic!/globals.hpp>

#ifdef TRACY_ENABLE
#include <tracy/Tracy.hpp>
#else
#define ZoneScoped
#define FrameMark
#define ZoneScopedN(...)
#endif

using namespace Magic;

EdgeDangerScene::EdgeDangerScene(Magic::SceneManager &sm)
    : LearnScene(sm)
{
}

EdgeDangerScene::~EdgeDangerScene() = default;

void EdgeDangerScene::make_map()
{
    EntityManager &em = *m_entity_manager;

    // Base obstacles for navigation, without curriculum boundary walls.
    makeWall(em, {0.f, 0.f, 0.f}, {1.5f, 3.f, 0.25f});
    makeWall(em, {-8.f, 0.f, 6.f}, {0.25f, 3.f, 6.f});
    makeWall(em, {8.f, 0.f, -6.f}, {0.25f, 3.f, 6.f});
    makeWall(em, {-14.f, 0.f, -8.f}, {0.25f, 3.f, 8.f});
    makeWall(em, {14.f, 0.f, 8.f}, {0.25f, 3.f, 8.f});

    // Edge danger obstacles near the perimeter to make the boundary riskier.
    makeWall(em, {-16.5f, 0.f, 0.f}, {0.25f, 3.f, 5.f});
    makeWall(em, {16.5f, 0.f, 0.f}, {0.25f, 3.f, 5.f});
    makeWall(em, {0.f, 0.f, -16.5f}, {5.f, 3.f, 0.25f});
    makeWall(em, {0.f, 0.f, 16.5f}, {5.f, 3.f, 0.25f});

    // Corner pillars to encourage navigation near the edges.
    makeWall(em, {-18.f, 0.f, -18.f}, {0.25f, 3.f, 0.25f});
    makeWall(em, {18.f, 0.f, 18.f}, {0.25f, 3.f, 0.25f});
}

void EdgeDangerScene::implement_curriculum(int episode_count)
{
    // In this scene the curriculum is just to remove the edge danger walls after a certain amount of experience collected.
    if (episode_count > 0)
    {
    }
}