#pragma once
#include <SDL3/SDL.h>
#include <string>
#include <Magic!/core/entity.hpp>
#include <Magic!/core/math.hpp>
#include <Magic!/core/camera.hpp>

#include <Jolt/Jolt.h>
#include <Jolt/Physics/Body/BodyID.h>

Magic::Entity find_entity_from_hitbox(const entt::registry& reg, JPH::BodyID id);
Magic::Vec3 hexToRgbStoi(std::string hex);
std::vector<std::string> get_files_in_folder(const fs::path &directory_path);

void free_cam_motion_input_handler(const SDL_Event &e, Magic::Camera &cam);
void free_cam_motion_update_handler(double dt, Magic::Camera &cam);

struct PauseHelper
{
    virtual void handle_input(const SDL_Event &);
    void pause();
    void unpause();
};


