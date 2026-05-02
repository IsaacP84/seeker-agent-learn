#pragma once
#include <Magic!/core/entity_manager.hpp>
#include <Magic!/render/light.hpp>
#include <Magic!/core/math.hpp>

using namespace Magic;

Entity makeCube(EntityManager &em, int index);
Entity makeGround(EntityManager &em, bool do_mesh = true);
Entity makeWall(EntityManager &em, glm::vec3 pos, glm::vec3 half_extents);

Entity makeLight(EntityManager &, const std::string name, Vec3 pos, const PointLight &);
Entity makeLight(EntityManager &, const std::string name, Vec3 pos, const DirectionLight &);
Entity makeLight(EntityManager &, const std::string name, Vec3 pos, const SpotLight &);
