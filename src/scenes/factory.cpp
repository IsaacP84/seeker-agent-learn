#include "factory.h"

#include <Magic!/app/application.hpp>
#include <Magic!/components/renderable.hpp>
#include <Magic!/core/resource_manager.hpp>
#include <Magic!/components/light.hpp>
#include <Magic!/render/render.hpp>

#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include "../navigation_env.h"
#include "../util.h"

Entity makeCube(EntityManager &em, int index)
{
    glm::vec3 random_vec3 = generateRandomVec3(-1.f, 1.f);
    glm::vec3 pos = generateRandomVec3(-10.f, 10.f);

    glm::quat rotation = glm::quat(random_vec3);
    pos.y = glm::abs(pos.y);
    Entity id = em.create("cube" + std::to_string(index), {pos, rotation, glm::vec3(1.0f)});

    // copy
    RenderObject cube = GetResourceManager().copy("cool_box");

    // {
    //     auto [lock, reg] = em.get_registry();
    //     Transform &t = reg.get<Transform>(id);
    //     t.pos = pos;
    //     // t.scale *= 0.25f;
    //     t.rotation = glm::rotate(t.rotation, glm::radians(-45.0f), random_vec3);
    // }

    // copy in order to keep all the underlying handles the same
    // there is no point to copy all of the data
    auto model_handle = GetResourceManager().copy(cube.handle());
    cube.model(model_handle);

    em.addComponent<Magic::RenderObject>(id, cube);

    Physics &physics = Application::get().engine().get_physics();
    JPH::BodyInterface &body_interface = physics.physics_system.GetBodyInterface();

    using namespace JPH;
    using namespace JPH::literals;
    assert(physics.initialized());

    // Now create a dynamic body to bounce on the floor
    // Note that this uses the shorthand version of creating and adding a body to the world
    JPH::BoxShapeSettings shape_settings(JPH::Vec3(0.5f, 0.5f, 0.5f) * 0.25f);
    JPH::Shape::ShapeResult shape_result = shape_settings.Create();
    JPH::ShapeRefC shape = shape_result.Get();
    JPH::BodyCreationSettings body_settings(shape, RVec3(pos.x, pos.y, pos.z), JPH::Quat::sIdentity(), JPH::EMotionType::Dynamic, Layers::MOVING);
    body_settings.mFriction = 1.0f;
    // body_settings.mMaxLinearVelocity = 20.0f;
    JPH::BodyID body_id = body_interface.CreateAndAddBody(body_settings, JPH::EActivation::Activate);
    em.addComponent<Magic::RigidBody>(id, Magic::RigidBody{body_id});
    return id;
}

Entity makeWall(EntityManager &em, glm::vec3 pos, glm::vec3 half_extents)
{
    // Place entity center at pos.y + half_extents.y so the bottom sits on y=pos.y.
    // Scale = half_extents * 2 so the unit cube [-0.5,0.5] fills the full wall volume.
    glm::vec3 center{pos.x, pos.y + half_extents.y, pos.z};
    Entity id = em.create({center, glm::quat{}, half_extents * 2.0f});

    ResourceManager &rm = GetResourceManager();

    // Build a bare Cube mesh with no extra model-space scale so the entity
    // Transform.scale alone controls the wall dimensions (same pattern as makeGround).
    // Load the ground texture if it hasn't been loaded yet.
    if (!rm.exists("brick_pavement_03_diff_2k"))
    {
        auto files = get_files_in_folder((ASSETS_FOLDER / "images" / "brick_pavement_03"));
        rm.load(files, Magic::Handle::Type::TEXTURE);
    }
    Magic::Cube cube_mesh{};
    cube_mesh.set(rm.get("brick_pavement_03_diff_2k", Magic::Handle::Type::TEXTURE), 1);
    MeshHandle mesh_handle = rm.add(cube_mesh);
    Model wall_model{};
    wall_model.add(mesh_handle);
    ModelHandle model_handle = rm.add(wall_model);

    RenderObject wall_ro(rm);
    wall_ro.model(model_handle);
    wall_ro.init();
    em.addComponent<Magic::RenderObject>(id, wall_ro);

    Physics &physics = Magic::GetEngine().get_physics();
    JPH::BodyInterface &body_interface = physics.physics_system.GetBodyInterface();

    using namespace JPH;
    using namespace JPH::literals;

    JPH::BoxShapeSettings shape_settings(JPH::Vec3(half_extents.x, half_extents.y, half_extents.z));
    JPH::Shape::ShapeResult shape_result = shape_settings.Create();
    JPH::ShapeRefC shape = shape_result.Get();

    // Physics body shares the same center as the visual entity.
    JPH::BodyCreationSettings settings(
        shape,
        JPH::RVec3(center.x, center.y, center.z),
        JPH::Quat::sIdentity(),
        JPH::EMotionType::Static,
        Layers::NON_MOVING);
    settings.mFriction = 1.0f;

    JPH::Body *body = physics.CreateBody(settings);
    body_interface.AddBody(body->GetID(), JPH::EActivation::DontActivate);

    return id;
}

Entity makeGround(EntityManager &em, bool do_mesh)
{
    using Type = Magic::Handle::Type;
    ResourceManager &rm = GetResourceManager();

    float x = 0.0;
    float y = 0.0;
    float z = 0.0;
    float w = NavigationEnv::World::max * 2.f;
    float l = NavigationEnv::World::max * 2.f;

    if (do_mesh)
    {

        if (!rm.exists("ground"))
        {
            {
                auto files = get_files_in_folder((ASSETS_FOLDER / "images" / "brick_pavement_03"));
                rm.load(files, Type::TEXTURE);
            }
            Magic::Rectangle mesh{};

            mesh.vertices[0].position = glm::vec3(x - w / 2, y, z - l / 2);
            mesh.vertices[1].position = glm::vec3(x - w / 2, y, z + l / 2);
            mesh.vertices[2].position = glm::vec3(x + w / 2, y, z + l / 2);
            mesh.vertices[3].position = glm::vec3(x + w / 2, y, z - l / 2);
            mesh.recalculateNormals();
            mesh.set(rm.get("brick_pavement_03_diff_2k", Type::TEXTURE), 1);
            // mesh.set(rm.get("brick_pavement_03_spec_2k", Type::TEXTURE), 2);

            MeshHandle mesh_handle = rm.add(mesh);
            Model model = Model();
            model.add(mesh_handle);

            ModelHandle model_handle = rm.add("ground", model);

            RenderObject &ground = rm.make("ground");

            ground.model(model_handle);
            ground.init();
        }
        else
        {
            LOG("Ground model already exists");
        }
    }

    Entity id = em.create();

    em.addComponent<Magic::RenderObject>(id, rm.copy("ground"));

    Physics &physics = Magic::GetEngine().get_physics();

    JPH::BodyInterface &body_interface = physics.physics_system.GetBodyInterface();

    // body_interface = &physics_system->GetBodyInterface();
    using namespace JPH;
    using namespace JPH::literals;
    assert(physics.initialized());

    // 2. Create the shape (Shapes are reference counted and thread-safe)
    JPH::BoxShapeSettings shape_settings(JPH::Vec3(w / 2, 0.5f, l / 2));
    JPH::Shape::ShapeResult shape_result = shape_settings.Create();
    JPH::ShapeRefC shape = shape_result.Get();

    JPH::BodyCreationSettings settings(
        shape,                         // Size
        JPH::RVec3(0.0f, -0.5f, 0.0f), // Position
        JPH::Quat::sIdentity(),
        JPH::EMotionType::Static,
        Layers::NON_MOVING);
    settings.mFriction = 1.0f;

    JPH::Body *floor = physics.CreateBody(settings);
    body_interface.AddBody(floor->GetID(), JPH::EActivation::DontActivate);

    return id;
}

Entity makeLight(EntityManager &em, const std::string name, Vec3 pos, const PointLight &lc)
{
    ResourceManager &rm = GetResourceManager();
    Entity id = em.create(name, {pos});

    // copy
    RenderObject light = rm.copy("light");

    // copy in order to keep all the underlying handles the same
    // there is no point to copy all of the data
    light.model(rm.copy(light.handle()));
    // Model &model = light.model();
    em.addComponent<Magic::RenderObject>(id, light);
    em.addComponent<Magic::LightComponent>(id, lc);
    return id;
}

Entity makeLight(EntityManager &em, const std::string name, Vec3 pos, const DirectionLight &lc)
{
    ResourceManager &rm = GetResourceManager();
    Entity id = em.create(name, {pos});

    // copy
    RenderObject light = rm.copy("light");

    // copy in order to keep all the underlying handles the same
    // there is no point to copy all of the data
    light.model(rm.copy(light.handle()));
    // Model &model = light.model();
    em.addComponent<Magic::RenderObject>(id, light);
    em.addComponent<Magic::LightComponent>(id, lc);
    return id;
}

Entity makeLight(EntityManager &em, const std::string name, Vec3 pos, const SpotLight &lc)
{
    ResourceManager &rm = GetResourceManager();
    Entity id = em.create(name, {pos});

    // copy
    RenderObject light = rm.copy("light");

    // copy in order to keep all the underlying handles the same
    // there is no point to copy all of the data
    light.model(rm.copy(light.handle()));
    // Model &model = light.model();
    em.addComponent<Magic::RenderObject>(id, light);
    em.addComponent<Magic::LightComponent>(id, lc);
    return id;
}
