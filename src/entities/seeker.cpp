#include "seeker.h"
#include <Jolt/Physics/Character/CharacterVirtual.h>
#include <Magic!/app/application.hpp>
#include <Magic!/components/renderable.hpp>
#include <Magic!/components/physics.hpp>
#include <Magic!/components/input.hpp>
#include <Magic!/components/events.hpp>

#include <Magic!/debug/debug.hpp>
#include <Magic!/debug/log.hpp>

#include <Magic!/core/math.hpp>

#include "../components/components.h"

#include <Jolt/Physics/Collision/RayCast.h>
#include <Jolt/Physics/Collision/CastResult.h>

using namespace Magic;

#include <queue>

// TODO: refactor this into a more generic character controller component that can be used by both player and npcs
// also add crouching and sprinting and wall jumping and stuff like that

// it would be cool if in free cam you can select a new entity to conrol by looking at it and pressing a button, maybe also add a debug menu that shows all entities and lets you select one to control
// Seeker::Action and Seeker::Data are now defined in seeker.h

class IgnoreSelfFilter : public JPH::BodyFilter
{
public:
    IgnoreSelfFilter(JPH::BodyID inSelfBody) : mSelfBody(inSelfBody) {}

    virtual bool ShouldCollide(const JPH::BodyID &inBodyID) const override
    {
        // Ignore the body if it is the one casting the ray
        return inBodyID != mSelfBody;
    }

private:
    JPH::BodyID mSelfBody;
};
// make a seeker ai controller component

struct SeekerUpdateHandler
{
    const Magic::Entity m_entity;
    std::function<Seeker::Action(EntityManager &, Entity, double)> on_move = nullptr;
    std::function<void(EntityManager &, Entity, double)> on_jump = nullptr;
    std::function<void(EntityManager &, Entity)> on_land = nullptr;
    std::function<void(EntityManager &, Entity, double)> on_shoot = nullptr;
    SeekerUpdateHandler(Magic::Entity e) : m_entity(e)
    {
    }
    void update(EntityManager &em, double dt)
    {
        auto [lock, reg] = em.get_registry();
        Physics &physics = Application::get().engine().get_physics();
        JPH::BodyInterface &body_interface = physics.physics_system.GetBodyInterface();

        Magic::RigidBody &rb = reg.get<Magic::RigidBody>(m_entity);

        auto &data = reg.get<Seeker::Data>(m_entity);

        bool in_air_prev = data.in_air;

        {
            const Transform &t = reg.get<Transform>(m_entity);
            JPH::Vec3 start(t.pos.x, t.pos.y, t.pos.z);
            JPH::Vec3 direction(0, -0.5, 0);
            JPH::RRayCast ray{start, direction};

            JPH::RayCastResult hit;
            JPH::RayCastSettings settings; // Can configure backface culling, etc.

            bool hadHit = physics.physics_system.GetNarrowPhaseQuery().CastRay(ray, hit, {}, {}, IgnoreSelfFilter(rb.id));

            if (hadHit)
            {
                data.in_air = false;
            }
            else
            {
                data.in_air = true;
            }
        }

        if (!data.in_air)
        {
            data.is_jumping = false;
            if (in_air_prev)
            {
                lock.unlock();
                on_land(em, m_entity);
                lock.lock();

            }
        }

        {
            bool is_sprinting = false;

            if (data.jumped)
            {
                if (!data.is_jumping && !data.in_air)
                {
                    data.is_jumping = true;
                    lock.unlock();
                    on_jump(em, m_entity, dt);
                    lock.lock();
                }
                data.jumped = false;
            }

            if (data.shoot)
            {
                data.shoot = false;
                lock.unlock();
                on_shoot(em, m_entity, dt);
                lock.lock();
            }

            Vec3 vec = {0, 0, 0};
            if (!data.in_air || true)
            {
                lock.unlock();
                // get the desired movement direction
                // Allow external systems (e.g. NavigationEnv) to override on_move.
                Seeker::Action action = data.on_move_override
                                            ? data.on_move_override(em, m_entity, dt)
                                            : on_move(em, m_entity, dt);
                lock.lock();
                Vec3 forward = {
                    sin(glm::radians(0 + data.looking_angle)),
                    0,
                    cos(glm::radians(0 + data.looking_angle))};

                switch (action)
                {
                case Seeker::MOVE_FORWARD:
                    vec += forward;
                    break;
                case Seeker::MOVE_BACKWARD:
                    vec -= forward;
                    break;
                case Seeker::STRAFE_LEFT:
                    vec -= glm::normalize(glm::cross(forward, {0, 1, 0}));
                    break;
                case Seeker::STRAFE_RIGHT:
                    vec += glm::normalize(glm::cross(forward, {0, 1, 0}));
                    break;
                case Seeker::TURN_LEFT:
                    data.looking_angle += data.turn_speed * (float)dt;
                    break;
                case Seeker::TURN_RIGHT:
                    data.looking_angle -= data.turn_speed * (float)dt;
                    break;
                case Seeker::JUMP:
                        data.jumped = true;
                    break;
                default:
                    break;
                }

                vec = (vec * float(dt) * (is_sprinting ? 2.f : 1.f)) * data.move_speed;
            }

            vec *= 10;

            // Apply movement or decelerate on XZ only (preserve Y for gravity/jumps/explosions)
            JPH::Vec3 l = body_interface.GetLinearVelocity(rb.id);
            if (glm::length(vec) > 0.0f)
            {
                body_interface.SetLinearVelocity(rb.id, JPH::Vec3{vec.x, l.GetY(), vec.z});
            }
            else if (!data.in_air)
            {
                // Friction: decay XZ velocity when grounded and no input
                float friction = 0.85f;
                body_interface.SetLinearVelocity(rb.id, JPH::Vec3{l.GetX() * friction, l.GetY(), l.GetZ() * friction});
            }

            JPH::Quat rotation = JPH::Quat::sRotation(JPH::Vec3::sAxisY(), glm::radians(data.looking_angle));
            body_interface.SetRotation(rb.id, rotation, JPH::EActivation::Activate);

            // body_interface.SetAngularVelocity(rb.id, JPH::Vec3(0, 0, 0));
        }
    }
};

struct SeekerRenderHandler
{
    const Magic::Entity m_entity;
    unsigned int VBO = 0;
    Shader *m_shader;
    SeekerRenderHandler(Magic::Entity e) : m_entity(e)
    {
    }
    ~SeekerRenderHandler() {

    };

    void init()
    {
        // float vertices[] = {
        //     -0.5f, -0.5f, 0.0f, // Point 1
        //     0.5f, 0.5f, 0.0f    // Point 2
        // };

        // glGenBuffers(1, &VBO);
        // glBindBuffer(GL_ARRAY_BUFFER, VBO);
        // glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

        // glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
        // glEnableVertexAttribArray(0);
    }

    void destroy()
    {
        // glDeleteBuffers(1, &VBO);
    }
    void render(EntityManager &em, ResourceManager &rm, const Shader &shader, const Camera &cam)
    {
        // render health bar or something like that

        // glDrawArrays(GL_LINES, 0, 2);
    }

    Shader *shader()
    {
        return m_shader;
    }
};

// Track all seeker body IDs so contacts between seekers can be suppressed
static std::vector<JPH::BodyID> s_seeker_body_ids;

struct SeekerContactListener : public Magic::ContactListener
{
    void OnContactAdded(Magic::EntityManager &, Magic::Entity, const JPH::Body &inBody1, const JPH::Body &inBody2,
                        const JPH::ContactManifold &, JPH::ContactSettings &ioSettings) override
    {
        auto is_seeker = [](JPH::BodyID id) {
            for (const auto &sid : s_seeker_body_ids)
                if (sid == id) return true;
            return false;
        };
        if (is_seeker(inBody1.GetID()) && is_seeker(inBody2.GetID()))
            ioSettings.mIsSensor = true;
    }
};

namespace
{

    Seeker::Action chase_player(EntityManager &em, Entity e, double dt)
    {
        auto [lock, reg] = ((const EntityManager &)em).get_registry();
        // Physics &physics = Application::get().engine().get_physics();

        // find player
        Entity player = em.get("Player");

        if (player == entt::null)
            return Seeker::NONE;

        const Transform &player_t = reg.get<Transform>(player);
        const Transform &seeker_t = reg.get<Transform>(e);
        const Seeker::Data &data = reg.get<Seeker::Data>(e);

        // get direction
        Vec3 dir = player_t.pos - seeker_t.pos;
        float angle = glm::radians(-data.looking_angle); // Convert 90 degrees to radians
        glm::vec3 axis(0.0f, 1.0f, 0.0f);                // Rotate around the Y-axis
        glm::mat4 rotationMat = glm::rotate(glm::mat4(1.0f), angle, axis);

        dir = glm::vec3(rotationMat * glm::vec4(dir, 1.0f));
        float distance = glm::length(dir);

        // if player is far away, move towards them
        if (distance > 2.0f)
        {
            glm::vec2 diff = {dir.x, dir.z};

            // if player is to the left, turn left, if to the right,turn right
            float angle_to_player = glm::degrees(glm::atan(diff.x, diff.y));

            if (angle_to_player > 4.0f)
                return Seeker::TURN_LEFT;
            else if (angle_to_player < -4.0f)
                return Seeker::TURN_RIGHT;
            else
                return Seeker::MOVE_FORWARD;
        }

        return Seeker::NONE;
    }

} // namespace

Seeker::Seeker(EntityManager &em) : m_e(em.create("Seeker"))
{
    // using Type = Magic::Handle::Type;
    auto &app = Magic::Application::get();
    auto &rm = app.getResourceManager();

    // copy seeker model for now, should make a separate one later
    RenderObject seeker = rm.copy("Seeker");
    seeker.model(rm.copy(seeker.handle()));

    em.addComponent<RenderObject>(m_e, seeker);

    // SeekerInputHandler ih{m_e};
    SeekerUpdateHandler uh{m_e};

    uh.on_move = chase_player;
    uh.on_jump = [](EntityManager &em, Entity e, double dt)
    {
        LOG("JUMP");
        auto [lock, reg] = ((const EntityManager &)em).get_registry();
        const auto &data = reg.get<Seeker::Data>(e);

        Physics &physics = Application::get().engine().get_physics();
        JPH::BodyInterface &body_interface = physics.physics_system.GetBodyInterface();
        JPH::Vec3 l = body_interface.GetLinearVelocity(reg.get<Magic::RigidBody>(e).id);
        body_interface.SetLinearVelocity(reg.get<Magic::RigidBody>(e).id, l + JPH::Vec3(0, data.jump_force, 0));
    };

    uh.on_land = [](EntityManager &, Entity)
    {
        LOG("LAND");
    };

    uh.on_shoot = [this](EntityManager &em, Entity e, double dt)
    {
        LOG("shoot");
        //
        // UNSAFE
        //
        Camera &cam = Application::get().engine().cameras[0];
        shoot(em, cam);
    };
    // em.addComponent<Magic::InputHandler>(m_e, ih);
    em.addComponent<Magic::OnUpdateHandler>(m_e, uh);

    SeekerRenderHandler rh{m_e};

    rh.m_shader = &g_debug_line_shader;

    // rh.init();

    // em.addComponent<Magic::OnRenderHandler>(m_e, rh);
    // em.addComponent<Magic::Position>(m_e);
    Seeker::Data data = {};
    data.jump_force = 10.0f;
    data.move_speed = 50.0f;
    data.turn_speed = 90.0f;
    em.addComponent<Seeker::Data>(m_e, data);

    Physics &physics = Application::get().engine().get_physics();
    JPH::BodyInterface &body_interface = physics.physics_system.GetBodyInterface();

    using namespace JPH;
    using namespace JPH::literals;
    assert(physics.initialized());

    // Now create a dynamic body to bounce on the floor
    // Note that this uses the shorthand version of creating and adding a body to the world
    JPH::BoxShapeSettings shape_settings(JPH::Vec3(0.5f, 0.5f, 0.5f));
    shape_settings.mDensity = 80.0f; // 80 kg/m^3 -> 80 kg for a 1m^3 box
    JPH::Shape::ShapeResult shape_result = shape_settings.Create();
    JPH::ShapeRefC shape = shape_result.Get();
    JPH::BodyCreationSettings body_settings(shape, RVec3(0.0_r, 0.0_r, 0.0_r), JPH::Quat::sIdentity(), JPH::EMotionType::Dynamic, Layers::MOVING);
    body_settings.mFriction = 1.0f;
    body_settings.mAllowedDOFs = EAllowedDOFs::TranslationX | EAllowedDOFs::TranslationY | EAllowedDOFs::TranslationZ;

    // body_settings.mMaxLinearVelocity = 20.0f;
    JPH::BodyID body_id = body_interface.CreateAndAddBody(body_settings, JPH::EActivation::Activate);
    em.addComponent<Magic::RigidBody>(m_e, Magic::RigidBody{body_id});

    s_seeker_body_ids.push_back(body_id);
    em.addComponent<Magic::ContactListenerComponent>(m_e, SeekerContactListener{});

    // sphere_id = body_interface.CreateAndAddBody(sphere_settings, JPH::EActivation::Activate);
}

void Seeker::ClearBodyIds()
{
    s_seeker_body_ids.clear();
}

// grab

void Seeker::shoot(EntityManager &em, Camera &cam)
{
    auto &app = Magic::Application::get();
    auto &rm = app.getResourceManager();

    RenderObject cube = rm.copy("cool_box");

    glm::vec3 pos = cam.pos + cam.front * 1.0f;
    // make bullet
    Entity bullet = em.create({pos});

    // Transform &t = em.get_registry().get<Transform>(bullet);

    // t.rotation = glm::quatLookAt(glm::normalize(random_vec3), glm::vec3(0, 1, 0));

    // copy in order to keep all the underlying handles the same
    // there is no point to copy all of the data
    cube.model(rm.copy(cube.handle()));

    em.addComponent<Magic::RenderObject>(bullet, cube);

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
    body_settings.mMotionQuality = JPH::EMotionQuality::LinearCast;
    // body_settings.mMaxLinearVelocity = 20.0f;
    JPH::BodyID body_id = body_interface.CreateAndAddBody(body_settings, JPH::EActivation::Activate);

    Magic::Vec3 force = cam.front * 10.0f;
    body_interface.SetLinearVelocity(body_id, JPH::Vec3(force.x, force.y, force.z));
    em.addComponent<Magic::RigidBody>(bullet, Magic::RigidBody{body_id});
}

void Seeker::LoadModel()
{
    using Type = Magic::Handle::Type;
    // Load seeker model here
    auto &rm = GetResourceManager();
    if (rm.exists("Seeker"))
    {
        LOG("Seeker model already exists");
        return;
    }
    RenderObject &obj = rm.make("Seeker");
    auto handle = rm.copy((ModelHandle &)rm.get("cube", Type::MODEL));
    Model &cube = rm.get(handle);
    MeshHandle mesh_handle = rm.copy(cube.get(0));
    Mesh &mesh = rm.get(mesh_handle);
    mesh.set(rm.get("container2", Type::TEXTURE), 1);
    mesh.set(rm.get("container2_specular2", Type::TEXTURE), 2);
    cube.set(mesh_handle, 0);
    obj.model(handle);
}

// python bindings
