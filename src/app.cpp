#include "app.h"
#include "ui.h"
#include "scenes/StartScene.hpp"
#include "scenes/LearnScene.hpp"
#include "scenes/TestScene.hpp"
#include "python/init.h"


#include <Magic!/debug/debug.hpp>
#include <Magic!/core/camera.hpp>

#include <Magic!/python/interpreter.hpp>
namespace nb = nanobind;

bool enable_zoom = true;
Magic::Shader *light_shader = nullptr;

Magic::Application *Magic::CreateApplication(Magic::ApplicationCommandLineArgs)
{
    ApplicationConfiguration config = {.use_vsync = true};
    return new ::Application(config);
}

Application::Application(Magic::ApplicationConfiguration config)
    : Magic::Application(config)
{
}

Application::~Application()
{
}

void Application::onInitPython(nb::module_ &m)
{
    Magic::Application::onInitPython(m);

    int code = Magic::Python::Interpreter::run(SCRIPTS_FOLDER / "app.py");
    if (code != 0)
    {
        std::string error = std::format("Error executing app.py: error code {}", code);
        Magic::Debug::Log(error);
        throw std::runtime_error(error);
    }
    else
    {
        Magic::Debug::Log("Python script executed successfully!");
    }
}
void Application::onAfterPython(nb::module_ &m)
{
}

void Application::onInit()
{
    using namespace Magic;
    LOG("init");
    auto &sm = getSceneManager();
    auto &rm = GetResourceManager();

    nb::gil_scoped_acquire acquire;

    CurrentCamera().set_zoom_constraints(40, 60);
    using Type = Magic::Handle::Type;

    {
        ModelHandle cube_handle;

        Model cube;
        Cube mesh = {};
        mesh.init();
        MeshHandle mesh_handle = rm.add(mesh);

        cube.add(mesh_handle);
        cube.init(rm);
        cube_handle = rm.add("cube", cube);
    }

    {
        RenderObject &light = rm.make("light");
        auto cube_handle = (ModelHandle &)rm.get("cube", Type::MODEL);
        auto handle = rm.copy(cube_handle);
        light.model(handle);

        Model &model = light.model();
        model.model = glm::scale(model.model, glm::vec3(0.1f));

        g_shaders.push_back(Shader("light"));
        light_shader = &g_shaders.back();
        light_shader->stage = RenderStage::FORWARD;
        model.m_shader = light_shader;
    }

    {
        RenderObject &cube_obj = rm.make("cool_box");
        auto cube_handle = (ModelHandle &)rm.get("cube", Type::MODEL);
        auto handle = rm.copy(cube_handle);
        Model &cube = rm.get(handle);
        MeshHandle mesh_handle = rm.copy(cube.get(0));
        Mesh &mesh = rm.get(mesh_handle);
        mesh.set(rm.get("container2", Type::TEXTURE), 1);
        mesh.set(rm.get("container2_specular2", Type::TEXTURE), 2);
        mesh.set(rm.get("matrix", Type::TEXTURE), 4);
        cube.model = glm::scale(cube.model, glm::vec3(0.25f));
        cube.set(mesh_handle, 0);
        cube_obj.model(handle);
    }

    sm.add_scene<StartScene>("start", sm);
    sm.add_scene<LearnScene>("learn", sm);
    sm.add_scene<TestScene>("test", sm);
    
}

void Application::onQuit()
{
    LOG("Quit");
}

void Application::onUpdate(const double &)
{
}

void Application::onRender()
{
}

void PauseHelper::handle_input(const SDL_Event &e)
{
    switch (e.type)
    {
    case SDL_EVENT_KEY_DOWN:
        if (e.key.repeat)
            return;
        switch (e.key.scancode)
        {
        case SDL_SCANCODE_ESCAPE:
            if (!Magic::GetEngine().is_paused())
                pause();
            else
                unpause();
            break;
        default:
            break;
        }

        {
            const bool *state = Magic::Engine::GetKeyboardState();
            
            if (state[SDL_SCANCODE_LCTRL] && state[SDL_SCANCODE_1])
            {
                // resize to 360p
                pause();
                Magic::GetApplication().getWindow().resize(640, 360);
                unpause();
            }

            if (state[SDL_SCANCODE_LCTRL] && state[SDL_SCANCODE_2])
            {
                // resize to 720p
                pause();
                Magic::GetApplication().getWindow().resize(1280, 720);
                unpause();
            }
        }
        
        break;
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
        if (Magic::GetEngine().is_paused())
            unpause();
        break;
    default:
        break;
    }
}

void PauseHelper::pause()
{
    LOG("Paused");
    if (!Application::get().SetRelativeMouseMode(false))
    {
        LOG(Magic::GetError());
        assert(false);
    }

    Magic::GetEngine().pause();
}

void PauseHelper::unpause()
{
    LOG("Unpaused");
    if (!Application::get().SetRelativeMouseMode(true))
    {
        LOG(Magic::GetError());
        assert(false);
    }

    Magic::GetEngine().unpause();
}

Magic::Vec3 hexToRgbStoi(std::string hex)
{
    // Remove the '#' if present
    if (hex[0] == '#')
    {
        hex.erase(0, 1);
    }

    // Ensure the string is the correct length for RGB (6 characters)
    if (hex.length() != 6 && hex.length() != 8)
    {
        // Handle error or default
        return {0, 0, 0};
    }

    // Extract substrings and convert to integers (base 16)
    Magic::Vec3 color;
    try
    {
        color.x = std::stoi(hex.substr(0, 2), nullptr, 16);
        color.y = std::stoi(hex.substr(2, 2), nullptr, 16);
        color.z = std::stoi(hex.substr(4, 2), nullptr, 16);
    }
    catch (...)
    {
        // Handle potential exceptions (e.g., invalid format)
        return {0, 0, 0};
    }

    // map to 0-1
    color.x /= 255.0f;
    color.y /= 255.0f;
    color.z /= 255.0f;

    return color;
}

void free_cam_motion_input_handler(const SDL_Event &e, Magic::Camera &cam)
{
    switch (e.type)
    {

    case SDL_EVENT_MOUSE_MOTION:
    {
        int dx = e.motion.xrel; // Delta X
        int dy = e.motion.yrel; // Delta Y

        Magic::move_with_mouse(cam, dx, dy);
        cam.update();
        break;
    }

    case SDL_EVENT_MOUSE_WHEEL:
    {
        if (enable_zoom)
        {
            Magic::zoom_with_mouse(cam, e.wheel.y);
            cam.update();
        }
        break;
    }
    }
}

void free_cam_motion_update_handler(double dt, Magic::Camera &cam)
{
    const bool *state = Magic::Engine::GetKeyboardState();
    float cameraSpeed = 1 * dt; // adjust accordingly
    if (state[SDL_SCANCODE_LSHIFT])
    {
        cameraSpeed *= 2;
    }

    if (state[SDL_SCANCODE_W])
        cam.pos += cameraSpeed * cam.front;
    if (state[SDL_SCANCODE_S])
        cam.pos -= cameraSpeed * cam.front;
    if (state[SDL_SCANCODE_A])
        cam.pos -= glm::normalize(glm::cross(cam.front, cam.up)) * cameraSpeed;
    if (state[SDL_SCANCODE_D])
        cam.pos += glm::normalize(glm::cross(cam.front, cam.up)) * cameraSpeed;
    if (state[SDL_SCANCODE_E])
        cam.pos += cam.up * cameraSpeed;
    if (state[SDL_SCANCODE_Q])
        cam.pos -= cam.up * cameraSpeed;
}

// should only be used if you need to get the entity
Magic::Entity find_entity_from_hitbox(const entt::registry &reg, JPH::BodyID id)
{
    auto view = reg.view<Magic::RigidBody>();

    for (auto [e, rb] : view.each())
    {
        if (rb.id == id)
            return e;
    }
    return entt::null;
}

std::vector<std::string> get_files_in_folder(const fs::path &directory_path)
{
    std::vector<std::string> file_list;

    if (fs::exists(directory_path) && fs::is_directory(directory_path))
    {
        // Iterate over the directory entries
        for (const auto &entry : fs::directory_iterator(directory_path))
        {
            // Check if the entry is a regular file and not a directory
            if (fs::is_regular_file(entry.status()))
            {
                file_list.push_back(entry.path().string());
            }
        }
    }
    else
    {
        std::cerr << "Directory not found or is not a directory: " << directory_path << std::endl;
    }

    return file_list;
}
