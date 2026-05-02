#pragma once
#include <Magic!/ui/ui.hpp>
class MyMenu : public Magic::Menu, public Magic::Draggable
{
public:
    struct Config : public Magic::Menu::Config
    {
        Config() : Magic::Menu::Config() {};
        Config(Magic::Menu::Config &&cfg) : Magic::Menu::Config(cfg) {};
        Config(double x, double y, int w, int h) : Magic::Menu::Config{x, y, w, h} {};
        // Config(double x, double y, int w, int h) : x{x}, m_y{y}, m_w{w}, m_h{h} {};
    };

    MyMenu(const std::string &id, const Config &cfg = {}) : Magic::Element{id, cfg}, Magic::Menu{id, cfg} {};
    ~MyMenu() {};
    virtual void handle_input(const SDL_Event &e) override
    {
        Magic::Menu::handle_input(e);
        Magic::Draggable::handle_input(e);
    }
    virtual void render(Magic::ResourceManager &rm) override
    {
        Magic::Menu::render(rm);
    }
};
