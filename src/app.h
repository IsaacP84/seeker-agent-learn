#pragma once
#include <Magic!/magic!.hpp>

class Application : public Magic::Application
{
public:
    // using Magic::Application::Application;
    Application(Magic::ApplicationConfiguration config);
    ~Application();

    virtual void onInit() override;
    virtual void onInitPython(nanobind::module_ &) override;
    virtual void onAfterPython(nanobind::module_ &) override;
    virtual void onQuit() override;
    virtual void onUpdate(const double &) override;
    virtual void onRender() override;  
private:
};

