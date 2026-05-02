#include <Magic!/python/registry.hpp>
#include "init.h"

namespace {

struct AppPythonRegistrator {
    AppPythonRegistrator() {
        Magic::Python::registerExtension([](nanobind::module_ &m){
            init_navigation(m);
        });
    }
};

static AppPythonRegistrator registrator;

} // anonymous namespace


