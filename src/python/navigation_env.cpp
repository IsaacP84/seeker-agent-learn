#include "init.h"
#include "../navigation_env.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/list.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/unique_ptr.h>

void init_navigation(nb::module_ &m)
{
    nb::class_<NavigationEnv>(m, "NavigationEnv")
        .def(nb::init<>())
        .def("reset",           &NavigationEnv::reset)
        .def("get_observation", &NavigationEnv::get_observation)
        .def("compute_reward",  &NavigationEnv::compute_reward)
        .def("is_done",         &NavigationEnv::is_done)
        .def("apply_action",    &NavigationEnv::apply_action)
        .def_prop_ro_static("num_states", [](nb::object) { return NavigationEnv::Sizes::num_states; })
        .def_prop_ro_static("num_actions", [](nb::object) { return NavigationEnv::Sizes::num_actions; })
        .def("get_env_data",  &NavigationEnv::get_env_data)
        .def("set_env_data",  &NavigationEnv::set_env_data);
}

