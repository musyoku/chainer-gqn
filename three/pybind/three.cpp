#include "../core/camera/perspective.h"
#include "../core/scene/object.h"
#include "../core/scene/scene.h"
#include <pybind11/pybind11.h>
namespace py = pybind11;
using namespace three;

PYBIND11_MODULE(three, module)
{
    py::class_<camera::PerspectiveCamera>(module, "PerspectiveCamera")
        .def(py::init<py::tuple, py::tuple, py::tuple, float, float, float, float>(),
            py::arg("eye"), py::arg("center"), py::arg("up"), py::arg("fov_rad"), py::arg("aspect_ratio"), py::arg("z_near"), py::arg("z_far"))
        .def("look_at", &camera::PerspectiveCamera::look_at, py::arg("eye"), py::arg("center"), py::arg("up"));

    py::class_<scene::Scene>(module, "Scene")
        .def(py::init<>())
        .def("add", &scene::Scene::add);

    py::class_<scene::Object>(module, "Object")
        .def(py::init<py::array_t<int>, py::array_t<float>, py::tuple>(),
            py::arg("faces"), py::arg("vertices"), py::arg("color"))
        .def("set_position", &scene::Object::set_position)
        .def("set_rotation", &scene::Object::set_rotation);
}