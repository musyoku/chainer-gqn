#include "../core/renderer/renderer.h"
#include <pybind11/pybind11.h>
namespace py = pybind11;
using namespace three;

PYBIND11_MODULE(renderer, module)
{
    py::class_<renderer::Renderer>(module, "Renderer")
        .def(py::init<int, int>(), py::arg("width"), py::arg("height"))
        .def("render_depth_map", &renderer::Renderer::render_depth_map);
}