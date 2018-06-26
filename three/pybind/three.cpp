#include "../core/camera/perspective.h"
#include "../core/renderer/renderer.h"
#include "../core/scene/cornell_box.h"
#include "../core/scene/object.h"
#include "../core/scene/scene.h"
#include <gl3w/gl3w.h>
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
        .def("add", (void (scene::Scene::*)(std::shared_ptr<scene::Object>)) & scene::Scene::add)
        .def("add", (void (scene::Scene::*)(std::shared_ptr<scene::Object>, py::tuple)) & scene::Scene::add,
            py::arg("object"), py::arg("position"))
        .def("add", (void (scene::Scene::*)(std::shared_ptr<scene::Object>, py::tuple, py::tuple)) & scene::Scene::add,
            py::arg("object"), py::arg("position"), py::arg("rotation"))
        .def("add", (void (scene::Scene::*)(std::shared_ptr<scene::CornellBox>)) & scene::Scene::add)
        .def("add", (void (scene::Scene::*)(std::shared_ptr<scene::CornellBox>, py::tuple)) & scene::Scene::add,
            py::arg("object"), py::arg("position"))
        .def("add", (void (scene::Scene::*)(std::shared_ptr<scene::CornellBox>, py::tuple, py::tuple)) & scene::Scene::add,
            py::arg("object"), py::arg("position"), py::arg("rotation"));

    py::class_<scene::Object, std::shared_ptr<scene::Object>>(module, "Object")
        .def(py::init<py::array_t<int>,
                 py::array_t<float>,
                 py::tuple,
                 py::tuple,
                 bool>(),
            py::arg("faces"),
            py::arg("vertices"),
            py::arg("color"),
            py::arg("scale"),
            py::arg("smoothness"))
        .def("clone", &scene::Object::clone)
        .def("set_color", &scene::Object::set_color)
        .def("set_scale", &scene::Object::set_scale)
        .def("set_position", &scene::Object::set_position)
        .def("set_rotation", &scene::Object::set_rotation);

    py::class_<scene::CornellBox, std::shared_ptr<scene::CornellBox>>(module, "CornellBox")
        .def(py::init<py::tuple,
                 py::tuple,
                 py::tuple,
                 py::tuple,
                 py::tuple>(),
            py::arg("north_wall_color"),
            py::arg("east_wall_color"),
            py::arg("south_wall_color"),
            py::arg("west_wall_color"),
            py::arg("scale"))
        .def("clone", &scene::CornellBox::clone)
        .def("set_north_wall_color", &scene::CornellBox::set_north_wall_color)
        .def("set_east_wall_color", &scene::CornellBox::set_east_wall_color)
        .def("set_south_wall_color", &scene::CornellBox::set_south_wall_color)
        .def("set_west_wall_color", &scene::CornellBox::set_west_wall_color)
        .def("set_scale", &scene::CornellBox::set_scale)
        .def("set_position", &scene::CornellBox::set_position)
        .def("set_rotation", &scene::CornellBox::set_rotation);

    py::class_<renderer::Renderer>(module, "Renderer")
        .def(py::init<int, int>(), py::arg("width"), py::arg("height"))
        .def(py::init<scene::Scene*, int, int>(), py::arg("scene"), py::arg("width"), py::arg("height"))
        .def("render_depth_map", (void (renderer::Renderer::*)(camera::PerspectiveCamera*, py::array_t<GLfloat, py::array::c_style>)) & renderer::Renderer::render_depth_map)
        .def("render_depth_map", (void (renderer::Renderer::*)(scene::Scene*, camera::PerspectiveCamera*, py::array_t<GLfloat, py::array::c_style>)) & renderer::Renderer::render_depth_map)
        .def("render", (void (renderer::Renderer::*)(camera::PerspectiveCamera*, py::array_t<GLuint, py::array::c_style>)) & renderer::Renderer::render)
        .def("render", (void (renderer::Renderer::*)(scene::Scene*, camera::PerspectiveCamera*, py::array_t<GLuint, py::array::c_style>)) & renderer::Renderer::render)
        .def("set_scene", &renderer::Renderer::set_scene);
}