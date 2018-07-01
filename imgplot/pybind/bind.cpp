#include "../core/data/image.h"
#include "../core/figure.h"
#include "../core/view/image.h"
#include "../core/window.h"
#include <pybind11/pybind11.h>
namespace py = pybind11;
using namespace imgplot;

PYBIND11_MODULE(imgplot, module)
{
    py::class_<data::ImageData, std::shared_ptr<data::ImageData>>(module, "image")
        .def(py::init<>())
        .def(py::init<py::array_t<GLubyte>>(), py::arg("data"))
        .def("update", &data::ImageData::update);

    py::class_<Figure>(module, "figure")
        .def(py::init<>())
        .def("add", &Figure::add, py::arg("data"), py::arg("x"), py::arg("y"), py::arg("width"), py::arg("height"));

    py::class_<Window>(module, "window")
        .def(py::init<Figure*, py::tuple, std::string>(), py::arg("figure"), py::arg("size"), py::arg("title"))
        .def("closed", &Window::closed)
        .def("show", &Window::show);
}