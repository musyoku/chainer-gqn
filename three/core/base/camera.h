#pragma once
#include <glm/glm.hpp>
#include <pybind11/pybind11.h>

namespace three {
namespace base {
    namespace py = pybind11;
    class Camera {
    public:
        glm::mat4 _view_matrix;
        glm::mat4 _projection_matrix;
        void look_at(py::tuple eye, py::tuple center, py::tuple up);
    };
}
}