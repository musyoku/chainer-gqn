#pragma once
#include "../base/camera.h"
#include <pybind11/numpy.h>

namespace environment {
namespace camera {
    namespace py = pybind11;
    class PerspectiveCamera : public Camera {
    public:
        PerspectiveCamera(py::tuple eye, py::tuple center, py::tuple up, float fov_rad, float aspect_ratio, float z_near, float z_far);
        void look_at(py::tuple eye, py::tuple center, py::tuple up);
    };
}
}