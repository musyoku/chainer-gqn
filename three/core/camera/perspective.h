#pragma once
#include "../base/camera.h"
#include <pybind11/pybind11.h>

namespace three {
namespace camera {
    namespace py = pybind11;
    class PerspectiveCamera : public base::Camera {
    public:
        PerspectiveCamera(py::tuple eye, py::tuple center, py::tuple up, float fov_rad, float aspect_ratio, float z_near, float z_far);
    };
}
}