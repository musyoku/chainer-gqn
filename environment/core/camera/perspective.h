#pragma once
#include "../base/camera.h"
#include <pybind11/numpy.h>

namespace environment {
namespace camera {
    namespace py = pybind11;
    class PerspectiveCamera : public Camera {
        PerspectiveCamera(py::tuple eye, py::tuple center, py::tuple up);
    };
}
}