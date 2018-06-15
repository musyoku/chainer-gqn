#include "perspective.h"
#include <glm/glm.hpp>

namespace environment {
namespace camera {
    PerspectiveCamera::PerspectiveCamera(py::tuple eye, py::tuple center, py::tuple up)
    {
        _view_matrix[0] = glm::vec3(eye[0].cast<float>(), eye[1].cast<float>(), eye[2].cast<float>());
        _view_matrix[1] = glm::vec3(center[0].cast<float>(), center[1].cast<float>(), center[2].cast<float>());
        _view_matrix[2] = glm::vec3(up[0].cast<float>(), up[1].cast<float>(), up[2].cast<float>());
    }
}
}