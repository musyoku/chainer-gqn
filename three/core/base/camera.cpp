#include "camera.h"
#include <glm/gtc/matrix_transform.hpp>

namespace three {
namespace camera {
    void Camera::look_at(py::tuple eye, py::tuple center, py::tuple up)
    {
        _view_matrix = glm::lookAtRH(glm::vec3(eye[0].cast<float>(), eye[1].cast<float>(), eye[2].cast<float>()),
            glm::vec3(center[0].cast<float>(), center[1].cast<float>(), center[2].cast<float>()),
            glm::vec3(up[0].cast<float>(), up[1].cast<float>(), up[2].cast<float>()));
    }
}
}