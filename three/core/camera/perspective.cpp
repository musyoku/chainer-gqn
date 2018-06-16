#include "perspective.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace three {
namespace camera {
    PerspectiveCamera::PerspectiveCamera(py::tuple eye, py::tuple center, py::tuple up, float fov_rad, float aspect_ratio, float z_near, float z_far)
    {
        _view_matrix = glm::lookAtRH(glm::vec3(eye[0].cast<float>(), eye[1].cast<float>(), eye[2].cast<float>()),
            glm::vec3(center[0].cast<float>(), center[1].cast<float>(), center[2].cast<float>()),
            glm::vec3(up[0].cast<float>(), up[1].cast<float>(), up[2].cast<float>()));
        _projection_matrix = glm::perspectiveFovRH_ZO(fov_rad, aspect_ratio, 1.0f, z_near, z_far);
    }
}
}