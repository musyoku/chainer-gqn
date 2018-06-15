#pragma once
#include <glm/glm.hpp>

namespace environment {
namespace camera {
    class Camera {
    public:
        glm::mat4 _view_matrix;
        glm::mat4 _projection_matrix;
    };
}
}