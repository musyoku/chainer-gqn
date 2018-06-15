#pragma once
#include <glm/glm.hpp>

namespace environment {
namespace camera {
    class Camera {
    public:
        glm::mat3 _view_matrix;
    };
}
}