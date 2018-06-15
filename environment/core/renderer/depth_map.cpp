#include "depth_map.h"
#include "../scene/object.h"
#include <glm/glm.hpp>
#include <memory>
#include <vector>

namespace environment {
namespace renderer {
    namespace rasterizer {
        void render_depth_map(scene::Scene* scene, camera::PerspectiveCamera* camera)
        {
            glm::mat3& view_mat = camera->_view_matrix;
            std::vector<std::unique_ptr<scene::Object>>& objects = scene->_objects;
            for (auto& object : objects) {
            }
        }
    }
}
}