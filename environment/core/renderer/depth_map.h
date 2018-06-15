#pragma once
#include "../camera/perspective.h"
#include "../scene/scene.h"

namespace environment {
namespace renderer {
    namespace rasterizer {
        void render_depth_map(scene::Scene* scene, camera::PerspectiveCamera* camera);
    }
}
}