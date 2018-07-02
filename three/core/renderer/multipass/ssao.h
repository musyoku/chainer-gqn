#pragma once
#include "../pass.h"
#include <gl3w/gl3w.h>
#include <memory>

namespace three {
namespace renderer {
    namespace multipass {
        class ScreenSpaceAmbientOcculusion : public RenderPass {
        private:
            GLuint _ssao_sampler;
            GLuint _depth_texture;
            GLuint _ssao_texture;
            std::unique_ptr<GLfloat[]> _ssao_texture_data;

        public:
            ScreenSpaceAmbientOcculusion();
            bool bind(int num_sampling_points);
            void unbind();
        };
    }
}
}