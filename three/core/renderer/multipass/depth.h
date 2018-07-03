#pragma once
#include "../pass.h"
#include <gl3w/gl3w.h>

namespace three {
namespace renderer {
    namespace multipass {
        class DepthBuffer : public RenderPass {
        private:
            GLuint _render_result_texture_id;
            int _viewport_width;
            int _viewport_height;

        public:
            DepthBuffer(int viewport_width, int viewport_height);
            bool bind();
            void unbind();
            GLuint get_buffer_texture_id();
        };
    }
}
}