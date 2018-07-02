#pragma once
#include "../pass.h"
#include <gl3w/gl3w.h>
#include <memory>

namespace three {
namespace renderer {
    namespace multipass {
        class Blur : public RenderPass {
        private:
            GLuint _color_render_buffer;
            GLuint _buffer_sampler;
            GLuint _render_result_texture_id;
            int _viewport_width;
            int _viewport_height;

        public:
            Blur(int viewport_width, int viewport_height);
            bool bind(GLuint ssao_buffer_texture_id);
            void unbind();
            GLuint get_buffer_texture_id();
        };
    }
}
}