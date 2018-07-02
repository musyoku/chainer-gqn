#pragma once
#include "../pass.h"
#include <gl3w/gl3w.h>
#include <memory>

namespace three {
namespace renderer {
    namespace multipass {
        class Main : public RenderPass {
        private:
            GLuint _color_render_buffer;
            GLuint _depth_render_buffer;
            GLuint _ssao_buffer_sampler;
            int _viewport_width;
            int _viewport_height;

        public:
            Main(int viewport_width, int viewport_height);
            bool bind();
            void unbind();
            virtual void set_additional_uniform_variables(base::Object* object);
        };
    }
}
}