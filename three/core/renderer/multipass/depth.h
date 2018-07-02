#pragma once
#include "../pass.h"
#include <gl3w/gl3w.h>

namespace three {
namespace renderer {
    namespace multipass {
        class DepthBuffer : public RenderPass {
        private:
            GLuint _sampler_id;
            GLuint _texture_id;
            int _viewport_width;
            int _viewport_height;
            void reserve(int viewport_width, int viewport_height);
            void reserve_if_needed(int viewport_width, int viewport_height);

        public:
            DepthBuffer(int viewport_width, int viewport_height);
            bool bind(int viewport_width, int viewport_height);
            void unbind();
        };
    }
}
}