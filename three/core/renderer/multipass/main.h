#pragma once
#include <gl3w/gl3w.h>
#include <memory>

namespace three {
namespace renderer {
    namespace multipass {
        class Main {
        private:
            GLuint _program;
            GLuint _depth_sampler;
            GLuint _ssao_sampler;
            GLuint _depth_texture;
            GLuint _ssao_texture;
            std::unique_ptr<GLfloat[]> _ssao_texture_data;

        public:
            Main();
            void use();
            void uniform_matrix(GLuint location, const GLfloat* matrix);
            void uniform_float(GLuint location, const GLfloat value);
            void attach_depth_texture();
            void bind_textures();
        };
    }
}
}