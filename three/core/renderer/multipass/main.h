#pragma once
#include <gl3w/gl3w.h>

namespace three {
namespace renderer {
    namespace multipass {
        class Main {
        private:
            GLuint _program;
            GLuint _uniform_projection_mat;
            GLuint _uniform_model_mat;
            GLuint _uniform_view_mat;
            GLuint _uniform_smoothness;

        public:
            Main();
            void use();
            void uniform_matrix(GLuint location, const GLfloat* matrix);
            void uniform_float(GLuint location, const GLfloat value);
        };
    }
}
}