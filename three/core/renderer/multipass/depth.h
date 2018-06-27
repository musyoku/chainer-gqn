#pragma once
#include <gl3w/gl3w.h>

namespace three {
namespace renderer {
    namespace multipass {
        class Depth {
        private:
            GLuint _program;

        public:
            Depth();
            void use();
            void uniform_matrix(GLuint location, const GLfloat* matrix);
        };
    }
}
}