#pragma once
#include "../base/object.h"
#include <gl3w/gl3w.h>
#include <memory>

namespace three {
namespace renderer {
    class RenderPass {
    protected:
        GLuint _program;
        GLuint _fbo;

    public:
        void uniform_matrix_4fv(GLuint location, const GLfloat* matrix);
        void uniform_1f(GLuint location, const GLfloat value);
        void uniform_1i(GLuint location, const GLint value);
        virtual void set_additional_uniform_variables(base::Object* object);
        void check_framebuffer_status();
    };
}
}