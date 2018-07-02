#include "pass.h"

namespace three {
namespace renderer {
    void RenderPass::uniform_matrix_4fv(GLuint location, const GLfloat* matrix)
    {
        glUniformMatrix4fv(location, 1, GL_FALSE, matrix);
    }
    void RenderPass::uniform_1f(GLuint location, const GLfloat value)
    {
        glUniform1f(location, value);
    }
    void RenderPass::uniform_1i(GLuint location, const GLint value)
    {
        glUniform1i(location, value);
    }
    void RenderPass::set_additional_uniform_variables(base::Object* object)
    {
    }
    void RenderPass::check_framebuffer_status()
    {
        GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (status == GL_FRAMEBUFFER_COMPLETE) {
            return;
        }
        if (status == GL_FRAMEBUFFER_UNDEFINED) {
            throw std::runtime_error("GL_FRAMEBUFFER_UNDEFINED: The specified framebuffer is the default read or draw framebuffer, but the default framebuffer does not exist.");
        }
        if (status == GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT) {
            throw std::runtime_error("GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT : Any of the framebuffer attachment points are framebuffer incomplete.");
        }
        if (status == GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT) {
            throw std::runtime_error("GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT: The framebuffer does not have at least one image attached to it.");
        }
        if (status == GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER) {
            throw std::runtime_error("GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER: The value of GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE is GL_NONE for any color attachment point(s) named by GL_DRAW_BUFFERi.");
        }
        if (status == GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER) {
            throw std::runtime_error("GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER: GL_READ_BUFFER is not GL_NONE and the value of GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE is GL_NONE for the color attachment point named by GL_READ_BUFFER.");
        }
        if (status == GL_FRAMEBUFFER_UNSUPPORTED) {
            throw std::runtime_error("GL_FRAMEBUFFER_UNSUPPORTED: The combination of internal formats of the attached images violates an implementation-dependent set of restrictions.");
        }
        if (status == GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE) {
            throw std::runtime_error("GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE: The value of GL_RENDERBUFFER_SAMPLES is not the same for all attached renderbuffers; if the value of GL_TEXTURE_SAMPLES is the not same for all attached textures; or, if the attached images are a mix of renderbuffers and textures, the value of GL_RENDERBUFFER_SAMPLES does not match the value of GL_TEXTURE_SAMPLES.");
        }
        if (status == GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE) {
            throw std::runtime_error("GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE: The value of GL_TEXTURE_FIXED_SAMPLE_LOCATIONS is not the same for all attached textures; or, if the attached images are a mix of renderbuffers and textures, the value of GL_TEXTURE_FIXED_SAMPLE_LOCATIONS is not GL_TRUE for all attached textures.");
        }
        if (status == GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS) {
            throw std::runtime_error("GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS: Any framebuffer attachment is layered, and any populated attachment is not layered, or if all populated color attachments are not from textures of the same target.");
        }
    }
}
}