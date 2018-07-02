#include "depth.h"
#include "../opengl/functions.h"

namespace three {
namespace renderer {
    namespace multipass {
        DepthBuffer::DepthBuffer(int viewport_width, int viewport_height)
        {
            const GLchar vertex_shader[] = R"(
#version 450
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 face_normal_vector;
layout(location = 2) in vec3 vertex_normal_vector;
layout(location = 3) in vec4 vertex_color;
layout(location = 0) uniform mat4 model_mat;
layout(location = 1) uniform mat4 view_mat;
layout(location = 2) uniform mat4 projection_mat;
out vec3 frag_light_direction;
void main(void)
{
    gl_Position = projection_mat * view_mat * model_mat * vec4(position, 1.0f);
    vec4 model_position = model_mat * vec4(position, 1.0f);
    vec3 light_position = vec3(0.0f, 1.0f, 1.0f);
    frag_light_direction = light_position - model_position.xyz;
}
)";

            const GLchar fragment_shader[] = R"(
#version 450
in vec3 frag_light_direction;
out vec4 frag_color;
void main(){
    // float light_distance = length(frag_light_direction);
    // float attenuation = clamp(1.0 / (1.0 + 0.1 * light_distance + 0.2 * light_distance * light_distance), 0.0f, 1.0f);
    // frag_color = vec4(vec3(attenuation), 1.0);
}
)";

            _program = opengl::create_program(vertex_shader, fragment_shader);

            glCreateFramebuffers(1, &_fbo);

            reserve(viewport_width, viewport_height);
        }
        void DepthBuffer::reserve(int viewport_width, int viewport_height)
        {
            glDeleteTextures(1, &_texture_id);

            _viewport_width = viewport_width;
            _viewport_height = viewport_height;

            glGenTextures(1, &_texture_id);
            glBindTexture(GL_TEXTURE_2D, _texture_id);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, viewport_width, viewport_height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
            glBindTexture(GL_TEXTURE_2D, 0);

            glTextureParameteri(_texture_id, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTextureParameteri(_texture_id, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTextureParameteri(_texture_id, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
            glTextureParameteri(_texture_id, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
        }
        void DepthBuffer::reserve_if_needed(int viewport_width, int viewport_height)
        {
            if (viewport_width != _viewport_width) {
                return reserve(viewport_width, viewport_height);
            }
            if (viewport_height != _viewport_height) {
                return reserve(viewport_width, viewport_height);
            }
        }
        bool DepthBuffer::bind(int viewport_width, int viewport_height)
        {
            reserve_if_needed(viewport_width, viewport_height);

            glUseProgram(_program);
            glBindFramebuffer(GL_FRAMEBUFFER, _fbo);
            glNamedFramebufferTexture(_fbo, GL_DEPTH_ATTACHMENT, _texture_id, 0);
            glNamedFramebufferDrawBuffer(_fbo, GL_NONE);
            glClear(GL_DEPTH_BUFFER_BIT);
            check_framebuffer_status();
            glViewport(0, 0, viewport_width, viewport_height);
            return true;
        }
        void DepthBuffer::unbind()
        {
            glBindVertexArray(0);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glUseProgram(0);
            check_framebuffer_status();
        }
        void DepthBuffer::bind_depth_buffer()
        {
            glBindTexture(GL_TEXTURE_2D, _texture_id);
        }
    }
}
}