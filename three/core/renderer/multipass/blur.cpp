#include "blur.h"
#include "../opengl/functions.h"
#include <random>

namespace three {
namespace renderer {
    namespace multipass {
        Blur::Blur(int viewport_width, int viewport_height)
        {
            _viewport_width = viewport_width;
            _viewport_height = viewport_height;

            const GLchar vertex_shader[] = R"(
#version 450
layout(location = 0) in vec3 position;
layout(location = 0) uniform mat4 model_mat;
layout(location = 1) uniform mat4 view_mat;
layout(location = 2) uniform mat4 projection_mat;
layout(location = 3) uniform float screen_width;
layout(location = 4) uniform float screen_height;
flat out float frag_screen_width;
flat out float frag_screen_height;
void main(void)
{
    gl_Position = projection_mat * view_mat * model_mat * vec4(position, 1.0f);
    frag_screen_width = screen_width;
    frag_screen_height = screen_height;
}
)";

            const GLchar fragment_shader[] = R"(
#version 450
layout(binding = 0) uniform sampler2D ssao_buffer;
flat in float frag_screen_width;
flat in float frag_screen_height;
out float frag_color;

void main(){
    vec2 texcoord = gl_FragCoord.xy / vec2(frag_screen_width, frag_screen_height);
    vec2 texel_size = vec2(1.0 / frag_screen_width, 1.0 / frag_screen_height);
    float sum = 0.0;
    int blur_radius = 1;
    for(int y = -blur_radius;y <= blur_radius;y++){
        for(int x = -blur_radius;x <= blur_radius;x++){
            vec2 offset = vec2(texel_size.x * float(x), texel_size.y * float(y));
            float luminance = texture(ssao_buffer, texcoord + offset)[0];
            sum += luminance;
        }
    }
    frag_color = sum / float((2 * blur_radius + 1) * (2 * blur_radius + 1));
}
)";

            _program = opengl::create_program(vertex_shader, fragment_shader);

            glCreateFramebuffers(1, &_fbo);

            glGenTextures(1, &_render_result_texture_id);
            glBindTexture(GL_TEXTURE_2D, _render_result_texture_id);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, viewport_width, viewport_height, 0, GL_RGB, GL_FLOAT, 0);
            glBindTexture(GL_TEXTURE_2D, 0);

            glGenSamplers(1, &_buffer_sampler);
            glSamplerParameteri(_buffer_sampler, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
            glSamplerParameteri(_buffer_sampler, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
            glSamplerParameteri(_buffer_sampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glSamplerParameteri(_buffer_sampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        }
        bool Blur::bind(GLuint ssao_buffer_texture_id)
        {
            glUseProgram(_program);
            glBindFramebuffer(GL_FRAMEBUFFER, _fbo);
            glNamedFramebufferTexture(_fbo, GL_COLOR_ATTACHMENT0, _render_result_texture_id, 0);
            glClear(GL_COLOR_BUFFER_BIT);
            glDisable(GL_DEPTH_TEST);
            glViewport(0, 0, _viewport_width, _viewport_height);

            uniform_1f(3, _viewport_width);
            uniform_1f(4, _viewport_height);

            glBindTextureUnit(0, ssao_buffer_texture_id);
            glBindSampler(0, _buffer_sampler);

            check_framebuffer_status();
            return true;
        }
        void Blur::unbind()
        {
            glBindVertexArray(0);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glUseProgram(0);
            check_framebuffer_status();
            glEnable(GL_DEPTH_TEST);
        }
        GLuint Blur::get_buffer_texture_id()
        {
            return _render_result_texture_id;
        }
    }
}
}