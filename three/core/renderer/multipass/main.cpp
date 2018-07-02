#include "main.h"
#include "../opengl/functions.h"
#include <random>

namespace three {
namespace renderer {
    namespace multipass {
        Main::Main(int viewport_width, int viewport_height)
        {
            _viewport_width = viewport_width;
            _viewport_height = viewport_height;

            const GLchar vertex_shader[] = R"(
#version 450
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 face_normal_vector;
layout(location = 2) in vec3 vertex_normal_vector;
layout(location = 3) in vec4 vertex_color;
layout(location = 0) uniform mat4 model_mat;
layout(location = 1) uniform mat4 view_mat;
layout(location = 2) uniform mat4 projection_mat;
layout(location = 3) uniform float smoothness;
out vec4 frag_object_color;
out vec3 frag_smooth_normal_vector;
out vec3 frag_face_normal_vector;
out vec3 frag_light_direction;
out vec4 frag_position;
out float frag_w;
void main(void)
{
    vec4 model_position = model_mat * vec4(position, 1.0f);
    gl_Position = projection_mat * view_mat * model_position;
    vec3 light_position = vec3(0.0f, 1.0f, 1.0f);
    frag_light_direction = light_position - model_position.xyz;
    frag_object_color = vertex_color;
    frag_smooth_normal_vector = smoothness * vertex_normal_vector 
        + (1.0 - smoothness) * face_normal_vector;
    frag_face_normal_vector = (view_mat * vec4(face_normal_vector, 1.0)).xyz;
    frag_position = view_mat * model_position;
    frag_w = gl_Position.w;
}
)";

            const GLchar fragment_shader[] = R"(
#version 450
layout(binding = 0) uniform sampler2D ssao_buffer;
in vec4 frag_object_color;
in vec3 frag_light_direction;
in vec3 frag_smooth_normal_vector;
in vec3 frag_face_normal_vector;
in vec4 frag_position;
in float frag_w;
out vec4 frag_color;

void main(){
    vec3 face_normal = normalize(frag_face_normal_vector);

    vec3 unit_smooth_normal_vector = normalize(frag_smooth_normal_vector);
    vec3 unit_light_direction = normalize(frag_light_direction);
    vec3 unit_eye_direction = normalize(-frag_position.xyz);
    vec3 unit_reflection = normalize(2.0 * (dot(unit_light_direction, unit_smooth_normal_vector)) * unit_smooth_normal_vector - unit_light_direction);
    float is_frontface = step(0.0, dot(unit_reflection, unit_smooth_normal_vector));
    float light_distance = length(frag_light_direction);
    float attenuation = clamp(1.0 / (1.0 + 0.1 * light_distance + 0.2 * light_distance * light_distance), 0.0f, 1.0f);
    vec3 eye_direction = -frag_position.xyz;
    float diffuse = dot(unit_smooth_normal_vector, unit_light_direction);
    float specular = clamp(dot(unit_reflection, unit_eye_direction), 0.0f, 1.0f) * is_frontface;
    specular = pow(specular, 2.0);
    // frag_color = vec4((attenuation) * object_color.xyz, 1.0);
    vec3 attenuation_color = attenuation * frag_object_color.rgb;
    vec3 diffuse_color = diffuse * frag_object_color.rgb;
    vec3 specular_color = vec3(specular);
    vec3 ambient_color = frag_object_color.rgb;
    frag_color = vec4(clamp(diffuse_color + ambient_color * 0.5, 0.0, 1.0), 1.0);
    frag_color = vec4(vec3(attenuation), 1.0);
    vec3 composite_color = ambient_color * 0.5 + diffuse_color * 0.5
        + specular_color * 0.08;

    // vec3 top = 0.5 * diffuse_color;
    // vec3 bottom = 0.5 * ambient_color;
    // vec3 screen = 1.0 - (1.0 - top) * (1.0 - bottom);

    frag_color = vec4(composite_color, 1.0);
    // frag_color = vec4(screen * ao + specular_color * 0.08, 1.0);
    // frag_color = vec4(vec3(ao), 1.0);


    // frag_color = vec4(vec3(ao), 1.0);

    vec2 texcoord = gl_FragCoord.xy / 640.0;
    float ssao_luminance = texture(ssao_buffer, texcoord)[0];

    if(gl_FragCoord.x > 320){
        frag_color = vec4(composite_color * ssao_luminance, 1.0);
    }

    // vec2 texcoord = gl_FragCoord.xy / 640.0;
    // frag_color = vec4(vec3(ssao_luminance), 1.0);
}
)";

            _program = opengl::create_program(vertex_shader, fragment_shader);

            glCreateFramebuffers(1, &_fbo);

            glCreateRenderbuffers(1, &_color_render_buffer);
            glNamedRenderbufferStorage(_color_render_buffer, GL_RGB, viewport_width, viewport_height);

            glCreateRenderbuffers(1, &_depth_render_buffer);
            glNamedRenderbufferStorage(_depth_render_buffer, GL_DEPTH_COMPONENT, viewport_width, viewport_height);

            glGenSamplers(1, &_ssao_buffer_sampler);
            glSamplerParameteri(_ssao_buffer_sampler, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
            glSamplerParameteri(_ssao_buffer_sampler, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
            glSamplerParameteri(_ssao_buffer_sampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glSamplerParameteri(_ssao_buffer_sampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        }
        bool Main::bind(GLuint ssao_buffer_texture_id)
        {
            glUseProgram(_program);
            glBindFramebuffer(GL_FRAMEBUFFER, _fbo);
            glNamedFramebufferRenderbuffer(_fbo, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, _color_render_buffer);
            glNamedFramebufferRenderbuffer(_fbo, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, _depth_render_buffer);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glViewport(0, 0, _viewport_width, _viewport_height);

            glBindTextureUnit(0, ssao_buffer_texture_id);
            glBindSampler(0, _ssao_buffer_sampler);

            check_framebuffer_status();
            return true;
        }
        void Main::unbind()
        {
            glBindVertexArray(0);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glUseProgram(0);
            check_framebuffer_status();
        }
        void Main::set_additional_uniform_variables(base::Object* object)
        {
            float smoothness = object->_smoothness ? 1.0 : 0.0;
            uniform_1f(3, smoothness);
        }
    }
}
}