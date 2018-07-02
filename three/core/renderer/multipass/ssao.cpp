#include "ssao.h"
#include "../opengl/functions.h"
#include <random>

namespace three {
namespace renderer {
    namespace multipass {
        ScreenSpaceAmbientOcculusion::ScreenSpaceAmbientOcculusion(int viewport_width, int viewport_height, int total_sampling_points)
        {
            _viewport_width = viewport_width;
            _viewport_height = viewport_height;
            _total_sampling_points = total_sampling_points;

            const GLchar vertex_shader[] = R"(
#version 450
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 face_normal_vector;
layout(location = 2) in vec3 vertex_normal_vector;
layout(location = 3) in vec4 vertex_color;
layout(location = 0) uniform mat4 model_mat;
layout(location = 1) uniform mat4 view_mat;
layout(location = 2) uniform mat4 projection_mat;
layout(location = 3) uniform int num_sampling_points;
layout(location = 4) uniform float z_near;
layout(location = 5) uniform float z_far;
layout(location = 6) uniform float sampling_radius;
layout(location = 7) uniform float screen_width;
layout(location = 8) uniform float screen_height;
out float frag_w;
flat out int frag_num_sampling_points;
out float frag_z_near;
out float frag_z_far;
out float frag_sampling_radius;
out float frag_screen_width;
out float frag_screen_height;
void main(void)
{
    gl_Position = projection_mat * view_mat * model_mat * vec4(position, 1.0f);
    frag_w = gl_Position.w;
    frag_num_sampling_points = num_sampling_points;
    frag_z_near = z_near;
    frag_z_far = z_far;
    frag_sampling_radius = sampling_radius;
    frag_screen_width = screen_width;
    frag_screen_height = screen_height;
}
)";

            const GLchar fragment_shader[] = R"(
#version 450
layout(binding = 0) uniform sampler2D depth_buffer;
layout(binding = 3) uniform sampler1D ssao_sampling_points;
in float frag_w;
flat in int frag_num_sampling_points;
in float frag_z_near;
in float frag_z_far;
in float frag_sampling_radius;
in float frag_screen_width;
in float frag_screen_height;
layout(location = 0) out vec4 frag_color;

const float pi = 3.14159;  
const float distance_threshold = 0.001; // 距離が離れすぎている場合は無視

// 右手座標系に戻る
float compute_true_depth(float z)
{
    z -= 0.0005; // シャドウアクネ回避
    return ((frag_z_far * frag_z_near) / (frag_z_far - frag_z_near)) / (frag_z_far / (frag_z_near - frag_z_far) + z);
}

void main(){
    float fov = pi / 4.0;
    float fov_2 = fov / 2.0;
    float tan_fov_2 = tan(fov_2);
    vec2 texcoord = gl_FragCoord.xy / vec2(frag_screen_width, frag_screen_height);
    vec3 center = vec3(texcoord, compute_true_depth(texture(depth_buffer, texcoord).x));

    float radius = frag_sampling_radius / -(center.z * tan_fov_2);

    float occlusion = 0.0;
    int loop = int(frag_num_sampling_points);
    for(int i = 0;i < loop;i++){
        vec2 shift = texture(ssao_sampling_points, float(i) / frag_num_sampling_points).xy;

        vec2 s1 = center.xy + radius * shift;
        vec3 p1 = vec3(s1, compute_true_depth(texture(depth_buffer, s1).x));

        vec2 s2 = center.xy - 2.0 * radius * shift;
        vec3 p2 = vec3(s2, compute_true_depth(texture(depth_buffer, s2).x));
        
        vec3 t1 = p1 - center;
        vec3 t2 = p2 - center;
        
        float rad1 = atan(length(t1.xy) * frag_w / t1.z);
        float rad2 = atan(length(t2.xy) * frag_w / t2.z);
        if(rad1 < 0.0){
            rad1 += pi;
        }
        if(rad2 < 0.0){
            rad2 += pi;
        }
        rad1 = clamp(rad1 / pi, 0.0, 1.0);
        rad2 = clamp(rad2 / pi, 0.0, 1.0);

        float y = distance_threshold * distance_threshold;
        float u1 = max(t1.z - distance_threshold, 0.0);
        float u2 = max(t2.z - distance_threshold, 0.0);
        float k = 1.0 / (1.0 + 10.0 * (u1 * u1 + u2 * u2));
        float q = 1.0 - clamp((rad1 + rad2), 0.0, 1.0);
        occlusion += k * q;
    }
    float luminance = 1.0 - occlusion / frag_num_sampling_points;
    frag_color = vec4(vec3(luminance), 1.0);
}
)";

            _program = opengl::create_program(vertex_shader, fragment_shader);
            glCreateFramebuffers(1, &_fbo);

            glActiveTexture(GL_TEXTURE0 + 1);
            glGenTextures(1, &_render_result_texture_id);
            glBindTexture(GL_TEXTURE_2D, _render_result_texture_id);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, viewport_width, viewport_height, 0, GL_RGB, GL_FLOAT, 0);
            glBindTexture(GL_TEXTURE_2D, 0);
            glActiveTexture(GL_TEXTURE0);

            glActiveTexture(GL_TEXTURE0 + 2);
            glGenTextures(1, &_sampling_points_texture_id);
            std::random_device seed;
            std::default_random_engine engine(seed());
            std::normal_distribution<> normal_dist(0.0, 0.8);
            _sampling_points_data = std::make_unique<GLfloat[]>(total_sampling_points * 2);
            for (int i = 0; i < total_sampling_points; i++) {
                _sampling_points_data[i * 2 + 0] = normal_dist(engine);
                _sampling_points_data[i * 2 + 1] = normal_dist(engine);
            }
            glBindTexture(GL_TEXTURE_1D, _sampling_points_texture_id);
            glTexImage1D(GL_TEXTURE_1D, 0, GL_RG32F, total_sampling_points, 0, GL_RG, GL_FLOAT, _sampling_points_data.get());

            glTextureParameteri(_sampling_points_texture_id, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTextureParameteri(_sampling_points_texture_id, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTextureParameteri(_sampling_points_texture_id, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
            glTextureParameteri(_sampling_points_texture_id, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
            glBindTexture(GL_TEXTURE_1D, 0);
            glActiveTexture(GL_TEXTURE0);

            glGenSamplers(1, &_sampling_points_sampler_id);
            glSamplerParameteri(_sampling_points_sampler_id, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
            glSamplerParameteri(_sampling_points_sampler_id, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
            glSamplerParameteri(_sampling_points_sampler_id, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glSamplerParameteri(_sampling_points_sampler_id, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        }
        bool ScreenSpaceAmbientOcculusion::bind()
        {
            return bind(_total_sampling_points);
        }
        bool ScreenSpaceAmbientOcculusion::bind(int num_sampling_points)
        {
            if(num_sampling_points > _total_sampling_points){
                throw std::runtime_error("num_sampling_points > _total_sampling_points");
            }
            glUseProgram(_program);
            glBindFramebuffer(GL_FRAMEBUFFER, _fbo);
            glNamedFramebufferTexture(_fbo, GL_COLOR_ATTACHMENT0, _render_result_texture_id, 0);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glViewport(0, 0, _viewport_width, _viewport_height);
            check_framebuffer_status();
            uniform_1i(3, num_sampling_points);
            uniform_1f(4, 0.01);
            uniform_1f(5, 100.0);
            uniform_1f(6, 0.02);
            uniform_1f(7, _viewport_width);
            uniform_1f(8, _viewport_height);
            glBindTextureUnit(3, _sampling_points_texture_id);
            glBindSampler(3, _sampling_points_sampler_id);
            return true;
        }
        void ScreenSpaceAmbientOcculusion::unbind()
        {
            glBindVertexArray(0);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glUseProgram(0);
            check_framebuffer_status();
        }
        void ScreenSpaceAmbientOcculusion::bind_ssao_buffer()
        {
            glBindTextureUnit(1, _render_result_texture_id);
        }
    }
}
}