#include "ssao.h"
#include "../opengl/functions.h"
#include <random>

namespace three {
namespace renderer {
    namespace multipass {
        ScreenSpaceAmbientOcculusion::ScreenSpaceAmbientOcculusion()
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
uniform int num_sampling_points;
uniform float z_near;
uniform float z_far;
uniform float sampling_radius;
uniform float screen_width;
uniform float screen_height;
out float frag_w;
out int frag_num_sampling_points;
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
layout(binding = 0) uniform sampler2D depth_map;
layout(binding = 1) uniform sampler1D ssao_sampling_points;
in float frag_w;
in int frag_num_sampling_points;
in float frag_z_near;
in float frag_z_far;
in float frag_sampling_radius;
in float frag_screen_width;
in float frag_screen_height;
out vec4 frag_color;

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
    vec2 texcoord = gl_FragCoord.xy / vec2(screen_width, screen_height);
    vec3 center = vec3(texcoord, compute_true_depth(texture(depth_map, texcoord).x));

    float radius = frag_sampling_radius / -(center.z * tan_fov_2);

    float d = 0.0;
    for(int i = 0;i < 192;i++){
        vec2 shift = texture(ssao_sampling_points, float(i) / 192.0).xy;

        vec2 s1 = center.xy + radius * shift;
        vec3 p1 = vec3(s1, compute_true_depth(texture(depth_map, s1).x));

        vec2 s2 = center.xy - 2.0 * radius * shift;
        vec3 p2 = vec3(s2, compute_true_depth(texture(depth_map, s2).x));
        
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
        d += k * q;
    }
    float ao = 1.0 - d / 192.0;
    frag_color = vec4(vec3(ao), 1.0);
}
)";

            _program = opengl::create_program(vertex_shader, fragment_shader);

            // ScreenSpaceAmbientOcculusion
            glGenTextures(1, &_ssao_texture);
            std::random_device seed;
            std::default_random_engine engine(seed());
            std::normal_distribution<> normal_dist(0.0, 0.8);
            _ssao_texture_data = std::make_unique<GLfloat[]>(192 * 2);
            for (int i = 0; i < 192; i++) {
                _ssao_texture_data[i * 2 + 0] = normal_dist(engine);
                _ssao_texture_data[i * 2 + 1] = normal_dist(engine);
            }
            glActiveTexture(GL_TEXTURE0 + 1);
            glBindTexture(GL_TEXTURE_1D, _ssao_texture);
            glTexImage1D(GL_TEXTURE_1D, 0, GL_RG32F, 192, 0, GL_RG, GL_FLOAT, _ssao_texture_data.get());

            glTextureParameteri(_ssao_texture, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTextureParameteri(_ssao_texture, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTextureParameteri(_ssao_texture, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
            glTextureParameteri(_ssao_texture, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
            glBindTexture(GL_TEXTURE_1D, 0);

            glGenSamplers(1, &_ssao_sampler);
            glSamplerParameteri(_ssao_sampler, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
            glSamplerParameteri(_ssao_sampler, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
            glSamplerParameteri(_ssao_sampler, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glSamplerParameteri(_ssao_sampler, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        }
        bool ScreenSpaceAmbientOcculusion::bind(int num_sampling_points)
        {
            return true;
        }
        void ScreenSpaceAmbientOcculusion::unbind()
        {
        }
    }
}
}