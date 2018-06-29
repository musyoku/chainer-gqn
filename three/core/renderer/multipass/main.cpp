#include "main.h"
#include "../opengl/functions.h"
#include <random>

namespace three {
namespace renderer {
    namespace multipass {
        Main::Main()
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
layout(binding = 0) uniform sampler2D depth_map;
layout(binding = 1) uniform sampler1D ssao_sampling_points;
in vec4 frag_object_color;
in vec3 frag_light_direction;
in vec3 frag_smooth_normal_vector;
in vec3 frag_face_normal_vector;
in vec4 frag_position;
in float frag_w;
out vec4 frag_color;

const float pi = 3.14159;  
const float distance_threshold = 0.08; // 距離が離れすぎている場合は無視
const float z_near = 0.01;
const float z_far = 100.0;

// 右手座標系に戻る
float compute_true_depth(float z)
{
    z -= 0.0005; // シャドウアクネ回避
    return ((z_far * z_near) / (z_far - z_near)) / (z_far / (z_near - z_far) + z);
}

float random(const vec2 p) {
    float v = fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453123);
    return (v - 0.5) * 2.0;
}


void main(){
    float fov = pi / 4.0;
    float fov_2 = fov / 2.0;
    float tan_fov_2 = tan(fov_2);
    vec2 texcoord = gl_FragCoord.xy / 640.0;
    vec3 center = vec3(texcoord, compute_true_depth(texture(depth_map, texcoord).x));

    vec3 face_normal = normalize(frag_face_normal_vector);
    float radius = max(0.02 / -(center.z * tan_fov_2), 1.0 / 640.0);


    // float xyz = 0.0;
    // for(int i = 0;i < 192;i++){
    //     vec2 sample_point = texture(ssao_sampling_points, float(i) / 192.0).xy;

    //     vec2 s1 = center.xy + radius * sample_point;
    //     vec3 p1 = vec3(s1, compute_true_depth(texture(depth_map, s1).x));
    //     xyz += -p1.z;
    // }
    // frag_color = vec4(vec3(xyz / 192.0 / 5.0), 1.0);
    // return;

    // vec2 s1 = center.xy + radius * vec2(0.0, 1.0);
    // vec3 p1 = vec3(s1, compute_true_depth(texture(depth_map, s1).x));

    // // if( p1.z < center.z - distance_threshold ) {
    // //     return;  
    // // }

    // vec2 s2 = center.xy - radius * vec2(0.0, 1.0);
    // vec3 p2 = vec3(s2, compute_true_depth(texture(depth_map, s2).x));
    
    // // if( p2.z < center.z - distance_threshold ){
    // //     return;  
    // // }

    // vec3 p1_n = p1 - center;
    // vec3 p2_n = p2 - center;

    // float dd = dot(normalize(p1_n), normalize(p2_n));

    // float z1 = texture(depth_map, texcoord).x;
    // float z2 = texture(depth_map, s1).x;

    // float rad1 = atan(length(p1.xy - center.xy) / (p1.z - center.z));
    // // frag_color = vec4(vec3((z1 * 0.5 + z2 * 0.5 - 0.996923) / (1.0 - 0.996923)), 1.0);
    // frag_color = vec4((normalize(p1_n) + 1.0) * 0.5, 1.0);
    // return;






    float d = 0.0;
    for(int i = 0;i < 192;i++){
        vec2 shift = texture(ssao_sampling_points, float(i) / 192.0).xy;

        vec2 s1 = center.xy + radius * shift;
        vec3 p1 = vec3(s1, compute_true_depth(texture(depth_map, s1).x));

        // if( p1.z > center.z + distance_threshold ) {
        //     continue;  
        // }

        vec2 s2 = center.xy - 2.0 * radius * shift;
        vec3 p2 = vec3(s2, compute_true_depth(texture(depth_map, s2).x));
        
        // if( p2.z > center.z + distance_threshold ){
        //     continue;  
        // } 

        vec3 t1 = p1 - center;
        vec3 t2 = p2 - center;
        

        // float ddd = dot(normalize(t1), normalize(t2));
        // float ddd = dot(normalize(t1), face_normal) + dot(normalize(t2), face_normal);
        // d += clamp(0.5 * ddd, 0.0, 1.0);


        // float rad1 = atan((center.z - p1.z) / length(p1.xy - center.xy));
        // float rad2 = atan((center.z - p2.z) / length(p2.xy - center.xy));
        // d += clamp((rad1 + rad2) / pi, 0.0, 1.0);
        
        float rad1 = atan(length(t1.xy) * frag_w / t1.z);
        float rad2 = atan(length(t2.xy) * frag_w / t2.z);
        if(rad1 < 0.0){
            rad1 += pi;
        }
        if(rad2 < 0.0){
            rad2 += pi;
        }
        // float rad1 = atan(t1.z / length(t1.xy));
        // float rad2 = atan(t2.z / length(t2.xy));
        rad1 = clamp(rad1 / pi, 0.0, 1.0);
        rad2 = clamp(rad2 / pi, 0.0, 1.0);

        float y = distance_threshold * distance_threshold;
        float k = 1.0 / (1.0 + 1.0 * (t1.z * t1.z + t2.z * t2.z) / y);
        float q = 1.0 - clamp((rad1 + rad2), 0.0, 1.0);
        d += k * q;

        

        // float rad1 = atan(length(p1 - center) / (center.z - p1.z));
        // float rad2 = atan(length(p2 - center) / (center.z - p2.z));
        // d += clamp((rad1 + rad2) / pi, 0.0, 1.0);
    }
    float ao = 1.0 - d / 192.0;
    frag_color = vec4(vec3(ao), 1.0);
    frag_color = vec4(1.0 - vec3(d / 192.0), 1.0);
    return;



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
    vec3 composite_color = ambient_color * 0.1 + diffuse_color
        + specular_color * 0.5;

    vec3 top = 0.5 * diffuse_color;
    vec3 bottom = 0.5 * ambient_color;
    vec3 screen = 1.0 - (1.0 - top) * (1.0 - bottom);

    frag_color = vec4(screen * ao + specular_color * 0.08, 1.0);
    // frag_color = vec4(vec3(ao), 1.0);

    // frag_color = vec4(vec3(texture(depth_map, texcoord).x - 0.996923) / (1.0 - 0.996923), 1.0);

    // frag_color = vec4(vec3(ao), 1.0);

    if(gl_FragCoord.x > 320){
        frag_color = vec4(screen + specular_color * 0.08, 1.0);
    }

}
)";

            _program = opengl::create_program(vertex_shader, fragment_shader);

            // depth
            glGenTextures(1, &_depth_texture);
            glBindTexture(GL_TEXTURE_2D, _depth_texture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 640, 640, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
            glBindTexture(GL_TEXTURE_2D, 0);

            glGenSamplers(1, &_depth_sampler);
            glSamplerParameteri(_depth_sampler, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
            glSamplerParameteri(_depth_sampler, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
            glSamplerParameteri(_depth_sampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glSamplerParameteri(_depth_sampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

            // SSAO
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

        void Main::use()
        {
            glUseProgram(_program);
        }
        void Main::uniform_matrix(GLuint location, const GLfloat* matrix)
        {
            glUniformMatrix4fv(location, 1, GL_FALSE, matrix);
        }
        void Main::uniform_float(GLuint location, const GLfloat value)
        {
            glUniform1f(location, value);
        }
        void Main::attach_depth_texture()
        {
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, _depth_texture, 0);
        }
        void Main::bind_textures()
        {
            // glActiveTexture(GL_TEXTURE0);
            glBindTextureUnit(0, _depth_texture);
            glBindSampler(0, _depth_sampler);
            // glActiveTexture(GL_TEXTURE0 + 1);
            // glBindTexture(GL_TEXTURE_1D, _ssao_texture);
            glBindTextureUnit(1, _ssao_texture);
            glBindSampler(1, _ssao_sampler);
        }
    }
}
}