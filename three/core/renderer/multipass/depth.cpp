#include "depth.h"
#include "../opengl/functions.h"
#include <iostream>

namespace three {
namespace renderer {
    namespace multipass {
        Depth::Depth()
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
    float light_distance = length(frag_light_direction);
    float attenuation = clamp(1.0 / (1.0 + 0.1 * light_distance + 0.2 * light_distance * light_distance), 0.0f, 1.0f);
    frag_color = vec4(vec3(attenuation), 1.0);
}
)";

            _program = opengl::create_program(vertex_shader, fragment_shader);
        }

        void Depth::use()
        {
            glUseProgram(_program);
        }
        void Depth::uniform_matrix(GLuint location, const GLfloat* matrix)
        {
            glUniformMatrix4fv(location, 1, GL_FALSE, matrix);
        }
    }
}
}