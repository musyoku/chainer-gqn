#include "depth.h"
#include "../opengl/functions.h"

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
void main(void)
{
    gl_Position = projection_mat * view_mat * model_mat * vec4(position, 1.0f);
}
)";

            const GLchar fragment_shader[] = R"(
#version 450
out vec4 frag_color;
void main(){
    frag_color = vec4(1.0);
}
)";

            _program = opengl::create_program(vertex_shader, fragment_shader);
        }

        void Depth::use()
        {
            glUseProgram(_program);
        }
    }
}
}