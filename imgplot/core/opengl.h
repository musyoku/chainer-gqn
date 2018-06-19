#pragma once
#include <gl3w/gl3w.h>

namespace imgplot {
namespace opengl {
    GLboolean print_shader_info_log(GLuint shader, const char* str);
    GLuint create_program(const char* vertex_shader, const char* fragment_shader);
    GLboolean print_program_info_log(GLuint program);
}
}