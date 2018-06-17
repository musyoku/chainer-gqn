#include "renderer.h"
#include "opengl.h"
#include <iostream>

namespace three {
namespace renderer {
    Renderer::Renderer(scene::Scene* scene, int width, int height)
    {
        _width = width;
        _height = height;

        glfwSetErrorCallback([](int error, const char* description) {
            fprintf(stderr, "Error %d: %s\n", error, description);
        });
        if (!!glfwInit() == false) {
            throw std::runtime_error("Failed to initialize GLFW.");
        }
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_VISIBLE, GL_TRUE);
#if __APPLE__
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
        _window = glfwCreateWindow(width, height, "Renderer", NULL, NULL);
        glfwMakeContextCurrent(_window);
        glfwSwapInterval(1);
        gl3wInit();

        const GLchar vertex_shader[] = R"(
#version 410
in vec3 position;
in vec3 normal_vector;
uniform mat4 pvm_mat;
flat out float power;
void main(void)
{
    gl_Position = pvm_mat * vec4(position, 1.0);
    vec3 _normal_vector = pvm_mat * vec4(normal_vector, 1,0);
    vec4 light_direction = vec4(0.0, -1.0, -1.5, 1.0);
    power = clamp(dot(_normal_vector, -normalize(light_direction.xyz)), 0.0, 1.0);
}
)";

        const GLchar fragment_shader[] = R"(
#version 410
flat in float power;
out vec4 frag_color;
void main(){
    frag_color = vec4(power * vec3(1.0), 1.0);
}
)";

        _program = opengl::create_program(vertex_shader, fragment_shader);

        _attribute_position = glGetAttribLocation(_program, "position");
        _attribute_normal_vector = glGetAttribLocation(_program, "normal_vector");
        _uniform_pvm_mat = glGetUniformLocation(_program, "pvm_mat");
        _uniform_light_mat = glGetUniformLocation(_program, "light_mat");

        glGenVertexArrays(1, &_vao);
        glBindVertexArray(_vao);

        int num_objects = scene->_objects.size();
        _vbo_faces = std::make_unique<GLuint[]>(num_objects);
        glGenBuffers(num_objects, _vbo_faces.get());

        _vbo_vertices = std::make_unique<GLuint[]>(num_objects);
        glGenBuffers(num_objects, _vbo_vertices.get());

        for (int n = 0; n < num_objects; n++) {
            glBindBuffer(GL_ARRAY_BUFFER, _vbo_vertices[n]);
            glVertexAttribPointer(_attribute_position, 3, GL_FLOAT, GL_FALSE, 0, 0);
            glEnableVertexAttribArray(_attribute_position);
        }

        _vbo_normal_vectors = std::make_unique<GLuint[]>(num_objects);
        glGenBuffers(num_objects, _vbo_normal_vectors.get());
        for (int n = 0; n < num_objects; n++) {
            glBindBuffer(GL_ARRAY_BUFFER, _vbo_normal_vectors[n]);
            glVertexAttribPointer(_attribute_normal_vector, 3, GL_FLOAT, GL_FALSE, 0, 0);
            glEnableVertexAttribArray(_attribute_normal_vector);
        }

        glBindVertexArray(0);
    }
    Renderer::~Renderer()
    {
        glfwSetWindowShouldClose(_window, GL_TRUE);
        glfwDestroyWindow(_window);
        glfwTerminate();
    }
    void Renderer::render_depth_map(
        camera::PerspectiveCamera* camera,
        py::array_t<float, py::array::c_style> np_depth_map)
    {
        glEnable(GL_BLEND);
        glEnable(GL_CULL_FACE);
        glEnable(GL_DEPTH_TEST);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        int screen_width, screen_height;
        glfwGetFramebufferSize(_window, &screen_width, &screen_height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(0.9, 0.9, 0.9, 1.0);
        glViewport(0, 0, _width, _height);
        glfwSwapBuffers(_window);
    }
}
}