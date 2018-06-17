#include "renderer.h"
#include "opengl.h"
#include <glm/gtc/type_ptr.hpp>
#include <iostream>

namespace three {
namespace renderer {
    Renderer::Renderer(scene::Scene* scene, int width, int height)
    {
        _width = width;
        _height = height;
        _scene = scene;

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
    gl_Position = pvm_mat * vec4(position, 1.0f);
    vec4 _normal_vector = pvm_mat * vec4(normal_vector, 1.0f);
    vec4 light_direction = vec4(0.0f, -1.0f, -1.5f, 1.0f);
    power = clamp(dot(_normal_vector.xyz, -normalize(light_direction.xyz)), 0.0f, 1.0f);
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

        int num_objects = scene->_objects.size();

        _vao = std::make_unique<GLuint[]>(num_objects);
        glGenVertexArrays(num_objects, _vao.get());

        _vbo_faces = std::make_unique<GLuint[]>(num_objects);
        glGenBuffers(num_objects, _vbo_faces.get());

        _vbo_vertices = std::make_unique<GLuint[]>(num_objects);
        glGenBuffers(num_objects, _vbo_vertices.get());

        _vbo_normal_vectors = std::make_unique<GLuint[]>(num_objects);
        glGenBuffers(num_objects, _vbo_normal_vectors.get());

        glGenRenderbuffers(1, &_depth_buffer);
        glBindRenderbuffer(GL_RENDERBUFFER, _depth_buffer);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, _depth_buffer);

        for (int n = 0; n < num_objects; n++) {
            glBindVertexArray(_vao[n]);

            auto& object = scene->_objects[n];
            int num_faces = object->_num_faces;
            int num_vertices = object->_num_vertices;

            glBindBuffer(GL_ARRAY_BUFFER, _vbo_vertices[n]);
            glBufferData(GL_ARRAY_BUFFER, 3 * num_faces * sizeof(glm::vec3f), object->_face_vertices.get(), GL_STATIC_DRAW);
            glVertexAttribPointer(_attribute_position, 3, GL_FLOAT, GL_FALSE, 0, 0);
            glEnableVertexAttribArray(_attribute_position);

            glBindBuffer(GL_ARRAY_BUFFER, _vbo_normal_vectors[n]);
            glBufferData(GL_ARRAY_BUFFER, num_faces * sizeof(glm::vec3f), object->_face_normal_vectors.get(), GL_STATIC_DRAW);
            glVertexAttribPointer(_attribute_normal_vector, 3, GL_FLOAT, GL_FALSE, 0, 0);
            glEnableVertexAttribArray(_attribute_normal_vector);

            glBindVertexArray(0);
        }

        glBindVertexArray(0);
    }
    Renderer::~Renderer()
    {
        glfwDestroyWindow(_window);
        glfwTerminate();
    }
    void Renderer::render_depth_map(
        camera::PerspectiveCamera* camera,
        py::array_t<float, py::array::c_style> np_depth_map)
    {
        if (glfwWindowShouldClose(_window)) {
            glfwDestroyWindow(_window);
            glfwTerminate();
            return;
        }

        glUseProgram(_program);
        glEnable(GL_BLEND);
        glEnable(GL_CULL_FACE);
        glEnable(GL_DEPTH_TEST);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glViewport(0, 0, _width, _height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(0.0, 0.0, 0.0, 1.0);

        glm::mat4& view_mat = camera->_view_matrix;
        glm::mat4& projection_mat = camera->_projection_matrix;
        std::vector<std::shared_ptr<scene::Object>>& objects = _scene->_objects;
        for (int object_index = 0; object_index < objects.size(); object_index++) {
            glBindVertexArray(_vao[object_index]);
            std::shared_ptr<scene::Object> object = objects[object_index];
            glm::mat4& model_mat = object->_model_matrix;
            glm::mat4 pvm_mat = projection_mat * view_mat * model_mat;
            glUniformMatrix4fv(_uniform_pvm_mat, 1, GL_FALSE, glm::value_ptr(pvm_mat));
            glDrawArrays(GL_TRIANGLES, 0, 3 * object->_num_faces);
        }

        GLfloat* depths;
        depths = new GLfloat[_width * _height];
        glReadPixels(0, 0, _width, _height, GL_DEPTH_COMPONENT, GL_FLOAT, depths);

        auto depth_map = np_depth_map.mutable_unchecked<2>();
        for (int h = 0; h < _height; h++) {
            for (int w = 0; w < _width; w++) {
                depth_map(h, w) = depths[(_height - h - 1) * _width + w];
            }
        }
        delete[] depths;

        glUseProgram(0);
        glBindVertexArray(0);
        glfwSwapBuffers(_window);
    }
}
}