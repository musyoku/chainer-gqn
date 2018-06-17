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
        _depth_buffer = std::make_unique<GLfloat[]>(width * height);
        _color_buffer = std::make_unique<GLubyte[]>(width * height * 3);

        glfwSetErrorCallback([](int error, const char* description) {
            fprintf(stderr, "Error %d: %s\n", error, description);
        });
        if (!!glfwInit() == false) {
            throw std::runtime_error("Failed to initialize GLFW.");
        }
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_VISIBLE, GL_FALSE);
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
uniform mat4 camera_mat;
uniform mat4 projection_mat;
uniform vec4 color;
flat out float power;
out vec4 object_color;
void main(void)
{
    gl_Position = projection_mat * camera_mat * vec4(position, 1.0f);
    vec4 face_direction = camera_mat * vec4(normal_vector, 0.0f);
    vec4 light_direction = vec4(0.0, -1.0, -1.0f, 0.0f);
    power = clamp(dot(normalize(face_direction.xyz), -normalize(light_direction.xyz)), 0.0f, 1.0f);
    object_color = color;
}
)";

        const GLchar fragment_shader[] = R"(
#version 410
flat in float power;
in vec4 object_color;
out vec4 frag_color;
void main(){
    frag_color = vec4(power * vec3(1.0), 1.0);
}
)";

        _program = opengl::create_program(vertex_shader, fragment_shader);

        _attribute_position = glGetAttribLocation(_program, "position");
        _attribute_normal_vector = glGetAttribLocation(_program, "normal_vector");
        _uniform_projection_mat = glGetUniformLocation(_program, "projection_mat");
        _uniform_camera_mat = glGetUniformLocation(_program, "camera_mat");
        _uniform_color = glGetUniformLocation(_program, "color");

        int num_objects = scene->_objects.size();

        _vao = std::make_unique<GLuint[]>(num_objects);
        glGenVertexArrays(num_objects, _vao.get());

        _vbo_faces = std::make_unique<GLuint[]>(num_objects);
        glGenBuffers(num_objects, _vbo_faces.get());

        _vbo_vertices = std::make_unique<GLuint[]>(num_objects);
        glGenBuffers(num_objects, _vbo_vertices.get());

        _vbo_normal_vectors = std::make_unique<GLuint[]>(num_objects);
        glGenBuffers(num_objects, _vbo_normal_vectors.get());

        glGenRenderbuffers(1, &_attachment_depth);
        glGenRenderbuffers(1, &_attachment_color);

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
            glBufferData(GL_ARRAY_BUFFER, 3 * num_faces * sizeof(glm::vec3f), object->_face_normal_vectors.get(), GL_STATIC_DRAW);
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
    void Renderer::_render_objects(camera::PerspectiveCamera* camera)
    {
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
            glm::mat4 camera_mat = view_mat * model_mat;
            glUniformMatrix4fv(_uniform_projection_mat, 1, GL_FALSE, glm::value_ptr(projection_mat));
            glUniformMatrix4fv(_uniform_camera_mat, 1, GL_FALSE, glm::value_ptr(camera_mat));
            glUniform4fv(_uniform_color, 1, glm::value_ptr(object->_color));
            glDrawArrays(GL_TRIANGLES, 0, 3 * object->_num_faces);
        }
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

        glBindRenderbuffer(GL_RENDERBUFFER, _attachment_depth);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, _attachment_depth);

        _render_objects(camera);

        glReadPixels(0, 0, _width, _height, GL_DEPTH_COMPONENT, GL_FLOAT, _depth_buffer.get());
        auto depth_map = np_depth_map.mutable_unchecked<2>();
        for (int h = 0; h < _height; h++) {
            for (int w = 0; w < _width; w++) {
                depth_map(h, w) = _depth_buffer[(_height - h - 1) * _width + w];
            }
        }

        glUseProgram(0);
        glBindVertexArray(0);
        glBindRenderbuffer(GL_RENDERBUFFER, 0);
    }
    void Renderer::render(
        camera::PerspectiveCamera* camera,
        py::array_t<GLuint, py::array::c_style> np_rgb_map)
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
        glPixelStorei(GL_PACK_ALIGNMENT, 1);

        glBindRenderbuffer(GL_RENDERBUFFER, _attachment_depth);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, _attachment_depth);

        _render_objects(camera);

        glReadPixels(0, 0, _width, _height, GL_RGB, GL_UNSIGNED_BYTE, _color_buffer.get());
        auto rgb_map = np_rgb_map.mutable_unchecked<3>();
        for (int h = 0; h < _height; h++) {
            for (int w = 0; w < _width; w++) {
                rgb_map(h, w, 0) = _color_buffer[(_height - h - 1) * _width * 3 + w * 3 + 0];
                rgb_map(h, w, 1) = _color_buffer[(_height - h - 1) * _width * 3 + w * 3 + 1];
                rgb_map(h, w, 2) = _color_buffer[(_height - h - 1) * _width * 3 + w * 3 + 2];
            }
        }

        glUseProgram(0);
        glBindVertexArray(0);
        glBindRenderbuffer(GL_RENDERBUFFER, 0);
    }
}
}