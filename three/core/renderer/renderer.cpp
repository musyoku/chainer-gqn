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
        _prev_num_objects = -1;

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
        gl3wInit();

        const GLchar vertex_shader[] = R"(
#version 410
in vec3 position;
in vec3 normal_vector;
in vec3 vertex_normal_vector;
uniform float quadratic_attenuation;
uniform mat4 model_mat;
uniform mat4 view_mat;
uniform mat4 projection_mat;
uniform vec4 color;
out float power;
out vec4 object_color;
out vec3 light_direction;
out float qa;
out mat4 _camera_mat;
out vec4 face_direction;
out vec3 _normal_vector;
void main(void)
{
    mat4 camera_mat = view_mat * model_mat;
    vec4 model_position = camera_mat * vec4(position, 1.0f);
    gl_Position = projection_mat * model_position;
    face_direction = camera_mat * vec4(vertex_normal_vector, 1.0f);
    vec4 light_position = view_mat * vec4(0.0f, -2.0f, 0.0f, 1.0f);
    light_direction = model_position.xyz - light_position.xyz;
    _camera_mat = camera_mat;
    _normal_vector = vertex_normal_vector;
    // power = clamp(attenuation, 0.0f, 1.0f);
    object_color = color;
    qa = quadratic_attenuation;
}
)";

        const GLchar fragment_shader[] = R"(
#version 410
in float power;
in vec4 face_direction;
in vec4 object_color;
in vec3 _normal_vector;
in vec3 light_direction;
in float qa;
in mat4 _camera_mat;
out vec4 frag_color;
void main(){
    float light_distance = length(light_direction);
    float attenuation = clamp(1.0 / (qa * light_distance * light_distance), 0.0f, 1.0f);
    vec3 half = normalize(face_direction.xyz) - normalize(light_direction.xyz);
    float diffuse = clamp(dot(normalize(face_direction.xyz), -normalize(light_direction)), 0.0f, 1.0f);
    float power = pow(diffuse, 10.0);
    frag_color = vec4((attenuation) * object_color.xyz, 1.0);
}
)";

        _program = opengl::create_program(vertex_shader, fragment_shader);

        _attribute_position = glGetAttribLocation(_program, "position");
        _attribute_normal_vector = glGetAttribLocation(_program, "normal_vector");
        _attribute_vertex_normal_vector = glGetAttribLocation(_program, "vertex_normal_vector");
        _uniform_projection_mat = glGetUniformLocation(_program, "projection_mat");
        _uniform_view_mat = glGetUniformLocation(_program, "view_mat");
        _uniform_model_mat = glGetUniformLocation(_program, "model_mat");
        _uniform_color = glGetUniformLocation(_program, "color");
        _uniform_quadratic_attenuation = glGetUniformLocation(_program, "quadratic_attenuation");

        glGenRenderbuffers(1, &_render_buffer);

        set_scene(scene);
    }
    Renderer::~Renderer()
    {
        _delete_buffers();
        glfwDestroyWindow(_window);
        glfwTerminate();
    }
    void Renderer::_delete_buffers()
    {
        if (_prev_num_objects == -1) {
            return;
        }
        glDeleteVertexArrays(_prev_num_objects, _vao.get());
        glDeleteBuffers(_prev_num_objects, _vbo_faces.get());
        glDeleteBuffers(_prev_num_objects, _vbo_vertices.get());
        glDeleteBuffers(_prev_num_objects, _vbo_normal_vectors.get());
        glDeleteBuffers(_prev_num_objects, _vbo_vertex_normal_vectors.get());
        glBindRenderbuffer(GL_RENDERBUFFER, 0);
    }
    void Renderer::set_scene(scene::Scene* scene)
    {
        glfwMakeContextCurrent(_window);
        glUseProgram(_program);
        _delete_buffers();
        _scene = scene;

        int num_objects = scene->_objects.size();
        _prev_num_objects = num_objects;

        _vao = std::make_unique<GLuint[]>(num_objects);
        glGenVertexArrays(num_objects, _vao.get());

        _vbo_faces = std::make_unique<GLuint[]>(num_objects);
        glGenBuffers(num_objects, _vbo_faces.get());

        _vbo_vertices = std::make_unique<GLuint[]>(num_objects);
        glGenBuffers(num_objects, _vbo_vertices.get());

        _vbo_normal_vectors = std::make_unique<GLuint[]>(num_objects);
        glGenBuffers(num_objects, _vbo_normal_vectors.get());

        _vbo_vertex_normal_vectors = std::make_unique<GLuint[]>(num_objects);
        glGenBuffers(num_objects, _vbo_vertex_normal_vectors.get());

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

            glBindBuffer(GL_ARRAY_BUFFER, _vbo_vertex_normal_vectors[n]);
            glBufferData(GL_ARRAY_BUFFER, 3 * num_faces * sizeof(glm::vec3f), object->_face_vertex_normal_vectors.get(), GL_STATIC_DRAW);
            glVertexAttribPointer(_attribute_vertex_normal_vector, 3, GL_FLOAT, GL_FALSE, 0, 0);
            glEnableVertexAttribArray(_attribute_vertex_normal_vector);

            glBindVertexArray(0);
        }

        glBindVertexArray(0);
    }
    void Renderer::_render_objects(camera::PerspectiveCamera* camera)
    {
        // OpenGL commands are executed in global context (per thread).
        glfwMakeContextCurrent(_window);
        glViewport(0, 0, _width, _height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(0.0, 0.0, 0.0, 1.0);

        static float quadratic_attenuation = 0.2;

        glm::mat4& view_mat = camera->_view_matrix;
        glm::mat4& projection_mat = camera->_projection_matrix;
        std::vector<std::shared_ptr<scene::Object>>& objects = _scene->_objects;
        for (int object_index = 0; object_index < objects.size(); object_index++) {
            std::shared_ptr<scene::Object> object = objects[object_index];
            glBindVertexArray(_vao[object_index]);
            glm::mat4& model_mat = object->_model_matrix;
            glUniformMatrix4fv(_uniform_projection_mat, 1, GL_FALSE, glm::value_ptr(projection_mat));
            glUniformMatrix4fv(_uniform_view_mat, 1, GL_FALSE, glm::value_ptr(view_mat));
            glUniformMatrix4fv(_uniform_model_mat, 1, GL_FALSE, glm::value_ptr(model_mat));
            glUniform1f(_uniform_quadratic_attenuation, quadratic_attenuation);
            glUniform4fv(_uniform_color, 1, glm::value_ptr(object->_color));
            glDrawArrays(GL_TRIANGLES, 0, 3 * object->_num_faces);
        }
    }
    void Renderer::render_depth_map(
        camera::PerspectiveCamera* camera,
        py::array_t<GLfloat, py::array::c_style> np_depth_map)
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

        glBindRenderbuffer(GL_RENDERBUFFER, _render_buffer);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, _render_buffer);

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
        glViewport(0, 0, _width, _height);

        glBindRenderbuffer(GL_RENDERBUFFER, _render_buffer);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, _render_buffer);

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
    void Renderer::render_depth_map(scene::Scene* scene, camera::PerspectiveCamera* camera,
        py::array_t<GLfloat, py::array::c_style> np_depth_map)
    {
        set_scene(scene);
        render_depth_map(camera, np_depth_map);
    }
    void Renderer::render(scene::Scene* scene, camera::PerspectiveCamera* camera,
        py::array_t<GLuint, py::array::c_style> np_rgb_map)
    {
        set_scene(scene);
        render(camera, np_rgb_map);
    }
}
}