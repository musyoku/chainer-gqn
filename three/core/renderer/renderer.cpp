#include "renderer.h"
#include "opengl.h"
#include <glm/gtc/type_ptr.hpp>

namespace three {
namespace renderer {
    void Renderer::initialize(int width, int height)
    {
        _width = width;
        _height = height;
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
#version 450
in vec3 position;
in vec3 face_normal_vector;
in vec3 vertex_normal_vector;
in vec4 vertex_color;
uniform float quadratic_attenuation;
uniform mat4 model_mat;
uniform mat4 view_mat;
uniform mat4 projection_mat;
uniform float smoothness;
out float power;
out float qa;
out vec4 frag_object_color;
out vec3 frag_smooth_normal_vector;
out vec3 frag_vertex_normal_vector;
out vec3 frag_light_direction;
out vec4 frag_position;
void main(void)
{
    vec4 model_position = model_mat * vec4(position, 1.0f);
    gl_Position = projection_mat * view_mat * model_position;
    vec3 light_position = vec3(0.0f, 1.0f, 1.0f);
    frag_light_direction = light_position - model_position.xyz;
    // power = clamp(attenuation, 0.0f, 1.0f);
    frag_object_color = vertex_color;
    qa = quadratic_attenuation;
    frag_smooth_normal_vector = smoothness * vertex_normal_vector 
        + (1.0 - smoothness) * face_normal_vector;
    frag_vertex_normal_vector = vertex_normal_vector;
    frag_position = view_mat * model_position;
}
)";

        const GLchar fragment_shader[] = R"(
#version 450
in float power;
in float qa;
in vec4 frag_object_color;
in vec3 frag_light_direction;
in vec3 frag_smooth_normal_vector;
in vec3 frag_vertex_normal_vector;
in vec4 frag_position;
out vec4 frag_color;
void main(){
    vec3 unit_smooth_normal_vector = normalize(frag_smooth_normal_vector);
    vec3 unit_light_direction = normalize(frag_light_direction);
    vec3 unit_eye_direction = normalize(-frag_position.xyz);
    vec3 unit_reflection = normalize(2.0 * (dot(unit_light_direction, unit_smooth_normal_vector)) * unit_smooth_normal_vector - unit_light_direction);
    float is_frontface = step(0.0, dot(unit_reflection, unit_smooth_normal_vector));
    float light_distance = length(frag_light_direction);
    float attenuation = clamp(
        1.0 / (1.0 + 0.1 * light_distance + 0.2 * light_distance * light_distance), 0.0f, 1.0f);
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

    // top = 0.5 * specular_color;
    // bottom = screen;
    // screen = 1.0 - (1.0 - top) * (1.0 - bottom);
    frag_color = vec4(screen + specular_color * 0.1, 1.0);

    if(gl_FragCoord.x > 320){
        // frag_color = vec4(specular_color, 1.0);
    }

    
    // frag_color = vec4((unit_smooth_normal_vector + 1.0) * 0.5, 1.0);
    // frag_color = vec4(vec3(attenuation), 1.0);
}
)";

        _program = opengl::create_program(vertex_shader, fragment_shader);

        _attribute_position = glGetAttribLocation(_program, "position");
        _attribute_face_normal_vector = glGetAttribLocation(_program, "face_normal_vector");
        _attribute_vertex_normal_vector = glGetAttribLocation(_program, "vertex_normal_vector");
        _attribute_vertex_color = glGetAttribLocation(_program, "vertex_color");
        _uniform_projection_mat = glGetUniformLocation(_program, "projection_mat");
        _uniform_view_mat = glGetUniformLocation(_program, "view_mat");
        _uniform_model_mat = glGetUniformLocation(_program, "model_mat");
        _uniform_quadratic_attenuation = glGetUniformLocation(_program, "quadratic_attenuation");
        _uniform_smoothness = glGetUniformLocation(_program, "smoothness");

        glGenRenderbuffers(1, &_render_buffer);
    }
    Renderer::Renderer(int width, int height)
    {
        initialize(width, height);
    }
    Renderer::Renderer(scene::Scene* scene, int width, int height)
    {
        initialize(width, height);
        set_scene(scene);
    }
    Renderer::~Renderer()
    {
        delete_buffers();
        glfwDestroyWindow(_window);
        glfwTerminate();
    }
    void Renderer::delete_buffers()
    {
        if (_prev_num_objects == -1) {
            return;
        }
        glDeleteVertexArrays(_prev_num_objects, _vao.get());
        glDeleteBuffers(_prev_num_objects, _vbo_faces.get());
        glDeleteBuffers(_prev_num_objects, _vbo_vertices.get());
        glDeleteBuffers(_prev_num_objects, _vbo_normal_vectors.get());
        glDeleteBuffers(_prev_num_objects, _vbo_vertex_normal_vectors.get());
        glDeleteBuffers(_prev_num_objects, _vbo_vertex_colors.get());
        glBindRenderbuffer(GL_RENDERBUFFER, 0);
    }
    void Renderer::set_scene(scene::Scene* scene)
    {
        glfwMakeContextCurrent(_window);
        glUseProgram(_program);
        delete_buffers();
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

        _vbo_vertex_colors = std::make_unique<GLuint[]>(num_objects);
        glGenBuffers(num_objects, _vbo_vertex_colors.get());

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
            glVertexAttribPointer(_attribute_face_normal_vector, 3, GL_FLOAT, GL_FALSE, 0, 0);
            glEnableVertexAttribArray(_attribute_face_normal_vector);

            glBindBuffer(GL_ARRAY_BUFFER, _vbo_vertex_normal_vectors[n]);
            glBufferData(GL_ARRAY_BUFFER, 3 * num_faces * sizeof(glm::vec3f), object->_face_vertex_normal_vectors.get(), GL_STATIC_DRAW);
            glVertexAttribPointer(_attribute_vertex_normal_vector, 3, GL_FLOAT, GL_FALSE, 0, 0);
            glEnableVertexAttribArray(_attribute_vertex_normal_vector);

            glBindBuffer(GL_ARRAY_BUFFER, _vbo_vertex_colors[n]);
            glBufferData(GL_ARRAY_BUFFER, 3 * num_faces * sizeof(glm::vec4f), object->_face_vertex_colors.get(), GL_STATIC_DRAW);
            glVertexAttribPointer(_attribute_vertex_color, 4, GL_FLOAT, GL_FALSE, 0, 0);
            glEnableVertexAttribArray(_attribute_vertex_color);

            glBindVertexArray(0);
        }

        glBindVertexArray(0);
    }
    void Renderer::render_objects(camera::PerspectiveCamera* camera)
    {
        // OpenGL commands are executed in global context (per thread).
        glfwMakeContextCurrent(_window);
        glViewport(0, 0, _width, _height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(0.0, 0.0, 0.0, 1.0);

        static float quadratic_attenuation = 2.0;

        glm::mat4& view_mat = camera->_view_matrix;
        glm::mat4& projection_mat = camera->_projection_matrix;
        std::vector<std::shared_ptr<base::Object>>& objects = _scene->_objects;
        for (int object_index = 0; object_index < objects.size(); object_index++) {
            std::shared_ptr<base::Object> object = objects[object_index];
            glBindVertexArray(_vao[object_index]);
            glm::mat4& model_mat = object->_model_matrix;
            glUniformMatrix4fv(_uniform_projection_mat, 1, GL_FALSE, glm::value_ptr(projection_mat));
            glUniformMatrix4fv(_uniform_view_mat, 1, GL_FALSE, glm::value_ptr(view_mat));
            glUniformMatrix4fv(_uniform_model_mat, 1, GL_FALSE, glm::value_ptr(model_mat));
            glUniform1f(_uniform_quadratic_attenuation, quadratic_attenuation);
            float smoothness = object->_smoothness ? 1.0 : 0.0;
            glUniform1f(_uniform_smoothness, smoothness);
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

        render_objects(camera);

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

        render_objects(camera);

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