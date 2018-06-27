#pragma once
#include "../base/object.h"
#include "../camera/perspective.h"
#include "../scene/scene.h"
#include <gl3w/gl3w.h>
#include <glfw/glfw3.h>
#include <memory>
#include <pybind11/numpy.h>

namespace three {
namespace renderer {
    namespace multipass {
        namespace py = pybind11;
        class Depth {
        private:
            int _width;
            int _height;
            GLuint _program;
            GLuint _attribute_position;
            GLuint _attribute_face_normal_vector;
            GLuint _attribute_vertex_normal_vector;
            GLuint _attribute_vertex_color;
            GLuint _uniform_projection_mat;
            GLuint _uniform_model_mat;
            GLuint _uniform_view_mat;
            GLuint _render_buffer;
            GLuint _texture_depth_map;
            std::unique_ptr<GLuint[]> _vao;
            std::unique_ptr<GLuint[]> _vbo_vertices;
            std::unique_ptr<GLuint[]> _vbo_normal_vectors;
            std::unique_ptr<GLuint[]> _vbo_vertex_normal_vectors;
            std::unique_ptr<GLuint[]> _vbo_vertex_colors;
            std::unique_ptr<GLuint[]> _vbo_faces;
            std::unique_ptr<GLubyte[]> _color_buffer;
            std::unique_ptr<GLfloat[]> _depth_buffer;
            GLFWwindow* _window;
            scene::Scene* _scene;
            int _prev_num_objects;
            void render_objects(camera::PerspectiveCamera* camera);
            void delete_buffers();
            void initialize(int width, int height);

        public:
            Renderer(int width, int height);
            Renderer(scene::Scene* scene, int width, int height);
            ~Renderer();
            void set_scene(scene::Scene* scene);
            void render_depth_map(camera::PerspectiveCamera* camera,
                py::array_t<GLfloat, py::array::c_style> np_depth_map);
            void render_depth_map(scene::Scene* scene, camera::PerspectiveCamera* camera,
                py::array_t<GLfloat, py::array::c_style> np_depth_map);
            void render(camera::PerspectiveCamera* camera,
                py::array_t<GLuint, py::array::c_style> np_rgb_map);
            void render(scene::Scene* scene, camera::PerspectiveCamera* camera,
                py::array_t<GLuint, py::array::c_style> np_rgb_map);
        };
    }
}
}