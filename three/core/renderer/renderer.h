#pragma once
#include "../camera/perspective.h"
#include "../scene/scene.h"
#include <gl3w/gl3w.h>
#include <glfw/glfw3.h>
#include <pybind11/numpy.h>
#include <memory>

namespace three {
namespace renderer {
    namespace py = pybind11;
    class Renderer {
    private:
        int _width;
        int _height;
        GLuint _program;
        GLuint _attribute_position;
        GLuint _attribute_normal_vector;
        GLuint _uniform_pvm_mat;
        GLuint _uniform_light_mat;
        GLuint _vao;
        std::unique_ptr<GLuint[]> _vbo_vertices;
        std::unique_ptr<GLuint[]> _vbo_normal_vectors;
        std::unique_ptr<GLuint[]> _vbo_faces;
        GLFWwindow* _window;

    public:
        Renderer(scene::Scene* scene, int width, int height);
        ~Renderer();
        void render_depth_map(camera::PerspectiveCamera* camera,
            py::array_t<float, py::array::c_style> np_depth_map);
    };
}
}