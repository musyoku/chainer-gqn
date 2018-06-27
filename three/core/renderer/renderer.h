#pragma once
#include "../base/object.h"
#include "../camera/perspective.h"
#include "../scene/scene.h"
#include "multipass/depth.h"
#include "multipass/main.h"
#include "opengl/vao.h"
#include <gl3w/gl3w.h>
#include <glfw/glfw3.h>
#include <memory>
#include <pybind11/numpy.h>

namespace three {
namespace renderer {
    namespace py = pybind11;
    class Renderer {
    private:
        int _width;
        int _height;
        GLuint _frame_buffer;
        GLuint _color_render_buffer;
        GLuint _depth_render_buffer;
        GLuint _depth_texture;
        GLuint _color_texture;
        const GLuint _uniform_model_mat = 0;
        const GLuint _uniform_view_mat = 1;
        const GLuint _uniform_projection_mat = 2;
        const GLuint _uniform_soothness = 3;
        std::unique_ptr<opengl::VertexArrayObject> _vao;
        std::unique_ptr<multipass::Depth> _depth_program;
        std::unique_ptr<multipass::Main> _main_program;
        std::unique_ptr<GLubyte[]> _color_pixels;
        std::unique_ptr<GLfloat[]> _depth_pixels;
        std::unique_ptr<GLfloat[]> _depth_texture_data;
        GLFWwindow* _window;
        scene::Scene* _scene;
        int _prev_num_objects;
        void draw_objects(camera::PerspectiveCamera* camera);
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