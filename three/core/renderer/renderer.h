#pragma once
#include "../base/object.h"
#include "../camera/perspective.h"
#include "../scene/scene.h"
#include "multipass/depth.h"
#include "multipass/main.h"
#include "multipass/ssao.h"
#include "opengl/vao.h"
#include "pass.h"
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
        const GLuint _uniform_model_mat = 0;
        const GLuint _uniform_view_mat = 1;
        const GLuint _uniform_projection_mat = 2;
        const GLuint _uniform_soothness = 3;
        std::unique_ptr<opengl::VertexArrayObject> _vao;
        std::unique_ptr<multipass::DepthBuffer> _depth_render_pass;
        std::unique_ptr<multipass::Main> _main_render_pass;
        std::unique_ptr<GLubyte[]> _color_pixels;
        std::unique_ptr<GLfloat[]> _depth_pixels;
        GLFWwindow* _window;
        scene::Scene* _scene;
        int _prev_num_objects;
        void draw_objects(camera::PerspectiveCamera* camera, RenderPass* pass);
        void initialize(int width, int height);

    public:
        Renderer(int width, int height);
        Renderer(scene::Scene* scene, int width, int height);
        ~Renderer();
        void set_scene(scene::Scene* scene);
        void render(camera::PerspectiveCamera* camera,
            py::array_t<GLuint, py::array::c_style> np_rgb_map);
        void render(scene::Scene* scene, camera::PerspectiveCamera* camera,
            py::array_t<GLuint, py::array::c_style> np_rgb_map);
    };
}
}