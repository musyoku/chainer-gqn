#include "renderer.h"
#include "opengl/functions.h"
#include <glm/gtc/type_ptr.hpp>
#include <iostream>

namespace three {
namespace renderer {
    void DebugCallbackFunc(GLenum source, GLenum type, GLuint eid, GLenum severity, GLsizei length, const GLchar* message, const void* user_param)
    {
        std::cout << message << std::endl;
    }
    void Renderer::initialize(int width, int height)
    {
        glfwSetErrorCallback([](int error, const char* description) {
            fprintf(stderr, "Error %d: %s\n", error, description);
        });
        if (!!glfwInit() == false) {
            throw std::runtime_error("Failed to initialize GLFW.");
        }

        _width = width;
        _height = height;
        _color_pixels = std::make_unique<GLubyte[]>(width * height * 3);

        // // SSAO
        // std::random_device seed;
        // std::default_random_engine engine(seed());
        // std::normal_distribution<> normal_dist(0.0, 1.0);
        // _ssao_texture_data = std::make_unique<GLfloat[]>(192 * 3);
        // for (int i = 0; i < 192; i++) {
        //     _ssao_texture_data[i * 3 + 0] = normal_dist(engine);
        //     _ssao_texture_data[i * 3 + 1] = normal_dist(engine);
        //     _ssao_texture_data[i * 3 + 2] = 0;
        // }

        _prev_num_objects = -1;

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_VISIBLE, GL_FALSE);

        _window = glfwCreateWindow(width, height, "Renderer", NULL, NULL);
        glfwMakeContextCurrent(_window);
        gl3wInit();

        _vao = std::make_unique<opengl::VertexArrayObject>();
        _depth_render_pass = std::make_unique<multipass::DepthBuffer>(width, height);
        _ssao_render_pass = std::make_unique<multipass::ScreenSpaceAmbientOcculusion>(width, height, 192);
        _blur_render_pass = std::make_unique<multipass::Blur>(width, height);
        _main_render_pass = std::make_unique<multipass::Main>(width, height);

        glEnable(GL_CULL_FACE);
        glEnable(GL_DEPTH_TEST);
        // glEnable(GL_TEXTURE_1D);
        // glDepthMask(GL_TRUE);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glClearColor(0.0, 0.0, 0.0, 1.0);

        glDebugMessageCallback(DebugCallbackFunc, nullptr);
        glEnable(GL_DEBUG_OUTPUT);
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
        glfwDestroyWindow(_window);
        glfwTerminate();
    }
    void Renderer::set_scene(scene::Scene* scene)
    {
        glfwMakeContextCurrent(_window);
        _scene = scene;
        _vao->build(scene);
    }
    void Renderer::draw_objects(camera::PerspectiveCamera* camera, RenderPass* pass)
    {
        glm::mat4& view_mat = camera->_view_matrix;
        glm::mat4& projection_mat = camera->_projection_matrix;
        std::vector<std::shared_ptr<base::Object>>& objects = _scene->_objects;
        for (int object_index = 0; object_index < objects.size(); object_index++) {
            std::shared_ptr<base::Object> object = objects[object_index];
            float smoothness = object->_smoothness ? 1.0 : 0.0;
            glm::mat4& model_mat = object->_model_matrix;

            _vao->bind_object(object_index);
            pass->uniform_matrix_4fv(_uniform_model_mat, glm::value_ptr(model_mat));
            pass->uniform_matrix_4fv(_uniform_view_mat, glm::value_ptr(view_mat));
            pass->uniform_matrix_4fv(_uniform_projection_mat, glm::value_ptr(projection_mat));

            pass->set_additional_uniform_variables(object.get());

            glDrawArrays(GL_TRIANGLES, 0, 3 * object->_num_faces);
        }
    }
    void Renderer::render(
        camera::PerspectiveCamera* camera,
        py::array_t<GLuint, py::array::c_style> np_rgb_map)
    {
        if (glfwWindowShouldClose(_window)) {
            return;
        }

        // OpenGL commands are executed in global context (per thread).
        glfwMakeContextCurrent(_window);

        int screen_width, screen_height;
        glfwGetFramebufferSize(_window, &screen_width, &screen_height);

        // Z pre-pass
        if (_depth_render_pass->bind(_width, _height)) {
            draw_objects(camera, _depth_render_pass.get());

            // glBindTexture(GL_TEXTURE_2D, _depth_texture);

            // std::unique_ptr<GLfloat[]> buffer = std::make_unique<GLfloat[]>(_width * _height);
            // glReadPixels(0, 0, _width, _height, GL_DEPTH_COMPONENT, GL_FLOAT, buffer.get());
            // auto rgb_map = np_rgb_map.mutable_unchecked<3>();

            // GLfloat min_value = 1.0;
            // for (int h = 0; h < _height; h++) {
            //     for (int w = 0; w < _width; w++) {
            //         GLfloat depth = buffer[(_height - h - 1) * _width + w];
            //         if(depth < min_value){
            //             min_value = depth;
            //         }
            //     }
            // }
            // std::cout << min_value << std::endl;

            // for (int h = 0; h < _height; h++) {
            //     for (int w = 0; w < _width; w++) {
            //         rgb_map(h, w, 0) = (buffer[(_height - h - 1) * _width + w] - min_value) / (1.0 - min_value) * 255.0;
            //         rgb_map(h, w, 1) = (buffer[(_height - h - 1) * _width + w] - min_value) / (1.0 - min_value) * 255.0;
            //         rgb_map(h, w, 2) = (buffer[(_height - h - 1) * _width + w] - min_value) / (1.0 - min_value) * 255.0;
            //     }
            // }

            // glReadPixels(0, 0, _width, _height, GL_RGB, GL_UNSIGNED_BYTE, _color_pixels.get());
            // for (int h = 0; h < _height; h++) {
            //     for (int w = 0; w < _width; w++) {
            //         rgb_map(h, w, 0) = _color_pixels[(_height - h - 1) * _width * 3 + w * 3 + 0];
            //         rgb_map(h, w, 1) = _color_pixels[(_height - h - 1) * _width * 3 + w * 3 + 1];
            //         rgb_map(h, w, 2) = _color_pixels[(_height - h - 1) * _width * 3 + w * 3 + 2];
            //     }
            // }


            _depth_render_pass->unbind();
        }

        // SSAO
        if (_ssao_render_pass->bind(64, 0.5, _depth_render_pass->get_buffer_texture_id())) {
            draw_objects(camera, _ssao_render_pass.get());
            _ssao_render_pass->unbind();
        }

        // SSAO Blur
        if (_blur_render_pass->bind(_ssao_render_pass->get_buffer_texture_id())) {
            draw_objects(camera, _blur_render_pass.get());
            _blur_render_pass->unbind();
        }

        // Main pass
        if (_main_render_pass->bind(_blur_render_pass->get_buffer_texture_id())) {
            draw_objects(camera, _main_render_pass.get());

            glReadPixels(0, 0, _width, _height, GL_RGB, GL_UNSIGNED_BYTE, _color_pixels.get());
            auto rgb_map = np_rgb_map.mutable_unchecked<3>();
            for (int h = 0; h < _height; h++) {
                for (int w = 0; w < _width; w++) {
                    rgb_map(h, w, 0) = _color_pixels[(_height - h - 1) * _width * 3 + w * 3 + 0];
                    rgb_map(h, w, 1) = _color_pixels[(_height - h - 1) * _width * 3 + w * 3 + 1];
                    rgb_map(h, w, 2) = _color_pixels[(_height - h - 1) * _width * 3 + w * 3 + 2];
                }
            }

            _main_render_pass->unbind();
        }
    }
    void Renderer::render(scene::Scene* scene, camera::PerspectiveCamera* camera,
        py::array_t<GLuint, py::array::c_style> np_rgb_map)
    {
        set_scene(scene);
        render(camera, np_rgb_map);
    }
}
}