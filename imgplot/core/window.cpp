#include "window.h"
#include <iostream>

namespace imgplot {
Window::Window(Figure* figure, pybind11::tuple initial_size)
{
    _figure = figure;
    _closed = false;
    _mouse = { 0, 0, false };
    _initial_width = initial_size[0].cast<int>();
    _initial_height = initial_size[1].cast<int>();

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
    _window = glfwCreateWindow(1, 1, "", NULL, NULL);
    glfwMakeContextCurrent(_window);
    glfwSwapInterval(1);
    gl3wInit();
}
Window::~Window()
{
    glfwSetWindowShouldClose(_shared_window, GL_TRUE);
    glfwSetWindowShouldClose(_window, GL_TRUE);
    _thread.join();
    glfwDestroyWindow(_shared_window);
    glfwDestroyWindow(_window);
    glfwTerminate();
}
void Window::_run()
{
    glfwWindowHint(GLFW_VISIBLE, GL_TRUE);
    _shared_window = glfwCreateWindow(_initial_width, _initial_height, "Generative Query Network", NULL, _window);
    glfwMakeContextCurrent(_shared_window);
    glfwSetWindowUserPointer(_shared_window, this);

    glfwSetScrollCallback(_shared_window, [](GLFWwindow* window, double x, double y) {
        static_cast<Window*>(glfwGetWindowUserPointer(window))->_callback_scroll(window, x, y);
    });
    glfwSetCursorPosCallback(_shared_window, [](GLFWwindow* window, double x, double y) {
        static_cast<Window*>(glfwGetWindowUserPointer(window))->_callback_cursor_move(window, x, y);
    });
    glfwSetMouseButtonCallback(_shared_window, [](GLFWwindow* window, int button, int action, int mods) {
        static_cast<Window*>(glfwGetWindowUserPointer(window))->_callback_mouse_button(window, button, action, mods);
    });

    // glEnable(GL_BLEND);
    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glClearColor(0.0, 0.0, 0.0, 1.0);

    for (const auto& frame : _figure->_images) {
        data::ImageData* data = std::get<0>(frame);
        double x = std::get<1>(frame);
        double y = std::get<2>(frame);
        double width = std::get<3>(frame);
        double height = std::get<4>(frame);
        _images.push_back(std::make_unique<view::ImageView>(data, x, y, width, height));
    }

    for (const auto& frame : _figure->_objects) {
        data::ObjectData* data = std::get<0>(frame);
        double x = std::get<1>(frame);
        double y = std::get<2>(frame);
        double width = std::get<3>(frame);
        double height = std::get<4>(frame);
        _objects.push_back(std::make_unique<view::ObjectView>(data, x, y, width, height));
    }

    while (!!glfwWindowShouldClose(_shared_window) == false) {
        glfwPollEvents();
        int screen_width, screen_height;
        glfwGetFramebufferSize(_shared_window, &screen_width, &screen_height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(0, 0, 0, 1.0);

        for (const auto& view : _images) {
            _render_view(view.get());
        }

        for (const auto& view : _objects) {
            _render_view(view.get());
        }

        glfwSwapBuffers(_shared_window);
    }
    _closed = true;
}
void Window::_render_view(View* view)
{
    int screen_width, screen_height;
    glfwGetFramebufferSize(_shared_window, &screen_width, &screen_height);
    int x = screen_width * view->x();
    int y = screen_height * view->y();
    int width = screen_width * view->width();
    int height = screen_height * view->height();
    glViewport(x, screen_height - y - height, width, height);
    double aspect_ratio = (double)width / (double)height;
    view->render(aspect_ratio);
}
void Window::show()
{
    try {
        _thread = std::thread(&Window::_run, this);
    } catch (const std::system_error& e) {
        throw std::runtime_error(e.what());
    }
}
bool Window::closed()
{
    return _closed;
}
void Window::_callback_scroll(GLFWwindow* window, double x, double y)
{
    int screen_width, screen_height;
    glfwGetFramebufferSize(_shared_window, &screen_width, &screen_height);
    for (const auto& view : _objects) {
        if (view->contains(_mouse.x, _mouse.y, screen_width, screen_height)) {
            if (y < 0) {
                view->zoom_in();
            } else {
                view->zoom_out();
            }
        }
    }
}
void Window::_callback_cursor_move(GLFWwindow* window, double x, double y)
{
    if (_mouse.is_left_button_down) {
        int screen_width, screen_height;
        glfwGetFramebufferSize(_shared_window, &screen_width, &screen_height);
        for (const auto& view : _objects) {
            if (view->contains(_mouse.x, _mouse.y, screen_width, screen_height)) {
                double diff_x = _mouse.x - x;
                double diff_y = _mouse.y - y;
                view->rotate_camera(diff_x, diff_y);
            }
        }
    }
    _mouse.x = x;
    _mouse.y = y;
}

void Window::_callback_mouse_button(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        _mouse.is_left_button_down = (action == GLFW_PRESS);
    }
}
}