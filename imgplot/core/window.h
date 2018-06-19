#pragma once
#include "figure.h"
#include "view/image.h"
#include "view/object.h"
#include <gl3w/gl3w.h>
#include <glfw/glfw3.h>
#include <iostream>
#include <pybind11/pybind11.h>
#include <thread>
#include <vector>

namespace imgplot {
struct Mouse {
    double x;
    double y;
    bool is_left_button_down;
};
class Window {
protected:
    std::thread _thread;
    GLFWwindow* _window;
    GLFWwindow* _shared_window;
    Figure* _figure;
    std::vector<std::unique_ptr<view::ImageView>> _images;
    std::vector<std::unique_ptr<view::ObjectView>> _objects;
    bool _closed;
    int _initial_width;
    int _initial_height;
    Mouse _mouse;
    void _run();
    void _render_view(View* view);
    void _callback_scroll(GLFWwindow* window, double x, double y);
    void _callback_cursor_move(GLFWwindow* window, double x, double y);
    void _callback_mouse_button(GLFWwindow* window, int button, int action, int mods);

public:
    Window(Figure* figure, pybind11::tuple initial_size);
    ~Window();
    void show();
    bool closed();
};
}