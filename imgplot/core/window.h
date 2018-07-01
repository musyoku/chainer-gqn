#pragma once
#include "figure.h"
#include "view/image.h"
#include <gl3w/gl3w.h>
#include <glfw/glfw3.h>
#include <iostream>
#include <pybind11/pybind11.h>
#include <thread>
#include <vector>
#include <string>

namespace imgplot {
class Window {
protected:
    std::thread _thread;
    GLFWwindow* _window;
    GLFWwindow* _shared_window;
    Figure* _figure;
    std::vector<std::unique_ptr<view::ImageView>> _images;
    bool _closed;
    int _initial_width;
    int _initial_height;
    std::string _title;
    void run();
    void render_view(view::ImageView* view);

public:
    Window(Figure* figure, pybind11::tuple initial_size, std::string title);
    ~Window();
    void show();
    bool closed();
};
}