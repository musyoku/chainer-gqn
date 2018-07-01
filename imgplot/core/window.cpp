#include "window.h"

namespace imgplot {
Window::Window(Figure* figure, pybind11::tuple initial_size, std::string title)
{
    _figure = figure;
    _closed = false;
    _title = title;
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
void Window::run()
{
    glfwWindowHint(GLFW_VISIBLE, GL_TRUE);
    _shared_window = glfwCreateWindow(_initial_width, _initial_height, _title.c_str(), NULL, _window);
    glfwMakeContextCurrent(_shared_window);
    glfwSetWindowUserPointer(_shared_window, this);

    // glEnable(GL_BLEND);
    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    // glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glClearColor(0.0, 0.0, 0.0, 1.0);

    for (const auto& frame : _figure->_images) {
        std::shared_ptr<data::ImageData> data = std::get<0>(frame);
        double x = std::get<1>(frame);
        double y = std::get<2>(frame);
        double width = std::get<3>(frame);
        double height = std::get<4>(frame);
        _images.push_back(std::make_unique<view::ImageView>(data, x, y, width, height));
    }

    while (!!glfwWindowShouldClose(_shared_window) == false) {
        glfwPollEvents();
        int screen_width, screen_height;
        glfwGetFramebufferSize(_shared_window, &screen_width, &screen_height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(0, 0, 0, 1.0);
        for (const auto& view : _images) {
            render_view(view.get());
        }
        glfwSwapBuffers(_shared_window);
    }
    _closed = true;
}
void Window::render_view(view::ImageView* view)
{
    int screen_width, screen_height;
    glfwGetFramebufferSize(_shared_window, &screen_width, &screen_height);
    int data_width = view->_data->width();
    int data_height = view->_data->height();
    double data_aspect_ratio = (double)data_width / (double)data_height;
    int view_x = screen_width * view->x();
    int view_y = screen_height * view->y();
    int view_width = screen_width * view->width();
    int view_height = screen_height * view->height();
    double view_aspect_ratio = (double)view_width / (double)view_height;
    glViewport(view_x, screen_height - view_y - view_height, view_width, view_height);
    double scale_x = 1.0;
    double scale_y = 1.0;
    if (view_aspect_ratio > data_aspect_ratio) {
        scale_x = 1.0 / (view_aspect_ratio / data_aspect_ratio);
    }
    if (view_aspect_ratio < data_aspect_ratio) {
        scale_y = (view_aspect_ratio / data_aspect_ratio);
    }
    view->render(scale_x, scale_y);
}
void Window::show()
{
    try {
        _thread = std::thread(&Window::run, this);
    } catch (const std::system_error& e) {
        throw std::runtime_error(e.what());
    }
}
bool Window::closed()
{
    return _closed;
}
}