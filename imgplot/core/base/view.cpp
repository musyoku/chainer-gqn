#include "view.h"
#include <stdexcept>

namespace imgplot {
View::View(double x, double y, double width, double height)
{
    _x = x;
    _y = y;
    _width = width;
    _height = height;
}
double View::x()
{
    return _x;
}
double View::y()
{
    return _y;
}
double View::width()
{
    return _width;
}
double View::height()
{
    return _height;
}
void View::render(double aspect_ratio)
{
    throw std::runtime_error("Function `render` must be overridden.");
}

bool View::contains(double px, double py, int screen_width, int screen_height)
{
    int x = screen_width * _x;
    int y = screen_height * _y;
    int width = screen_width * _width;
    int height = screen_height * _height;
    y = screen_height - y - height;

    if (px < x) {
        return false;
    }
    if (px > x + width) {
        return false;
    }
    if (py < y) {
        return false;
    }
    if (py > y + height) {
        return false;
    }
    return true;
}
}