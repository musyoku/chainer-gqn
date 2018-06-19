#pragma once

namespace imgplot {
class View {
protected:
    double _x;
    double _y;
    double _width;
    double _height;

public:
    View(double x, double y, double width, double height);
    double x();
    double y();
    double width();
    double height();
    bool contains(double px, double py, int screen_width, int screen_height);
    virtual void render(double aspect_ratio);
};
}