#include "image.h"
#include "../opengl.h"
#include <glm/glm.hpp>
#include <iostream>

namespace imgplot {
namespace view {
    ImageView::ImageView(std::shared_ptr<data::ImageData> data, double x, double y, double width, double height)
    {
        _data = data;
        _x = x;
        _y = y;
        _width = width;
        _height = height;
        _renderer = std::make_unique<renderer::ImageRenderer>();
    }
    double ImageView::x()
    {
        return _x;
    }
    double ImageView::y()
    {
        return _y;
    }
    double ImageView::width()
    {
        return _width;
    }
    double ImageView::height()
    {
        return _height;
    }
    void ImageView::bind_data()
    {
        _renderer->set_data(_data->raw(), _data->width(), _data->height());
    }
    void ImageView::render(double scale_x, double scale_y)
    {
        if (_data->updated()) {
            bind_data();
            _data->mark_as_updated();
        }
        _renderer->render(scale_x, scale_y);
    }
}
}
