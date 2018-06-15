#include "figure.h"

namespace viewer {
void Figure::add(data::ImageData* data, double x, double y, double width, double height)
{
    _images.emplace_back(std::make_tuple(data, x, y, width, height));
}
void Figure::add(data::ObjectData* data, double x, double y, double width, double height)
{
    _objects.emplace_back(std::make_tuple(data, x, y, width, height));
}
}