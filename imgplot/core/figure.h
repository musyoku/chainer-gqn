#pragma once
#include "data/image.h"
#include "data/object.h"
#include <vector>

namespace imgplot {
class Figure {
public:
    std::vector<std::tuple<data::ImageData*, double, double, double, double>> _images;
    std::vector<std::tuple<data::ObjectData*, double, double, double, double>> _objects;
    void add(data::ImageData* data, double x, double y, double width, double height);
    void add(data::ObjectData* data, double x, double y, double width, double height);
};
}