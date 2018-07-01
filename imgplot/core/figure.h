#pragma once
#include "data/image.h"
#include <vector>
#include <memory>

namespace imgplot {
class Figure {
public:
    std::vector<std::tuple<std::shared_ptr<data::ImageData>, double, double, double, double>> _images;
    void add(std::shared_ptr<data::ImageData> data, double x, double y, double width, double height);
};
}