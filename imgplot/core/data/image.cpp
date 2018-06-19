#include "image.h"

namespace imgplot {
namespace data {
    ImageData::ImageData(int height, int width, int num_channels)
    {
        if (num_channels != 1 && num_channels != 3) {
            throw std::runtime_error("num_channels != 1 && num_channels != 3");
        }
        if (height <= 0) {
            throw std::runtime_error("height <= 0");
        }
        if (width <= 0) {
            throw std::runtime_error("width <= 0");
        }
        _height = height;
        _width = width;
        _num_channels = num_channels;
        _data = std::make_unique<GLubyte[]>(height * width * 3);
    }
    void ImageData::resize(int height, int width, int num_channels)
    {
        _data = std::make_unique<GLubyte[]>(height * width * 3);
    }
    void ImageData::update(pybind11::array_t<GLubyte> data)
    {
        auto size = data.size();
        if (size != _height * _width * _num_channels) {
            throw std::runtime_error("`data.size` muse be equal to `_height * _width * _num_channels`.");
        }
        if (data.ndim() < 2 || data.ndim() > 3) {
            throw std::runtime_error("(data.ndim() < 2 || data.ndim() > 3) -> false");
        }
        if (data.ndim() == 2 && _num_channels != 1) {
            throw std::runtime_error("(data.ndim() == 2 && _num_channels != 1) -> false");
        }
        if (data.ndim() == 3 && _num_channels != 3) {
            throw std::runtime_error("(data.ndim() == 3 && _num_channels != 3) -> false");
        }
        if (data.ndim() == 2) {
            auto ptr = data.mutable_unchecked<2>();
            for (ssize_t h = 0; h < data.shape(0); h++) {
                for (ssize_t w = 0; w < data.shape(1); w++) {
                    ssize_t index = h * _width + w;
                    GLubyte intensity = ptr(h, w);
                    _data[index * 3 + 0] = intensity;
                    _data[index * 3 + 1] = intensity;
                    _data[index * 3 + 2] = intensity;
                }
            }
        } else {
            auto ptr = data.mutable_unchecked<3>();
            for (ssize_t h = 0; h < data.shape(0); h++) {
                for (ssize_t w = 0; w < data.shape(1); w++) {
                    for (ssize_t c = 0; c < data.shape(2); c++) {
                        ssize_t index = h * _width * _num_channels + w * _num_channels + c;
                        _data[index] = ptr(h, w, c);
                    }
                }
            }
        }
        _updated = true;
    }
    bool ImageData::updated()
    {
        bool ret = _updated;
        _updated = false;
        return ret;
    }
    GLubyte* ImageData::raw()
    {
        return _data.get();
    }
    int ImageData::height()
    {
        return _height;
    }
    int ImageData::width()
    {
        return _width;
    }
}
}
