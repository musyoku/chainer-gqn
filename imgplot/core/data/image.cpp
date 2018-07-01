#include "image.h"
#include <iostream>

namespace imgplot {
namespace data {
    ImageData::ImageData()
    {
    }
    ImageData::ImageData(pybind11::array_t<GLubyte> data)
    {
        reserve(data.shape(1), data.shape(0), data.shape(2));
        update(data);
    }
    void ImageData::reserve(int width, int height, int num_channels)
    {
        if (num_channels != 1 && num_channels != 3) {
            throw std::runtime_error("num_channels != 1 && num_channels != 3");
        }
        if (width <= 0) {
            throw std::runtime_error("width <= 0");
        }
        if (height <= 0) {
            throw std::runtime_error("height <= 0");
        }
        if (width % 2 != 0) {
            throw std::runtime_error("width % 2 != 0");
        }
        if (height % 2 != 0) {
            throw std::runtime_error("height % 2 != 0");
        }
        _height = height;
        _width = width;
        _num_channels = num_channels;
        _data = std::make_unique<GLubyte[]>(height * width * 3);
    }
    void ImageData::reserve_if_needed(pybind11::array_t<GLubyte> data)
    {
        if (data.ndim() < 2 || data.ndim() > 3) {
            throw std::runtime_error("(data.ndim() < 2 || data.ndim() > 3) -> false");
        }
        if (data.shape(0) != _height || data.shape(1) != _width) {
            int num_channels = (data.ndim() == 2) ? 1 : data.shape(2);
            return reserve(data.shape(1), data.shape(0), num_channels);
        }
        if (data.ndim() == 2 && _num_channels == 3) {
            return reserve(data.shape(1), data.shape(0), 1);
        }
        if (data.ndim() == 3 && _num_channels == 1) {
            return reserve(data.shape(1), data.shape(0), 3);
        }
    }

    void ImageData::validate_data(pybind11::array_t<GLubyte> data)
    {
        if (!!(data.ndim() == 2 || data.ndim() == 3) == false) {
            throw std::runtime_error("Invalid ndim");
        }
        int num_channels = (data.ndim() == 2) ? 1 : data.shape(2);
        if (!!(num_channels == 1 || num_channels == 3) == false) {
            throw std::runtime_error("Invalid #channels");
        }
    }
    void ImageData::update(pybind11::array_t<GLubyte> data)
    {
        validate_data(data);
        reserve_if_needed(data);
        if (data.ndim() == 2) {
            auto ptr = data.mutable_unchecked<2>();
            for (ssize_t h = 0; h < _height; h++) {
                for (ssize_t w = 0; w < _width; w++) {
                    ssize_t index = h * _width + w;
                    GLubyte intensity = ptr(h, w);
                    _data[index * 3 + 0] = intensity;
                    _data[index * 3 + 1] = intensity;
                    _data[index * 3 + 2] = intensity;
                }
            }
        } else {
            auto ptr = data.mutable_unchecked<3>();
            for (ssize_t h = 0; h < _height; h++) {
                for (ssize_t w = 0; w < _width; w++) {
                    for (ssize_t c = 0; c < _num_channels; c++) {
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
        return _updated;
    }
    void ImageData::mark_as_updated()
    {
        _updated = false;
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
