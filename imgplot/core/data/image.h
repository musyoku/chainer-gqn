#pragma once
#include <gl3w/gl3w.h>
#include <memory>
#include <pybind11/numpy.h>

namespace imgplot {
namespace data {
    class ImageData {
    private:
        int _height;
        int _width;
        int _num_channels;
        bool _updated;
        std::unique_ptr<GLubyte[]> _data;
        void reserve(int width, int height, int num_channels);
        void reserve_if_needed(pybind11::array_t<GLubyte> data);
        void validate_data(pybind11::array_t<GLubyte> data);

    public:
        ImageData();
        ImageData(pybind11::array_t<GLubyte> data);
        void update(pybind11::array_t<GLubyte> data);
        bool updated();
        void mark_as_updated();
        GLubyte* raw();
        int height();
        int width();
    };
}
}