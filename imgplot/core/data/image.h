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

    public:
        ImageData(int height, int width, int num_channels);
        void resize(int height, int width, int num_channels);
        void update(pybind11::array_t<GLubyte> data);
        bool updated();
        GLubyte* raw();
        int height();
        int width();
    };
}
}