#include <glm/glm.hpp>
#include <memory>
#include <pybind11/numpy.h>

namespace environment {
namespace scene {
    class Object {
    private:
        std::unique_ptr<int[]> _faces;
        std::unique_ptr<float[]> _vertices;
        int _num_faces;
        int _num_vertices;

    public:
        Object(pybind11::array_t<int> np_faces, pybind11::array_t<float> np_vertices);
        glm::vec2 _location;
        glm::vec3 _rotation_rad;
    };
}
}