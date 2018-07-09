#include "light.h"

namespace three {
namespace base {
    Light::Light(float intensity)
    {
        set_intensity(intensity);
        _position = glm::vec3(0.0);
    }
    Light::Light(py::tuple position, float intensity)
    {
        set_intensity(intensity);
        set_position(position);
    }
    void Light::set_position(py::tuple position)
    {
        _position = glm::vec3(position[0].cast<float>(), position[1].cast<float>(), position[2].cast<float>());
    }
    void Light::set_intensity(float intensity)
    {
        _intensity = intensity;
    }
}
}