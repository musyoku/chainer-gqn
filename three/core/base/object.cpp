#include "object.h"
#include <glm/gtc/matrix_transform.hpp>
#include <stdexcept>

namespace three {
namespace base {
    void Object::reserve(int num_faces, int num_vertices){
        _faces = std::make_unique<glm::vec3i[]>(num_faces);
        _vertices = std::make_unique<glm::vec3f[]>(num_vertices);
        _face_vertices = std::make_unique<glm::vec3f[]>(num_faces * 3);
        _face_vertex_colors = std::make_unique<glm::vec4f[]>(num_faces * 3);
        _face_normal_vectors = std::make_unique<glm::vec3f[]>(num_faces * 3);
        _face_vertex_normal_vectors = std::make_unique<glm::vec3f[]>(num_faces * 3);
    }
    void Object::set_scale(py::tuple scale)
    {
        _scale[0] = scale[0].cast<float>();
        _scale[1] = scale[1].cast<float>();
        _scale[2] = scale[2].cast<float>();
        update_model_matrix();
    }
    void Object::set_position(py::tuple position)
    {
        _position[0] = position[0].cast<float>();
        _position[1] = position[1].cast<float>();
        _position[2] = position[2].cast<float>();
        update_model_matrix();
    }
    void Object::set_rotation(py::tuple rotation_rad)
    {
        _rotation_rad[0] = rotation_rad[0].cast<float>();
        _rotation_rad[1] = rotation_rad[1].cast<float>();
        _rotation_rad[2] = rotation_rad[2].cast<float>();
        update_model_matrix();
    }
    void Object::update_model_matrix()
    {
        _model_matrix = glm::mat4(1.0);
        _model_matrix = glm::translate(_model_matrix, _position);
        _model_matrix = glm::rotate(_model_matrix, _rotation_rad[0], glm::vec3(1.0f, 0.0f, 0.0f));
        _model_matrix = glm::rotate(_model_matrix, _rotation_rad[1], glm::vec3(0.0f, 1.0f, 0.0f));
        _model_matrix = glm::rotate(_model_matrix, _rotation_rad[2], glm::vec3(0.0f, 0.0f, 1.0f));
        _model_matrix = glm::scale(_model_matrix, _scale);
    }
}
}