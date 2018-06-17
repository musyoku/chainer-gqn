#include "object.h"
#include <glm/gtc/matrix_transform.hpp>
#include <stdexcept>

namespace three {
namespace scene {
    Object::Object(py::array_t<int> np_faces, py::array_t<float> np_vertices, py::tuple color, py::tuple scale)
    {
        if (np_faces.ndim() != 2) {
            throw std::runtime_error("(np_faces.ndim() != 2) -> false");
        }
        if (np_vertices.ndim() != 2) {
            throw std::runtime_error("(np_vertices.ndim() != 2) -> false");
        }
        if (np_faces.shape(1) != 3) {
            throw std::runtime_error("(np_faces.shape(1) != 3) -> false");
        }
        if (np_vertices.shape(1) != 3) {
            throw std::runtime_error("(np_vertices.shape(1) != 3) -> false");
        }
        _num_faces = np_faces.shape(0);
        _num_vertices = np_vertices.shape(0);
        _faces = std::make_unique<glm::vec3i[]>(_num_faces);
        _vertices = std::make_unique<glm::vec3f[]>(_num_vertices);
        _face_vertices = std::make_unique<glm::vec3f[]>(_num_faces * 3);
        _face_normal_vectors = std::make_unique<glm::vec3f[]>(_num_faces);

        auto faces = np_faces.mutable_unchecked<2>();
        for (ssize_t face_index = 0; face_index < np_faces.shape(0); face_index++) {
            _faces[face_index] = glm::vec3i(faces(face_index, 0), faces(face_index, 1), faces(face_index, 2));
        }

        auto vertices = np_vertices.mutable_unchecked<2>();
        for (ssize_t vertex_index = 0; vertex_index < np_vertices.shape(0); vertex_index++) {
            _vertices[vertex_index] = glm::vec3f(vertices(vertex_index, 0), vertices(vertex_index, 1), vertices(vertex_index, 2));
        }

        for (ssize_t face_index = 0; face_index < np_faces.shape(0); face_index++) {
            glm::vec3i face = _faces[face_index];

            glm::vec3f va = glm::vec3f(vertices(face[0], 0), vertices(face[0], 1), vertices(face[0], 2));
            glm::vec3f vb = glm::vec3f(vertices(face[1], 0), vertices(face[1], 1), vertices(face[1], 2));
            glm::vec3f vc = glm::vec3f(vertices(face[2], 0), vertices(face[2], 1), vertices(face[2], 2));

            glm::vec3f vba = vb - va;
            glm::vec3f vca = vc - va;
            glm::vec3f normal = glm::normalize(glm::cross(vba, vca));

            _face_vertices[face_index * 3 + 0] = va;
            _face_vertices[face_index * 3 + 1] = vb;
            _face_vertices[face_index * 3 + 2] = vc;

            _face_normal_vectors[face_index] = normal;
        }

        _position = glm::vec3(0.0);
        _rotation_rad = glm::vec3(0.0);

        _color[0] = color[0].cast<float>();
        _color[1] = color[1].cast<float>();
        _color[2] = color[2].cast<float>();
        _color[3] = color[3].cast<float>();

        _scale[0] = scale[0].cast<float>();
        _scale[1] = scale[1].cast<float>();
        _scale[2] = scale[2].cast<float>();

        _update_model_matrix();
    }
    void Object::set_position(py::tuple position)
    {
        _position[0] = position[0].cast<float>();
        _position[1] = position[1].cast<float>();
        _position[2] = position[2].cast<float>();
        _update_model_matrix();
    }
    void Object::set_rotation(py::tuple rotation_rad)
    {
        _rotation_rad[0] = rotation_rad[0].cast<float>();
        _rotation_rad[1] = rotation_rad[1].cast<float>();
        _rotation_rad[2] = rotation_rad[2].cast<float>();
        _update_model_matrix();
    }
    void Object::_update_model_matrix()
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