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
        reserve(_num_faces, _num_vertices);

        auto faces = np_faces.mutable_unchecked<2>();
        for (ssize_t face_index = 0; face_index < _num_faces; face_index++) {
            _faces[face_index] = glm::vec3i(faces(face_index, 0), faces(face_index, 1), faces(face_index, 2));
        }

        auto vertices = np_vertices.mutable_unchecked<2>();
        std::unique_ptr<glm::vec3f[]> vertex_normal_vectors = std::make_unique<glm::vec3f[]>(_num_vertices);
        for (ssize_t vertex_index = 0; vertex_index < _num_vertices; vertex_index++) {
            _vertices[vertex_index] = glm::vec3f(vertices(vertex_index, 0), vertices(vertex_index, 1), vertices(vertex_index, 2));
            vertex_normal_vectors[vertex_index] = glm::vec3f(0.0);
        }

        for (ssize_t face_index = 0; face_index < _num_faces; face_index++) {
            glm::vec3i face = _faces[face_index];

            glm::vec3f va = _vertices[face[0]];
            glm::vec3f vb = _vertices[face[1]];
            glm::vec3f vc = _vertices[face[2]];

            glm::vec3f vba = vb - va;
            glm::vec3f vca = vc - va;
            glm::vec3f normal = glm::normalize(glm::cross(vba, vca));

            _face_vertices[face_index * 3 + 0] = va;
            _face_vertices[face_index * 3 + 1] = vb;
            _face_vertices[face_index * 3 + 2] = vc;

            _face_normal_vectors[face_index * 3 + 0] = normal;
            _face_normal_vectors[face_index * 3 + 1] = normal;
            _face_normal_vectors[face_index * 3 + 2] = normal;

            vertex_normal_vectors[face[0]] += normal;
            vertex_normal_vectors[face[1]] += normal;
            vertex_normal_vectors[face[2]] += normal;
        }

        for (ssize_t vertex_index = 0; vertex_index < _num_vertices; vertex_index++) {
            vertex_normal_vectors[vertex_index] = glm::normalize(vertex_normal_vectors[vertex_index]);
        }

        for (ssize_t face_index = 0; face_index < _num_faces; face_index++) {
            glm::vec3i face = _faces[face_index];
            _face_vertex_normal_vectors[face_index * 3 + 0] = vertex_normal_vectors[face[0]];
            _face_vertex_normal_vectors[face_index * 3 + 1] = vertex_normal_vectors[face[1]];
            _face_vertex_normal_vectors[face_index * 3 + 2] = vertex_normal_vectors[face[2]];
        }

        _position = glm::vec3(0.0);
        _rotation_rad = glm::vec3(0.0);

        set_color(color);
        set_scale(scale);
        update_model_matrix();
    }
    Object::Object(const Object* source)
    {
        _num_faces = source->_num_faces;
        _num_vertices = source->_num_vertices;
        reserve(_num_faces, _num_vertices);

        for (ssize_t face_index = 0; face_index < _num_faces; face_index++) {
            _faces[face_index] = source->_faces[face_index];
            for (int offset = 0; offset < 3; offset++) {
                _face_vertices[face_index * 3 + offset] = source->_face_vertices[face_index * 3 + offset];
                _face_vertex_colors[face_index * 3 + offset] = source->_face_vertex_colors[face_index * 3 + offset];
                _face_normal_vectors[face_index * 3 + offset] = source->_face_normal_vectors[face_index * 3 + offset];
                _face_vertex_normal_vectors[face_index * 3 + offset] = source->_face_vertex_normal_vectors[face_index * 3 + offset];
            }
        }

        for (ssize_t vertex_index = 0; vertex_index < _num_vertices; vertex_index++) {
            _vertices[vertex_index] = source->_vertices[vertex_index];
        }

        _position = source->_position;
        _rotation_rad = source->_rotation_rad;
        _color = source->_color;
        _scale = source->_scale;
        _model_matrix = source->_model_matrix;
    }
    void Object::set_color(py::tuple color)
    {
        _color[0] = color[0].cast<float>();
        _color[1] = color[1].cast<float>();
        _color[2] = color[2].cast<float>();
        _color[3] = color[3].cast<float>();
        for (ssize_t face_index = 0; face_index < _num_faces; face_index++) {
            _face_vertex_colors[face_index * 3 + 0] = _color;
            _face_vertex_colors[face_index * 3 + 1] = _color;
            _face_vertex_colors[face_index * 3 + 2] = _color;
        }
    }
    std::shared_ptr<Object> Object::clone()
    {
        return std::make_shared<Object>(this);
    }
}
}