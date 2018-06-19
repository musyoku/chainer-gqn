#include "object.h"
#include <glm/glm.hpp>
#include <iostream>

namespace imgplot {
namespace data {
    ObjectData::ObjectData(pybind11::array_t<GLfloat> vertices, int num_vertices, pybind11::array_t<GLuint> faces, int num_faces)
    {
        _num_vertices = num_vertices;
        _num_faces = num_faces;
        _vertices = std::make_unique<GLfloat[]>(num_vertices * 3);
        _faces = std::make_unique<GLuint[]>(num_faces * 3);
        _vertices_normal_vectors = std::make_unique<GLfloat[]>(num_faces * 9);
        _extracted_vertices = std::make_unique<GLfloat[]>(num_faces * 9);
        _update(vertices, faces);
    }
    void ObjectData::_update(pybind11::array_t<GLfloat> vertices, pybind11::array_t<GLuint> faces)
    {
        _update_faces(faces);
        _update_vertices(vertices);
        _update_normal_vectors();
    }
    void ObjectData::_update_vertices(pybind11::array_t<GLfloat> vertices)
    {
        auto size = vertices.size();
        if (size != _num_vertices * 3) {
            throw std::runtime_error("`vertices.size` muse be equal to `_num_faces * 3`.");
        }
        if (vertices.ndim() != 2) {
            throw std::runtime_error("(vertices.ndim() != 2) -> false");
        }
        auto ptr = vertices.mutable_unchecked<2>();
        for (int n = 0; n < vertices.shape(0); n++) {
            _vertices[n * 3 + 0] = ptr(n, 0);
            _vertices[n * 3 + 1] = ptr(n, 1);
            _vertices[n * 3 + 2] = ptr(n, 2);
        }
        for (int n = 0; n < _num_faces; n++) {
            for (int f = 0; f < 3; f++) {
                int face_index = _faces[n * 3 + f];
                for (int v = 0; v < 3; v++) {
                    _extracted_vertices[n * 9 + f * 3 + v] = _vertices[face_index * 3 + v];
                }
            }
        }
        _vertices_updated = true;
    }
    void ObjectData::_update_faces(pybind11::array_t<GLuint> faces)
    {
        auto size = faces.size();
        if (size != _num_faces * 3) {
            throw std::runtime_error("`faces.size` muse be equal to `_num_faces * 3`.");
        }
        if (faces.ndim() != 2) {
            throw std::runtime_error("(faces.ndim() != 2) -> false");
        }
        auto ptr = faces.mutable_unchecked<2>();
        for (int n = 0; n < faces.shape(0); n++) {
            _faces[n * 3 + 0] = ptr(n, 0);
            _faces[n * 3 + 1] = ptr(n, 1);
            _faces[n * 3 + 2] = ptr(n, 2);
        }
        _faces_updated = true;
    }
    void ObjectData::_update_normal_vectors()
    {
        std::unique_ptr<glm::vec3> face_normal_vectors;
        for (int n = 0; n < _num_faces; n++) {
            int fa = _faces[n * 3 + 0];
            int fb = _faces[n * 3 + 1];
            int fc = _faces[n * 3 + 2];
            glm::vec3 va = glm::vec3(_vertices[fa * 3 + 0], _vertices[fa * 3 + 1], _vertices[fa * 3 + 2]);
            glm::vec3 vb = glm::vec3(_vertices[fb * 3 + 0], _vertices[fb * 3 + 1], _vertices[fb * 3 + 2]);
            glm::vec3 vc = glm::vec3(_vertices[fc * 3 + 0], _vertices[fc * 3 + 1], _vertices[fc * 3 + 2]);
            glm::vec3 vba = vb - va;
            glm::vec3 vca = vc - va;
            glm::vec3 normal = glm::normalize(glm::cross(vba, vca));
            for (int face = 0; face < 3; face++) {
                _vertices_normal_vectors[n * 9 + face * 3 + 0] = normal.x;
                _vertices_normal_vectors[n * 9 + face * 3 + 1] = normal.y;
                _vertices_normal_vectors[n * 9 + face * 3 + 2] = normal.z;
            }
        }
        _normal_vector_updated = true;
    }
    void ObjectData::update_vertices(pybind11::array_t<GLfloat> vertices)
    {
        _update_vertices(vertices);
        _update_normal_vectors();
    }
    void ObjectData::update_faces(pybind11::array_t<GLuint> faces)
    {
        _update_faces(faces);
        _update_normal_vectors();
    }
    void ObjectData::update(pybind11::array_t<GLfloat> vertices, pybind11::array_t<GLuint> faces)
    {
        _update_vertices(vertices);
        _update_faces(faces);
        _update_normal_vectors();
    }
    int ObjectData::num_vertices()
    {
        return _num_vertices;
    }
    int ObjectData::num_extracted_vertices()
    {
        return _num_faces * 3;
    }
    int ObjectData::num_faces()
    {
        return _num_faces;
    }
    bool ObjectData::vertices_updated()
    {
        bool ret = _vertices_updated;
        _vertices_updated = false;
        return ret;
    }
    bool ObjectData::faces_updated()
    {
        bool ret = _faces_updated;
        _faces_updated = false;
        return ret;
    }
    bool ObjectData::normal_vector_updated()
    {
        bool ret = _normal_vector_updated;
        _normal_vector_updated = false;
        return ret;
    }
    GLfloat* ObjectData::vertices()
    {
        return _vertices.get();
    }
    GLfloat* ObjectData::extracted_vertices()
    {
        return _extracted_vertices.get();
    }
    GLfloat* ObjectData::normal_vectors()
    {
        return _vertices_normal_vectors.get();
    }
    GLuint* ObjectData::faces()
    {
        return _faces.get();
    }
}
}
