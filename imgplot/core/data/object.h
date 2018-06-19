#pragma once
#include <gl3w/gl3w.h>
#include <memory>
#include <pybind11/numpy.h>

namespace imgplot {
namespace data {
    class ObjectData {
    private:
        int _num_vertices;
        int _num_faces;
        bool _vertices_updated;
        bool _faces_updated;
        bool _normal_vector_updated;
        std::unique_ptr<GLfloat[]> _vertices;
        std::unique_ptr<GLfloat[]> _extracted_vertices;
        std::unique_ptr<GLfloat[]> _vertices_normal_vectors;
        std::unique_ptr<GLuint[]> _faces;
        void _update_faces(pybind11::array_t<GLuint> faces);
        void _update_vertices(pybind11::array_t<GLfloat> vertices);
        void _update(pybind11::array_t<GLfloat> vertices, pybind11::array_t<GLuint> faces);
        void _update_normal_vectors();

    public:
        ObjectData(pybind11::array_t<GLfloat> vertices, int num_vertices, pybind11::array_t<GLuint> faces, int num_faces);
        void update_faces(pybind11::array_t<GLuint> faces);
        void update_vertices(pybind11::array_t<GLfloat> vertices);
        void update(pybind11::array_t<GLfloat> vertices, pybind11::array_t<GLuint> faces);
        bool vertices_updated();
        bool faces_updated();
        bool normal_vector_updated();
        int num_vertices();
        int num_extracted_vertices();
        int num_faces();
        GLfloat* vertices();
        GLfloat* extracted_vertices();
        GLfloat* normal_vectors();
        GLuint* faces();
    };
}
}