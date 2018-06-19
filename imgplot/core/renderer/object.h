#pragma once
#include <gl3w/gl3w.h>
#include <glfw/glfw3.h>
#include <glm/glm.hpp>
#include <memory>

namespace imgplot {
namespace renderer {
    class ObjectRenderer {
    private:
        GLuint _program;
        GLuint _attribute_position;
        GLuint _attribute_normal_vector;
        GLuint _uniform_pvm_mat;
        GLuint _uniform_light_mat;
        GLuint _vao;
        GLuint _vbo_vertices;
        GLuint _vbo_normal_vectors;
        GLuint _vbo_faces;
        GLuint _vbo_uv;
        int _num_faces;
        glm::mat4 _view_mat;
        glm::mat4 _model_mat;
        glm::mat4 _light_mat;
        glm::vec3 _camera_location;

    public:
        ObjectRenderer(GLfloat* vertices, int num_vertices, GLuint* faces, int num_faces);
        ~ObjectRenderer();
        void update_faces(GLuint* faces, int num_faces);
        void update_vertices(GLfloat* vertices, int num_vertices);
        void update_normal_vectors(GLfloat* normal_vectors, int num_vertices);
        void render(GLfloat aspect_ratio);
        void zoom_in();
        void zoom_out();
        void rotate_camera(double diff_x, double diff_y);
    };
}
}