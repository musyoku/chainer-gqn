#pragma once
#include <gl3w/gl3w.h>
#include <glfw/glfw3.h>

namespace imgplot {
namespace renderer {
    class ImageRenderer {
    private:
        GLuint _program;
        GLuint _attribute_uv;
        GLuint _attribute_position;
        GLuint _uniform_image;
        GLuint _uniform_mat;
        GLuint _vao;
        GLuint _vbo_vertices;
        GLuint _vbo_faces;
        GLuint _vbo_uv;
        GLuint _texture_id;
        GLuint _texture_unit;
        GLuint _sampler_id;

    public:
        ImageRenderer();
        void set_data(GLubyte* data, int width, int height);
        void render(GLfloat scale_x, GLfloat scale_y);
    };
}
}