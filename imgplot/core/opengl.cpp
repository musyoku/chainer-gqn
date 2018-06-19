#include "opengl.h"
#include <iostream>
#include <memory>

namespace imgplot {
namespace opengl {
    // シェーダオブジェクトのコンパイル結果を表示する
    GLboolean validate_shader(GLuint shader, const char* shader_name)
    {
        // コンパイル結果を取得する
        GLint status;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
        if (status == GL_FALSE) {
            std::cout << "Failed to compile " << shader_name << std::endl;
        }

        // シェーダのコンパイル時のログの長さを取得する
        GLsizei message_length;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &message_length);

        if (message_length > 1) {
            // シェーダのコンパイル時のログの内容を取得する
            GLchar* message = new GLchar[message_length + 1];
            GLsizei length;
            glGetShaderInfoLog(shader, message_length, &length, message);
            std::cout << message << std::endl;
            delete[] message;
        }

        return (GLboolean)status;
    }
    GLboolean validate_program(GLuint program)
    {
        // リンク結果を取得する
        GLint status;
        glGetProgramiv(program, GL_LINK_STATUS, &status);
        if (status == GL_FALSE)
            std::cout << "Failed to link program" << std::endl;

        // シェーダのリンク時のログの長さを取得する
        GLsizei message_length;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &message_length);

        if (message_length > 1) {
            // シェーダのリンク時のログの内容を取得する
            GLchar* message = new GLchar[message_length];
            GLsizei length;
            glGetProgramInfoLog(program, message_length, &length, message);
            std::cout << message << std::endl;
            delete[] message;
        }

        return (GLboolean)status;
    }
    GLuint create_program(const char* vertex_shader, const char* fragment_shader)
    {
        GLuint vobj = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vobj, 1, &vertex_shader, NULL);
        glCompileShader(vobj);
        validate_shader(vobj, "vertex shader");

        GLuint fobj = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fobj, 1, &fragment_shader, NULL);
        glCompileShader(fobj);
        validate_shader(fobj, "fragment shader");

        GLuint program = glCreateProgram();
        glAttachShader(program, vobj);
        glDeleteShader(vobj);
        glAttachShader(program, fobj);
        glDeleteShader(fobj);

        glLinkProgram(program);
        validate_program(program);
        glUseProgram(program);

        return program;
    }
}
}