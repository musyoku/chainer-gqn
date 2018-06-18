CXX = g++
INCLUDE = -I./external `pkg-config --cflags glfw3` `python3 -m pybind11 --includes`
LDFLAGS = `pkg-config --static --libs glfw3`
FLAGS = -O3 -DNDEBUG -Wall -Wformat -march=native -shared -std=c++14 -fPIC -pthread
SOURCES = ./external/gl3w/*.c \
		  ./core/base/*.cpp \
		  ./core/camera/*.cpp \
		  ./core/renderer/*.cpp \
		  ./core/scene/*.cpp
EXTENSION = `python3-config --extension-suffix`
OUTPUT = ../generative_query_network/gqn

UNAME := $(shell uname -s)
ifeq ($(UNAME), Linux)
	LDFLAGS += -lGL
endif
ifeq ($(UNAME), Darwin)
	LDFLAGS += -framework OpenGL -undefined dynamic_lookup
endif

make: 
	$(CXX) $(FLAGS) $(INCLUDE) $(SOURCES) $(LDFLAGS) pybind/three.cpp -o $(OUTPUT)/three$(EXTENSION)