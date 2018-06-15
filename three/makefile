CXX = g++
INCLUDE = -I./external `pkg-config --cflags glfw3` `python3 -m pybind11 --includes`
FLAGS = -O3 -DNDEBUG -Wall -Wformat -march=native -shared -std=c++14 -fPIC -pthread
SOURCES = ./core/base/*.cpp \
		  ./core/camera/*.cpp \
		  ./core/renderer/*.cpp \
		  ./core/scene/*.cpp
EXTENSION = `python3-config --extension-suffix`
OUTPUT = ../generative_query_network/python/gqn

make: 
	$(CXX) $(FLAGS) $(INCLUDE) $(SOURCES) pybind/renderer.cpp -o $(OUTPUT)/renderer$(EXTENSION)
	$(CXX) $(FLAGS) $(INCLUDE) $(SOURCES) pybind/three.cpp -o $(OUTPUT)/three$(EXTENSION)