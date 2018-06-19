CXX = g++
INCLUDE = -I./external `pkg-config --cflags glfw3`
LDFLAGS = `pkg-config --static --libs glfw3` `python3 -m pybind11 --includes`
FLAGS = -O3 -DNDEBUG -Wall -Wformat -march=native -shared -std=c++14 -fPIC -pthread
SOURCES = 	./external/gl3w/*.c \
			./core/base/*.cpp \
			./core/view/*.cpp \
			./core/data/*.cpp \
			./core/renderer/*.cpp \
			./core/*.cpp \
			./pybind/*.cpp
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
	$(CXX) $(FLAGS) $(INCLUDE) $(SOURCES) $(LDFLAGS) -o $(OUTPUT)/imgplot$(EXTENSION)