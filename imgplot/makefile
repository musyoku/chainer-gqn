CXX = g++
INCLUDE = -I./external $(shell pkg-config --cflags glfw3) $(shell python3 -m pybind11 --includes)
LDFLAGS = $(shell pkg-config --static --libs glfw3) -shared -pthread
CXXFLAGS = -O3 -DNDEBUG -Wall -Wformat -march=native -std=c++14 -fPIC
SOURCES = 	$(wildcard ./external/gl3w/*.c) \
			$(wildcard ./core/base/*.cpp) \
			$(wildcard ./core/view/*.cpp) \
			$(wildcard ./core/data/*.cpp) \
			$(wildcard ./core/renderer/*.cpp) \
			$(wildcard ./core/*.cpp) \
			$(wildcard ./pybind/*.cpp)
OBJS = $(patsubst %.c,%.o,$(patsubst %.cpp,%.o,$(SOURCES)))
EXTENSION = $(shell python3-config --extension-suffix)
OUTPUT = ../generative_query_network/gqn

UNAME := $(shell uname -s)
ifeq ($(UNAME), Linux)
	LDFLAGS += -lGL
endif
ifeq ($(UNAME), Darwin)
	LDFLAGS += -framework OpenGL -undefined dynamic_lookup
endif

$(OUTPUT)/imgplot$(EXTENSION): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $(OBJS)

.c.o:
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

.cpp.o:
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@
