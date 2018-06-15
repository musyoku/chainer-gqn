CXX = g++
INCLUDE = `python3 -m pybind11 --includes`
FLAGS = -O3 -DNDEBUG -Wall -Wformat -march=native -shared -std=c++14 -fPIC
SOURCES = ./core/*.cpp
EXTENSION = `python3-config --extension-suffix`

UNAME := $(shell uname -s)
ifeq ($(UNAME), Darwin)
	FLAGS += -undefined dynamic_lookup
endif

make: 
	$(CXX) $(FLAGS) $(INCLUDE) $(SOURCES) $(FLAGS) pybind/rasterizer.cpp -o ../generative_query_network/python/gqn/rasterizer/rasterizer_cpu$(EXTENSION)