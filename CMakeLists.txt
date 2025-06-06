cmake_minimum_required(VERSION 3.14)
project(ising_model)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "/Ox")

add_subdirectory(extern/pybind11)

pybind11_add_module(ising 
    wrapper.cpp 
    Ising.hpp
)