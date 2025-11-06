cmake_minimum_required(VERSION 3.15)
project(WeightliftingAnalysis VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math")

# Set architecture to match Python
set(CMAKE_OSX_ARCHITECTURES "x86_64" CACHE STRING "Build architecture" FORCE)

# Find Python first
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Find pybind11 using Python
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Find required packages
find_package(Eigen3 REQUIRED)
find_package(pybind11 REQUIRED CONFIG PATHS ${pybind11_DIR})

# Include directories
include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp
)

# Source files
set(SOURCES
    src/cpp/dynamics/inverse_dynamics.cpp
    src/cpp/dynamics/multibody_model.cpp
    src/cpp/dynamics/parameter_estimator.cpp
    src/cpp/physics/bar_bending.cpp
    src/cpp/bindings.cpp
)

# Create Python module
pybind11_add_module(weightanalysis_cpp ${SOURCES})

# Link libraries
target_link_libraries(weightanalysis_cpp PRIVATE Eigen3::Eigen)

# Install target
install(TARGETS weightanalysis_cpp DESTINATION src/python)
