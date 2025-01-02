cmake_minimum_required(VERSION 3.21)
project(KIV_PPR_MLADY)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Set compiler specific flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Wall /arch:AVX2")

# Find OpenCL
#find_package(OpenCL REQUIRED)

# Include directories
include_directories(src)
#include_directories(lib)
include_directories("include")

# Drawing library files from https://github.com/SmartCGMS/common.git
# set(
#     drawing_lib
#     lib/drawing/Drawing.cpp
#     lib/drawing/Drawing.h
#     lib/drawing/IRenderer.h
#     lib/drawing/SVGRenderer.cpp
#     lib/drawing/SVGRenderer.h
# )

file(GLOB prj_sources src/*.cpp)

# Add executable
add_executable(
    KIV_PPR_MLADY
    ${prj_sources}
#    ${drawing_lib}
#    src/my_drawing/svg_generator.h
#    src/my_drawing/svg_generator.cpp
)

# Link OpenCL
#target_link_libraries(KIV_PPR_MLADY OpenCL::OpenCL)