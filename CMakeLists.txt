cmake_minimum_required(VERSION 3.21)
project(KIV_PPR_MLADY)
set(CMAKE_CXX_STANDARD 17)

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /arch:AVX2")

# Find OpenCL
find_package(OpenCL REQUIRED)

# Include directories
include_directories(src)
include_directories("include")

# SVG drawing library stolen from https://github.com/SmartCGMS/common/tree/devel/scgms/utils/drawing
file(GLOB svg_sources include/drawing/*.cpp include/drawing/*.h)
# project sources
file(GLOB prj_sources src/*.cpp)

# Add sources to the project
add_executable(
    KIV_PPR_MLADY
    ${prj_sources}
    ${svg_sources}
#    src/my_drawing/svg_generator.h
#    src/my_drawing/svg_generator.cpp
)

# Link OpenCL
target_link_libraries(KIV_PPR_MLADY OpenCL::OpenCL)