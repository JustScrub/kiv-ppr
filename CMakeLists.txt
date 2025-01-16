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

# project sources
file(GLOB prj_sources src/*.cpp)

# Add sources to the project
add_executable(
    KIV_PPR_MLADY
    ${prj_sources}
)

# Link OpenCL
target_link_libraries(KIV_PPR_MLADY OpenCL::OpenCL)