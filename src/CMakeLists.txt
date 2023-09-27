cmake_minimum_required(VERSION 3.0.0)
project(face VERSION 0.1.0 LANGUAGES C CXX)

find_package(OpenCV REQUIRED)

file(GLOB SOURCES "src/*.cpp")
add_executable(face ${SOURCES})

target_link_libraries(face PRIVATE ${OpenCV_LIBS})