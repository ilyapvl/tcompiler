cmake_minimum_required(VERSION 3.15)
project(tcompiler VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

add_library(graph STATIC src/graph.cpp src/reader.cpp)
target_include_directories(graph PUBLIC include)
target_link_libraries(graph PUBLIC)

add_executable(main src/main.cpp)
target_link_libraries(main
    graph
)

enable_testing()

