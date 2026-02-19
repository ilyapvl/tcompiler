cmake_minimum_required(VERSION 3.15)
project(tcompiler VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_library(graph STATIC src/graph.cpp src/reader.cpp)
target_include_directories(graph PUBLIC include)
target_link_libraries(graph PUBLIC)

add_executable(main src/main.cpp)
target_link_libraries(main
    graph
)





add_executable(test_reader tests/test_reader.cpp)
target_link_libraries(test_reader
    gtest_main
    graph
)
target_include_directories(test_reader PRIVATE include)

add_test(NAME ProtoReader COMMAND test_reader)


add_executable(test_parse_tensor tests/test_parse_tensor.cpp)
target_link_libraries(test_parse_tensor
    gtest_main
    graph
)
target_include_directories(test_parse_tensor PRIVATE include)

add_test(NAME ParseTensor COMMAND test_parse_tensor)


add_executable(test_parse_valueinfo tests/test_parse_valueinfo.cpp)
target_link_libraries(test_parse_valueinfo
    gtest_main
    graph
)
target_include_directories(test_parse_valueinfo PRIVATE include)

add_test(NAME ParseValueInfo COMMAND test_parse_valueinfo)


add_executable(test_parse_attribute tests/test_parse_attribute.cpp)
target_link_libraries(test_parse_attribute
    gtest_main
    graph
)
target_include_directories(test_parse_attribute PRIVATE include)

add_test(NAME ParseAttribute COMMAND test_parse_attribute)


add_executable(test_parse_node tests/test_parse_node.cpp)
target_link_libraries(test_parse_node
    gtest_main
    graph
)
target_include_directories(test_parse_node PRIVATE include)

add_test(NAME ParseNode COMMAND test_parse_node)



add_executable(test_load_models tests/test_load_models.cpp)
target_link_libraries(test_load_models
    gtest_main
    graph
)
target_include_directories(test_load_models PRIVATE include)

add_test(NAME LoadModels COMMAND test_load_models)
