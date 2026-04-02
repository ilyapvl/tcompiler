cmake_minimum_required(VERSION 3.20)
project(TensorCompilerTests)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# GoogleTest
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    DOWNLOAD_EXTRACT_TIMESTAMP true

)
FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
    FetchContent_Populate(googletest)

    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
endif()

add_executable(tc_tests
    test_main.cpp
    test_tensor.cpp
    test_attribute.cpp
    test_node.cpp
    test_graph.cpp
    test_onnx_loader.cpp
)

target_link_libraries(tc_tests
    tc_lib
    gtest_main
    gtest
)

target_include_directories(tc_tests PRIVATE
    ${googletest_SOURCE_DIR}/googletest/include
)

include(GoogleTest)
gtest_discover_tests(tc_tests)
