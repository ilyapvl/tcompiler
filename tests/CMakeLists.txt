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
    frontend/test_tensor.cpp
    frontend/test_attribute.cpp
    frontend/test_node.cpp
    frontend/test_graph.cpp
    frontend/test_onnx_loader.cpp

    middle_end/test_infer_broadcast_shape.cpp
    middle_end/test_make_broadcast_map.cpp
    middle_end/test_build_elementwise_generic.cpp
    middle_end/test_build_relu_generic.cpp
    middle_end/test_build_matmul_generic.cpp
    middle_end/test_build_shape_op.cpp
    middle_end/test_build_reshape_op.cpp
    middle_end/test_build_concat_op.cpp
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
