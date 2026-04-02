cmake_minimum_required(VERSION 3.20)
project(TensorCompiler VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# abseil is required for protobuf code
find_package(absl CONFIG REQUIRED)
find_package(Protobuf REQUIRED)
message(STATUS "Protobuf version  : ${Protobuf_VERSION}")
message(STATUS "Protobuf includes : ${Protobuf_INCLUDE_DIRS}")
message(STATUS "protoc executable : ${Protobuf_PROTOC_EXECUTABLE}")




# downloading onnx.proto

set(TC_THIRD_PARTY_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
set(ONNX_PROTO_DIR      "${TC_THIRD_PARTY_DIR}/onnx")
set(ONNX_PROTO_FILE     "${ONNX_PROTO_DIR}/onnx.proto")

file(MAKE_DIRECTORY "${ONNX_PROTO_DIR}")

if(NOT EXISTS "${ONNX_PROTO_FILE}")
    message(STATUS "Downloading onnx.proto (ONNX v1.16.0) ...")
    file(DOWNLOAD
        "https://raw.githubusercontent.com/onnx/onnx/v1.16.0/onnx/onnx.proto3"
        "${ONNX_PROTO_FILE}"
        STATUS   _dl_status
        SHOW_PROGRESS
        TLS_VERIFY ON
    )
    list(GET _dl_status 0 _dl_code)
    list(GET _dl_status 1 _dl_msg)
    if(NOT _dl_code EQUAL 0)
        message(FATAL_ERROR
            "Cannot download onnx.proto: ${_dl_msg}\n"
            "Place the file manually:\n"
            "  curl -L https://raw.githubusercontent.com/onnx/onnx/v1.16.0/onnx/onnx.proto3"
            " -o ${ONNX_PROTO_FILE}")
    endif()
    message(STATUS "onnx.proto downloaded to ${ONNX_PROTO_FILE}")
endif()



# generating .pb.h / .pb.cc file from onnx.proto

set(PROTO_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/proto_gen")
file(MAKE_DIRECTORY "${PROTO_GEN_DIR}")

set(ONNX_PB_H  "${PROTO_GEN_DIR}/onnx.pb.h")
set(ONNX_PB_CC "${PROTO_GEN_DIR}/onnx.pb.cc")

add_custom_command(
    OUTPUT  "${ONNX_PB_H}" "${ONNX_PB_CC}"
    COMMAND "${Protobuf_PROTOC_EXECUTABLE}"
            "--proto_path=${ONNX_PROTO_DIR}"
            "--cpp_out=${PROTO_GEN_DIR}"
            "${ONNX_PROTO_FILE}"
    DEPENDS "${ONNX_PROTO_FILE}"
    COMMENT "Generating C++ from onnx.proto"
    VERBATIM
)

add_custom_target(generate_onnx_proto DEPENDS "${ONNX_PB_H}" "${ONNX_PB_CC}")



# tc_lib

add_library(tc_lib STATIC
    "${ONNX_PB_CC}"
    src/graph/attribute.cpp
    src/graph/tensor.cpp
    src/graph/node.cpp
    src/graph/graph.cpp
    src/frontend/onnx_loader.cpp
    src/visualization/dot_exporter.cpp
)

add_dependencies(tc_lib generate_onnx_proto)

target_include_directories(tc_lib PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${PROTO_GEN_DIR}"
    "${Protobuf_INCLUDE_DIRS}"
)

target_link_libraries(tc_lib PUBLIC
    protobuf::libprotobuf
    absl::log
    absl::log_internal_check_op

    absl::log_internal_voidify
    absl::base
    absl::strings
)

set_source_files_properties("${ONNX_PB_CC}"
    PROPERTIES COMPILE_FLAGS "-w"
)



# exec

add_executable(tcompiler src/main.cpp)
target_link_libraries(tcompiler PRIVATE tc_lib)

enable_testing()
add_subdirectory(tests)
