# TCompiler

A frontend of a tensor compiler for loading ONNX models, representing them as a computational graph, and visualizing the graph using GraphViz DOT format

## Features

- Load ONNX models (`.onnx`)
- Internal graph representation with:
  - Operations (`Add`, `MatMul`, `Conv`, `Gemm`, `Relu`, etc.)
  - Tensors (data type, shape, raw data for constants/weights)
  - Attributes (full support for all ONNX attribute types: float, int, string, tensor, graph, lists, etc.)
- Topological sorting of graph nodes (Kahn’s algorithm)
- Export to GraphViz DOT format
- Generate PNG images via `dot` (GraphViz required)
- Testing using GoogleTest.

## Dependencies

- **C++20** compiler
- **CMake** 3.20 or higher
- **Protobuf** (libprotobuf) – used for ONNX parsing
- **GoogleTest** – for tests (automatically fetched via CMake)
- **GraphViz** (optional) – for PNG export (`dot` executable)

On Ubuntu/Debian:
```
sudo apt install libprotobuf-dev protobuf-compiler graphviz
```

On macOS with Homebrew:
```
brew install protobuf graphviz
```



## Building

Clone the repository:

```
git clone https://github.com/ilyapvl/tcompiler.git
cd tcompiler
```

Create a build directory and configure:

```
mkdir build && cd build
cmake ..
```

File `onnx.proto3` will be downloaded

Build the project:

```
make
```

This will build C++ from `proto` using protobuf compiler. Files `onnx.pb.h` and `onnx.pb.cc` will be created and put in `PROTO_GEN_DIR`.

Then, these files will be produced:
- `libtc_lib.a` – static library
- `tcompiler` – executable

## Usage

### Command line

```
./tcompiler <model.onnx> [output.dot] [output.png]
```

- `<model.onnx>` – input ONNX model file.
- `[output.dot]` – optional DOT file path (default: `graph.dot`)
- `[output.png]` – optional PNG file path (GraphViz required)

The program also prints debug info:
- Model metadata (version, producer, etc.)
- Graph summary (number of nodes, tensors, inputs, outputs, operation breakdown)
- Topological order of nodes

Example of graph visualizing:

<p align="center">
  <img src="docs/images/example_graph.svg" alt="graph" width="600">
  <br>
</p>

## Project Structure

```
.
├── CMakeLists.txt
├── include/
│   ├── graph/
│   │   ├── attribute.hpp
│   │   ├── graph.hpp
│   │   ├── node.hpp
│   │   └── tensor.hpp
│   ├── frontend/
│   │   └── onnx_loader.hpp
│   └── visualization/
│       └── dot_exporter.hpp
├── src/
│   ├── graph/
│   │   ├── attribute.cpp
│   │   ├── graph.cpp
│   │   ├── node.cpp
│   │   └── tensor.cpp
│   ├── frontend/
│   │   └── onnx_loader.cpp
│   ├── visualization/
│   │   └── dot_exporter.cpp
│   └── main.cpp
├── tests/
│   ├── attribute_test.cpp
│   ├── tensor_test.cpp
│   ├── node_test.cpp
│   ├── graph_test.cpp
│   ├── onnx_loader_test.cpp
│   ├── dot_exporter_test.cpp
│   └── test_main.cpp
└── README.md
```

## Testing

Run all tests:

```
cd build
ctest
```

Or run the test executable directly:

```
./tests/tc_tests
```

## Limitations

- External data (weights stored in separate files) are not yet supported. Only single-file models
- Sparse tensors are stored as raw data without full interpretation.
