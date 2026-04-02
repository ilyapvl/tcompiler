#include "frontend/onnx_loader.hpp"
#include "visualization/dot_exporter.hpp"

#include <iostream>
#include <filesystem>
#include <stdexcept>

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        return 1;
    }

    const std::filesystem::path onnx_path = argv[1];
    const std::filesystem::path dot_path  = (argc >= 3) ? argv[2] : "graph.dot";
    const std::filesystem::path png_path  = (argc >= 4) ? argv[3] : "";

    try
    {
        auto info = tc::OnnxLoader::readModelInfo(onnx_path);
        std::cout << "ONNX Model Info\n"
                  << "  Version        : " << info.ir_version       << "\n"
                  << "  Producer       : " << info.producer_name     << " "
                                           << info.producer_version  << "\n"
                  << "  Domain         : " << info.domain            << "\n"
                  << "  Model version  : " << info.model_version     << "\n"
                  << "  Graph name     : " << info.graph_name        << "\n\n";

        tc::OnnxLoader loader;
        auto graph = loader.load(onnx_path);

        std::cout << graph->summary() << "\n";




        auto sorted = graph->topologicalSort();
        std::cout << "Topologically sorted (" << sorted.size() << " nodes)\n";
        for (const auto& n : sorted)
            std::cout << "  " << n->getOpStr() << "  " << n->getName() << "\n";
        std::cout << "\n";




        tc::DotExporter exporter;
        exporter.exportToFile(*graph, dot_path);
        std::cout << "DOT file written: " << dot_path << "\n";

        if (!png_path.empty())
        {
            exporter.exportToPng(*graph, png_path);
            std::cout << "PNG file written: " << png_path << "\n";
        }
    }
    
    catch (const std::exception& e)
    {
        std::cerr << "err: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
