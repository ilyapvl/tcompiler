#include "frontend/onnx_loader.hpp"
#include "visualization/dot_exporter.hpp"
#include "backend/mlir_gen.hpp"

#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/InitLLVM.h"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <stdexcept>

static void printUsage(const char* prog)
{
    std::cout << "Usage: " << prog
              << " <model.onnx> [graph.dot] [graph.png] [mlir-options...]\n\n";
    tc::printMLIRHelp();
}

int main(int argc, char* argv[])
{
    llvm::InitLLVM init_llvm(argc, argv);
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();

    if (argc < 2)
    {
        printUsage(argv[0]);
        return 1;
    }




    const std::filesystem::path onnx_path = argv[1];
    const std::filesystem::path dot_path  = (argc >= 3) ? argv[2] : "graph.dot";
    const std::filesystem::path png_path  = (argc >= 4) ? argv[3] : "";

    tc::MLIRGenOptions mlir_opts = tc::parseMLIROptions(argc, argv);

    try
    {
        auto info = tc::OnnxLoader::readModelInfo(onnx_path);
        std::cout << "ONNX Model Info\n"
                  << "  Version        : " << info.ir_version        << "\n"
                  << "  Producer       : " << info.producer_name     << " " << info.producer_version << "\n"
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

        /*
        tc::DotExporter exporter;
        exporter.exportToFile(*graph, dot_path);
        std::cout << "DOT file written: " << dot_path << "\n";

        if (!png_path.empty())
        {
            exporter.exportToPng(*graph, png_path);
            std::cout << "PNG file written: " << png_path << "\n";
        }
        */

        std::cout << "\nMLIR code generation\n";

        std::cout << "target triple : "
                  << (mlir_opts.target_triple.empty()
                      ? "(host)" : mlir_opts.target_triple) << "\n";

        std::cout << "cpu          : "
                  << (mlir_opts.cpu.empty()
                      ? "(host)" : mlir_opts.cpu) << "\n";

        std::cout << "features      : "
                  << (mlir_opts.features.empty()
                      ? "(none)" : mlir_opts.features) << "\n";

        std::cout << "optimized      : "
                  << (mlir_opts.optimize ? "yes" : "no") << "\n\n";

        mlir::MLIRContext ctx;
        tc::MLIRGen gen(ctx);

        auto mlir_module = gen.generate(*graph, mlir_opts);

        std::cout << "\ndone\n";
    }
    
    catch (const std::exception& e)
    {
        std::cerr << "err: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
