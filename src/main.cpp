#include "frontend/onnx_loader.hpp"
#include "visualization/dot_exporter.hpp"
#include "backend/codegen.hpp"

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
              << " <model.onnx> [codegen-options...]\n\n";
    tc::printMLIRHelp();
}

int main(int argc, char* argv[])
{
    llvm::InitLLVM init_llvm(argc, argv);
    

    if (argc < 2)
    {
        printUsage(argv[0]);
        return 1;
    }




    const std::filesystem::path onnx_path = argv[1];
    const std::filesystem::path dot_path  = "graph.dot";

    tc::CodeGenOptions mlir_opts = tc::parseMLIROptions(argc, argv);

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

        
        tc::DotExporter exporter;
        exporter.exportToFile(*graph, dot_path);
        std::cout << "DOT file written: " << dot_path << "\n";







        std::cout << "\nMLIR generation\n";

        
        mlir::MLIRContext mlir_ctx;
        llvm::LLVMContext llvm_ctx;

        std::string mlir_out = "";


        for (int i = 1; i < argc; ++i)
        {
            if (std::string(argv[i]) == "--mlir-out" && i + 1 < argc)
            {
                mlir_out = argv[++i];
            }
        }


        std::string asm_out = "";

        for (int i = 1; i < argc; ++i)
        {
            if (std::string(argv[i]) == "-o" && i + 1 < argc)
            {
                asm_out = argv[++i];
            }
        }


        tc::CodeGen gen(mlir_ctx, llvm_ctx);
        gen.generate(*graph, mlir_opts, mlir_out, asm_out);


        


    }
    
    catch (const std::exception& e)
    {
        std::cerr << "err: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
