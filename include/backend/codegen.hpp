#ifndef MLIR_GEN_HPP
#define MLIR_GEN_HPP

#include "graph/graph.hpp"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "llvm/IR/Module.h"

#include <filesystem>
#include <string>
#include <unordered_map>

namespace tc
{


    struct CodeGenOptions
    {
        bool print_mlir      = false; 
        bool print_mlir_opt  = false;
        bool optimize        = true;

        bool lower_to_llvm   = true;
        bool print_llvm_ir   = false;

        bool emit_asm        = false;
        bool emit_obj        = false;


        std::string target_triple = "arm64_bare_metal";
        std::string cpu           = "generic";
        std::string features      = "";


        std::filesystem::path mlir_out;
        std::filesystem::path llvm_ir_out;
        std::filesystem::path asm_out;
        std::filesystem::path obj_out;
    };


    CodeGenOptions parseMLIROptions(int argc, char* argv[]);

    void printMLIRHelp();


    class CodeGen
    {
    public:
        CodeGen(mlir::MLIRContext& mlir_ctx, llvm::LLVMContext& llvm_ctx);


        int generate(const Graph& graph, const CodeGenOptions& opts = {},
                        const std::string& mlir_out = "", const std::string& asm_out = "");


        




        

    private:
        mlir::MLIRContext& mlir_ctx_;
        llvm::LLVMContext& llvm_ctx_;

        using ValueMap = std::unordered_map<std::string, mlir::Value>;

        void processNode(
            mlir::OpBuilder& builder,
            const Node&      node,
            ValueMap&        vmap,
            const Graph&     graph) const;


        [[nodiscard]] mlir::RankedTensorType tensorTypeOf(const Tensor& t) const;

        [[nodiscard]] mlir::RankedTensorType makeTensorType(DataType dt, const TensorShape& shape) const;

        [[nodiscard]] mlir::Value makeWeightConstant(mlir::OpBuilder& builder, mlir::Location loc, const Tensor& weight) const;
        
        
        void lowerToLLVM(mlir::ModuleOp mod);
        static void runOptPipeline(mlir::ModuleOp mod);
        std::unique_ptr<llvm::Module> translateToLLVMIR(mlir::ModuleOp mod, llvm::raw_ostream &os);

        mlir::OwningOpRef<mlir::ModuleOp> runLoweringPipeline(mlir::ModuleOp mod);

        void emitObject(llvm::Module *llvmModule, const std::string &filename, const CodeGenOptions& opts);
        
    };

} // namespace tc

#endif // MLIR_GEN_HPP
