#ifndef MLIR_GEN_HPP
#define MLIR_GEN_HPP

#include "graph/graph.hpp"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

#include <filesystem>
#include <string>
#include <unordered_map>

namespace tc
{


    struct MLIRGenOptions
    {
        bool print_mlir      = false; 
        bool print_mlir_opt  = false;
        bool optimize        = true;

        bool lower_to_llvm   = true;
        bool print_llvm_ir   = false;

        bool emit_asm        = false;
        bool emit_obj        = false;


        std::string target_triple = "";
        std::string cpu           = "";
        std::string features      = "";


        std::filesystem::path mlir_out;
        std::filesystem::path llvm_ir_out;
        std::filesystem::path asm_out;
        std::filesystem::path obj_out;
    };


    MLIRGenOptions parseMLIROptions(int argc, char* argv[]);

    void printMLIRHelp();


    class MLIRGen
    {
    public:
        explicit MLIRGen(mlir::MLIRContext& ctx);


        [[nodiscard]] mlir::OwningOpRef<mlir::ModuleOp> generate(const Graph& graph, const MLIRGenOptions& opts = {});


        




        static std::string toLLVMIR(mlir::ModuleOp mod, const MLIRGenOptions& opts);


        static void emitCode(mlir::ModuleOp mod, const MLIRGenOptions& opts);

    private:
        mlir::MLIRContext& ctx_;

        using ValueMap = std::unordered_map<std::string, mlir::Value>;

        void processNode(
            mlir::OpBuilder& builder,
            const Node&      node,
            ValueMap&        vmap,
            const Graph&     graph) const;


        [[nodiscard]] mlir::RankedTensorType tensorTypeOf(const Tensor& t) const;

        [[nodiscard]] mlir::RankedTensorType makeTensorType(DataType dt, const TensorShape& shape) const;

        [[nodiscard]] mlir::Value makeWeightConstant(mlir::OpBuilder& builder, mlir::Location loc, const Tensor& weight) const;
        

        static void runOptPipeline(mlir::ModuleOp mod);
        static void runLoweringPipeline(mlir::ModuleOp mod);
    };

} // namespace tc

#endif // MLIR_GEN_HPP
