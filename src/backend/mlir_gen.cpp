#include "backend/mlir_gen.hpp"

// ── MLIR ───────────────────────────────────────────────────────────────────
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"

// ── MLIR IR ───────────────────────────────────────────────────────────────────
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/AffineMap.h"

// ── MLIR passes ───────────────────────────────────────────────────────────────────
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"

// ── conversions ───────────────────────────────────────────────────────────────────
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"

// ── buffering ───────────────────────────────────────────────────────────────────
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"

// ── MLIR to LLVM ───────────────────────────────────────────────────────────────────
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

// ── LLVM ───────────────────────────────────────────────────────────────────
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cstring>

namespace tc
{

    static void registerAllDialects(mlir::MLIRContext& ctx)
    {
        ctx.loadDialect<
            mlir::arith::ArithDialect,
            mlir::func::FuncDialect,
            mlir::linalg::LinalgDialect,
            mlir::memref::MemRefDialect,
            mlir::scf::SCFDialect,
            mlir::tensor::TensorDialect,
            mlir::affine::AffineDialect,
            mlir::math::MathDialect,
            mlir::bufferization::BufferizationDialect,
            mlir::LLVM::LLVMDialect
        >();
    }

    // tc::DataType to mlir::Type
    static mlir::Type mlirElemType(DataType dt, mlir::MLIRContext* ctx)
    {
        switch (dt)
        {
            case DataType::FLOAT:   return mlir::Float32Type::get(ctx);
            case DataType::DOUBLE:  return mlir::Float64Type::get(ctx);
            case DataType::FLOAT16: return mlir::Float16Type::get(ctx);
            case DataType::INT8:    return mlir::IntegerType::get(ctx,  8, mlir::IntegerType::Signed);
            case DataType::INT16:   return mlir::IntegerType::get(ctx, 16, mlir::IntegerType::Signed);
            case DataType::INT32:   return mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signed);
            case DataType::INT64:   return mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Signed);
            case DataType::UINT8:   return mlir::IntegerType::get(ctx,  8, mlir::IntegerType::Unsigned);
            case DataType::BOOL:    return mlir::IntegerType::get(ctx,  1);
            default:                return mlir::Float32Type::get(ctx);
        }
    }

    MLIRGen::MLIRGen(mlir::MLIRContext& ctx) : ctx_(ctx)
    {
        registerAllDialects(ctx_);
        mlir::registerBuiltinDialectTranslation(ctx_);
        mlir::registerLLVMDialectTranslation(ctx_);
    }

    //ranked tensor from data
    mlir::RankedTensorType MLIRGen::makeTensorType(DataType dt, const TensorShape& shape) const
    {
        auto elem = mlirElemType(dt, &ctx_);
        llvm::SmallVector<int64_t> dims;
        dims.reserve(shape.dims.size());

        for (auto d : shape.dims)
            dims.push_back(d < 0 ? mlir::ShapedType::kDynamic : d);

        return mlir::RankedTensorType::get(dims, elem);
    }

    mlir::RankedTensorType MLIRGen::tensorTypeOf(const Tensor& t) const
    {
        return makeTensorType(t.getDtype(), t.getShape());
    }

    mlir::Value MLIRGen::makeWeightConstant(mlir::OpBuilder& builder,
                                            mlir::Location   loc,
                                            const Tensor&    w) const
    {
        auto rtt = tensorTypeOf(w);

        if (w.getDtype() == DataType::FLOAT && w.hasValidData())
        {
            auto span = w.getDataAs<float>();
            llvm::SmallVector<float> vals(span.begin(), span.end());
            auto attr = mlir::DenseElementsAttr::get(rtt, llvm::ArrayRef<float>(vals));
            return mlir::arith::ConstantOp::create(builder, loc, rtt, attr);
        }

        if (w.getDtype() == DataType::INT64 && w.hasValidData())
        {
            auto span = w.getDataAs<int64_t>();
            llvm::SmallVector<int64_t> vals(span.begin(), span.end());
            auto attr = mlir::DenseIntElementsAttr::get(rtt, llvm::ArrayRef<int64_t>(vals));
            return mlir::arith::ConstantOp::create(builder, loc, rtt, attr);
        }

        //TODO - other data types

        // else zero tensor
        // std::cout << "No data for ConstantOp" << std::endl;
        auto zero = builder.getZeroAttr(rtt.getElementType());
        auto attr = mlir::DenseElementsAttr::get(rtt, zero);
        return mlir::arith::ConstantOp::create(builder, loc, rtt, attr);
    }

    void MLIRGen::runOptPipeline(mlir::ModuleOp mod)
    {
        //TODO - optimize
        return;
    }

    void MLIRGen::runLoweringPipeline(mlir::ModuleOp mod)
    {
        //FIXME - cannot bufferize
        return;

        mlir::PassManager pm(mod->getContext());

        mlir::bufferization::OneShotBufferizePassOptions opts;
        opts.bufferizeFunctionBoundaries = true;
        opts.allowUnknownOps = true;
        pm.addPass(mlir::bufferization::createOneShotBufferizePass(opts));

        pm.addPass(mlir::createCanonicalizerPass());

        pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertLinalgToLoopsPass());
        pm.addPass(mlir::createLowerAffinePass());
        pm.addPass(mlir::createSCFToControlFlowPass());
        pm.addPass(mlir::createArithToLLVMConversionPass());
        pm.addPass(mlir::createConvertFuncToLLVMPass());

        pm.addPass(mlir::createConvertToLLVMPass());

        pm.addPass(mlir::createReconcileUnrealizedCastsPass());

        if (mlir::failed(pm.run(mod)))
            throw std::runtime_error("MLIR lowering failed");
    }

    std::string MLIRGen::toLLVMIR(mlir::ModuleOp mod, const MLIRGenOptions& opts)
    {
        //TODO - to LLVM IR

        return std::string("not implemented");
    }

    void MLIRGen::emitCode(mlir::ModuleOp mod, const MLIRGenOptions& opts)
    {
        if (!opts.emit_asm && !opts.emit_obj) return;

        //TODO - to asm code

        return;
    }

    void MLIRGen::processNode(mlir::OpBuilder& builder,
                            const Node&      node,
                            ValueMap&        vmap,
                            const Graph&     graph) const
    {
        auto loc = builder.getUnknownLoc();

        // get data by tensor name
        auto resolve = [&](const std::string& name) -> mlir::Value
        {
            if (name.empty())
                throw std::runtime_error("Empty tensor name");

            auto it = vmap.find(name);
            if (it != vmap.end()) return it->second;

            auto opt = graph.findTensor(name);
            if (opt && (*opt)->hasData())
            {
                auto val = makeWeightConstant(builder, loc, **opt);
                vmap[name] = val;
                return val;
            }
            throw std::runtime_error(
                "Cannot resolve tensor '" + name +
                "' in node '" + node.getName() + "'");
        };

        // creates a tensor which has shape same as source
        auto createEmpyWithShapeLike = [&](mlir::Value source) -> mlir::Value
        {
            auto type = mlir::cast<mlir::RankedTensorType>(source.getType());
            llvm::SmallVector<mlir::Value> dynSizes;
            for (unsigned i = 0; i < type.getRank(); ++i)
            {
                if (type.isDynamicDim(i))
                {
                    auto dimVal = mlir::tensor::DimOp::create(builder, loc, source, i);
                    dynSizes.push_back(dimVal);
                }
            }
            return mlir::tensor::EmptyOp::create(builder, loc, type, dynSizes);
        };


        // ── Add / Mul ───────────────────────────────────────────────────────────────────
        if (node.getOpType() == OpType::Add || node.getOpType() == OpType::Mul)
        {
            auto lhs = resolve(node.getInputs()[0]);
            auto rhs = resolve(node.getInputs()[1]);
            auto lhsType = mlir::cast<mlir::RankedTensorType>(lhs.getType());
            auto rhsType = mlir::cast<mlir::RankedTensorType>(rhs.getType());

            // Add / Mul creation is actually the same except the operation name
            // so that lambda is used to reduce code amount
            auto createBodyOp = [&](mlir::OpBuilder &b, mlir::Location loc, mlir::Value l, mlir::Value r) -> mlir::Value
            {
                if (node.getOpType() == OpType::Add)
                {
                    if (mlir::isa<mlir::FloatType>(l.getType()))
                        return mlir::arith::AddFOp::create(b, loc, l, r);
                    else
                        return mlir::arith::AddIOp::create(b, loc, l, r);
                }
                else
                { // Mul
                    if (mlir::isa<mlir::FloatType>(l.getType()))
                        return mlir::arith::MulFOp::create(b, loc, l, r);
                    else
                        return mlir::arith::MulIOp::create(b, loc, l, r);
                }
            };

            // use simple linalg if dimensions are equal
            if (lhsType == rhsType)
            {
                auto out = createEmpyWithShapeLike(lhs);
                mlir::Operation *op = nullptr;
                if (node.getOpType() == OpType::Add)
                {
                    op = mlir::linalg::AddOp::create(builder, loc,
                                                    mlir::TypeRange{lhsType},
                                                    mlir::ValueRange{lhs, rhs},
                                                    mlir::ValueRange{out});
                }
                else
                {
                    op = mlir::linalg::MulOp::create(builder, loc,
                                                    mlir::TypeRange{lhsType},
                                                    mlir::ValueRange{lhs, rhs},
                                                    mlir::ValueRange{out});
                }
                vmap[node.getOutputs()[0]] = op->getResult(0);
                return;
            }

            // dimensions are not equal
            // ── broadcasting via linalg.generic ──────────────────────────────
            unsigned lhsRank = lhsType.getRank();
            unsigned rhsRank = rhsType.getRank();
            unsigned outRank = std::max(lhsRank, rhsRank);
            auto lhsShape = lhsType.getShape();
            auto rhsShape = rhsType.getShape();
            mlir::Type elemType = lhsType.getElementType();

            // inferring output shape
            llvm::SmallVector<int64_t> outShape(outRank);
            for (unsigned i = 0; i < outRank; ++i)
            {
                int lhsIdx = static_cast<int>(i) - static_cast<int>(outRank - lhsRank);
                int rhsIdx = static_cast<int>(i) - static_cast<int>(outRank - rhsRank);
                int64_t lhsDim = (lhsIdx >= 0) ? lhsShape[lhsIdx] : 1;
                int64_t rhsDim = (rhsIdx >= 0) ? rhsShape[rhsIdx] : 1;
                if (lhsDim == mlir::ShapedType::kDynamic || rhsDim == mlir::ShapedType::kDynamic)
                    outShape[i] = mlir::ShapedType::kDynamic;
                else
                    outShape[i] = std::max(lhsDim, rhsDim);
            }
            auto outType = mlir::RankedTensorType::get(outShape, elemType);

            // build affine maps
            auto buildMap = [&](unsigned operandRank, llvm::ArrayRef<int64_t> shape) -> mlir::AffineMap
            {
                if (operandRank == 0)
                {
                    // scalar
                    return mlir::AffineMap::get(outRank, 0, {}, &ctx_);
                }
                llvm::SmallVector<mlir::AffineExpr> exprs;
                for (unsigned i = 0; i < outRank; ++i)
                {
                    int idx = static_cast<int>(i) - static_cast<int>(outRank - operandRank);
                    if (idx < 0 || (idx < (int)operandRank && shape[idx] == 1))
                        exprs.push_back(mlir::getAffineConstantExpr(0, &ctx_));
                    else
                        exprs.push_back(mlir::getAffineDimExpr(i, &ctx_));
                }
                return mlir::AffineMap::get(outRank, 0, exprs, &ctx_);
            };
            llvm::SmallVector<mlir::AffineMap> maps =
            {
                buildMap(lhsRank, lhsShape),
                buildMap(rhsRank, rhsShape),
                mlir::AffineMap::getMultiDimIdentityMap(outRank, &ctx_)
            };
            llvm::SmallVector<mlir::utils::IteratorType> iterators(
                outRank, mlir::utils::IteratorType::parallel);

            // output ttensor
            llvm::SmallVector<mlir::Value> dynOutSizes;
            for (unsigned i = 0; i < outRank; ++i)
            {
                if (outShape[i] == mlir::ShapedType::kDynamic)
                {
                    mlir::Value source;
                    int lhsIdx = static_cast<int>(i) - static_cast<int>(outRank - lhsRank);
                    int rhsIdx = static_cast<int>(i) - static_cast<int>(outRank - rhsRank);
                    if (lhsIdx >= 0 && lhsShape[lhsIdx] == mlir::ShapedType::kDynamic)
                        source = lhs;
                    else if (rhsIdx >= 0 && rhsShape[rhsIdx] == mlir::ShapedType::kDynamic)
                        source = rhs;
                    else
                        source = lhs;
                    auto idx = mlir::arith::ConstantIndexOp::create(builder, loc, i);
                    auto dim = mlir::tensor::DimOp::create(builder, loc, source, idx);
                    dynOutSizes.push_back(dim);
                }
            }
            auto out = mlir::tensor::EmptyOp::create(builder, loc, outType, dynOutSizes);

            // build genericOp
            auto generic = mlir::linalg::GenericOp::create(builder, loc,
                                                        mlir::TypeRange{outType},
                                                        mlir::ValueRange{lhs, rhs},
                                                        mlir::ValueRange{out},
                                                        maps, iterators);

            // filling block
            mlir::OpBuilder::InsertionGuard guard(builder);
            mlir::Block *block = builder.createBlock(&generic.getRegion());
            block->addArgument(elemType, loc);
            block->addArgument(elemType, loc);
            block->addArgument(elemType, loc);
            builder.setInsertionPointToStart(block);
            auto result = createBodyOp(builder, loc, block->getArgument(0), block->getArgument(1));
            mlir::linalg::YieldOp::create(builder, loc, result);

            vmap[node.getOutputs()[0]] = generic->getResult(0);
            return;
        }




        // ── MatMul ────────────────────────────────────────────────────────────────
        if (node.getOpType() == OpType::MatMul)
        {
            //TODO - MatMul

            return;
        }

        // ── Gemm ──────────────────────────────────────────────────────────────────
        if (node.getOpType() == OpType::Gemm)
        {
            //TODO - Gemm
            return;
        }

        // ── Relu ──────────────────────────────────────────────────────────────────
        if (node.getOpType() == OpType::Relu)
        {
            //TODO - Relu
            return;
        }

        // ── Conv2d ────────────────────────────────────────────────────────────────
        if (node.getOpType() == OpType::Conv)
        {
            //TODO - Conv
            return;
        }





        std::cerr << "unsupported op '"
                << node.getOpStr() << "' (node: " << node.getName()
                << "), skipping\n";
    }



    mlir::OwningOpRef<mlir::ModuleOp> MLIRGen::generate(const Graph& graph, const MLIRGenOptions& opts)
    {
        auto module = mlir::ModuleOp::create(
            mlir::UnknownLoc::get(&ctx_), graph.getName());

        mlir::OpBuilder builder(&ctx_);
        builder.setInsertionPointToEnd(module.getBody());

        llvm::SmallVector<mlir::Type> arg_types;
        for (const auto& inp_name : graph.getInputs())
        {
            auto opt = graph.findTensor(inp_name);
            if (!opt)
                throw std::runtime_error("Input tensor not found: " + inp_name);
            arg_types.push_back(tensorTypeOf(**opt));
        }

        llvm::SmallVector<mlir::Type> ret_types;
        for (const auto& out_name : graph.getOutputs())
        {
            auto opt = graph.findTensor(out_name);
            if (!opt)
                throw std::runtime_error("Output tensor not found: " + out_name);
            ret_types.push_back(tensorTypeOf(**opt));
        }

        auto func_type = builder.getFunctionType(arg_types, ret_types);
        auto func = mlir::func::FuncOp::create(
            builder.getUnknownLoc(), graph.getName(), func_type);
        module.push_back(func);
        func.addEntryBlock();

        builder.setInsertionPointToStart(&func.getBody().front());




        ValueMap vmap;
        for (size_t i = 0; i < graph.getInputs().size(); ++i)
            vmap[graph.getInputs()[i]] = func.getArgument(static_cast<unsigned>(i));

        auto sorted = graph.topologicalSort();
        for (const auto& node : sorted)
            processNode(builder, *node, vmap, graph);


        llvm::SmallVector<mlir::Value> ret_vals;
        for (const auto& out_name : graph.getOutputs())
        {
            auto it = vmap.find(out_name);
            if (it == vmap.end())
                throw std::runtime_error(
                    "Output tensor '" + out_name + "' not computed");
            ret_vals.push_back(it->second);
        }


        mlir::func::ReturnOp::create(builder, builder.getUnknownLoc(), ret_vals);

        if (mlir::failed(mlir::verify(module)))
            throw std::runtime_error("MLIR module verification failed");



        if (opts.print_mlir)
        {
            llvm::outs() << "\nMLIR before optimization\n";
            module.print(llvm::outs());
            llvm::outs() << "\n";
        }



        if (!opts.mlir_out.empty())
        {
            std::error_code ec;
            llvm::raw_fd_ostream ofs(opts.mlir_out.string(), ec);
            if (!ec) module.print(ofs);
        }



        if (opts.optimize)
            runOptPipeline(module);


        return module;
    }




    void printMLIRHelp()
    {
        std::cout <<
                R"(
                MLIR / LLVM codegen options:
                --print-mlir            Print MLIR before optimization
                --print-mlir-opt        Print MLIR after optimization
                --no-optimize           Disable MLIR optimization passes

                --mlir-out=<path>       Write MLIR to file
    )";
    }





















    MLIRGenOptions parseMLIROptions(int argc, char* argv[])
    {
        MLIRGenOptions opts;

        auto startsWith = [](const std::string& s, const std::string& prefix)
        {
            return s.rfind(prefix, 0) == 0;
        };

        auto getValue = [](const std::string& s, const std::string& prefix)
        {
            return s.substr(prefix.size());
        };






        for (int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];

            if (arg == "--print-mlir")         { opts.print_mlir     = true; continue; }
            if (arg == "--print-mlir-opt")     { opts.print_mlir_opt = true; continue; }
            if (arg == "--no-optimize")        { opts.optimize       = false; continue; }

            if (startsWith(arg, "--target-triple="))
            { opts.target_triple = getValue(arg, "--target-triple="); continue; }

            if (startsWith(arg, "--cpu="))
            { opts.cpu = getValue(arg, "--cpu="); continue; }

            if (startsWith(arg, "--features="))
            { opts.features = getValue(arg, "--features="); continue; }

            if (startsWith(arg, "--mlir-out="))
            { opts.mlir_out = getValue(arg, "--mlir-out="); continue; }

        }

        return opts;
    }

} // namespace tc
