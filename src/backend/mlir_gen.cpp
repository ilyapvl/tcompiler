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
            case DataType::INT8:    return mlir::IntegerType::get(ctx,  8);
            case DataType::INT16:   return mlir::IntegerType::get(ctx, 16);
            case DataType::INT32:   return mlir::IntegerType::get(ctx, 32);
            case DataType::INT64:   return mlir::IntegerType::get(ctx, 64);
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

























    


    mlir::Value createConstantTensor(mlir::OpBuilder& builder, mlir::Location loc,
                                    mlir::RankedTensorType type,
                                    llvm::ArrayRef<mlir::Value> dynSizes,
                                    double value)
    {
        mlir::Value constant;
        auto elemType = type.getElementType();
        
        if (llvm::isa<mlir::FloatType>(elemType))
        {
            auto attr = builder.getFloatAttr(elemType, value);
            constant = mlir::arith::ConstantOp::create(builder, loc, elemType, attr);
        }
        
        else if (llvm::isa<mlir::IntegerType>(elemType))
        {
            auto attr = builder.getIntegerAttr(elemType, static_cast<int64_t>(value));
            constant = mlir::arith::ConstantOp::create(builder, loc, elemType, attr);
        }
        
        else
        {
            llvm::report_fatal_error("createConstantTensor: unsupported element type");
        }
        
        auto empty = mlir::tensor::EmptyOp::create(builder, loc, type, dynSizes);
        auto fill = mlir::linalg::FillOp::create(builder, loc, 
                                                mlir::ValueRange{constant}, 
                                                mlir::ValueRange{empty});
        return fill.getResult(0);
    }


    





    llvm::SmallVector<int64_t> inferBroadcastShape( llvm::ArrayRef<int64_t> shape1, llvm::ArrayRef<int64_t> shape2)
    {
        unsigned rank1 = shape1.size();
        unsigned rank2 = shape2.size();

        unsigned outRank = std::max(rank1, rank2);
        llvm::SmallVector<int64_t> outShape(outRank);

        for (unsigned i = 0; i < outRank; ++i)
        {
            int idx1 = static_cast<int>(i) - static_cast<int>(outRank - rank1);
            int idx2 = static_cast<int>(i) - static_cast<int>(outRank - rank2);

            int64_t dim1 = (idx1 >= 0) ? shape1[idx1] : 1;
            int64_t dim2 = (idx2 >= 0) ? shape2[idx2] : 1;

            if (dim1 != mlir::ShapedType::kDynamic &&
                dim2 != mlir::ShapedType::kDynamic &&
                dim1 != 1 && dim2 != 1 && dim1 != dim2)
            {
                throw std::runtime_error("Incompatible shapes for broadcasting");
            }



            if (dim1 == mlir::ShapedType::kDynamic || dim2 == mlir::ShapedType::kDynamic)
                outShape[i] = mlir::ShapedType::kDynamic;

            else
                outShape[i] = std::max(dim1, dim2);
        }

        return outShape;
    }



    mlir::AffineMap makeBroadcastMap(unsigned outRank, unsigned operandRank,
                                    llvm::ArrayRef<int64_t> operandShape,
                                    mlir::MLIRContext* ctx)
    {
        if (operandRank == 0)
        {
            return mlir::AffineMap::get(outRank, 0, {}, ctx);
        }


        llvm::SmallVector<mlir::AffineExpr> exprs;

        for (unsigned int i = 0; i < operandRank; ++i)
        {
            int iterIdx = static_cast<int>(i) - static_cast<int>(operandRank - outRank);
            if (iterIdx < 0 || (iterIdx < (int)operandRank && operandShape[iterIdx] == 1))
            {
                exprs.push_back(mlir::getAffineConstantExpr(0, ctx));
            }
            
            else
            {
                exprs.push_back(mlir::getAffineDimExpr(iterIdx, ctx));
            }
        }


        return mlir::AffineMap::get(outRank, 0, exprs, ctx);
    }






    mlir::Value broadcastToShape(mlir::OpBuilder& builder, mlir::Location loc,
                                mlir::Value input, mlir::RankedTensorType targetType,
                                llvm::ArrayRef<mlir::Value> dynSizes)
    {
        auto inputType = llvm::cast<mlir::RankedTensorType>(input.getType());
        if (inputType == targetType) return input;

        auto inferredShape = inferBroadcastShape(inputType.getShape(), targetType.getShape());
        if (inferredShape != llvm::to_vector(targetType.getShape()))
        {
            throw std::runtime_error("broadcastToShape: incompatible shapes");
        }

        unsigned targetRank = targetType.getRank();

        auto ctx = builder.getContext();
        auto inputMap = makeBroadcastMap(targetRank, inputType.getRank(), inputType.getShape(), ctx);
        
        auto identityMap = mlir::AffineMap::getMultiDimIdentityMap(targetRank, ctx);
        auto empty = mlir::tensor::EmptyOp::create(builder, loc, targetType, dynSizes);

        llvm::SmallVector<mlir::utils::IteratorType> iterators(targetRank, mlir::utils::IteratorType::parallel);
        auto generic = mlir::linalg::GenericOp::create(builder,
                                                        loc,
                                                        targetType,
                                                        mlir::ValueRange{input},
                                                        mlir::ValueRange{empty},
                                                        {inputMap, identityMap},
                                                        iterators);
        
        mlir::OpBuilder::InsertionGuard guard(builder);

        auto* block = builder.createBlock(&generic.getRegion());
        block->addArgument(inputType.getElementType(), loc);
        block->addArgument(targetType.getElementType(), loc);

        builder.setInsertionPointToStart(block);

        mlir::linalg::YieldOp::create(builder, loc, mlir::ValueRange{block->getArgument(0)});
        
        return generic.getResult(0);
    }










    llvm::SmallVector<mlir::Value> collectDynamicSizesForBinary(
        mlir::OpBuilder& builder, mlir::Location loc,
        llvm::ArrayRef<int64_t> outShape,
        llvm::ArrayRef<mlir::Value> inputs,
        llvm::ArrayRef<llvm::ArrayRef<int64_t>> inputShapes,
        llvm::ArrayRef<int64_t> inputRanks)
    {

        assert(inputs.size() == 2 && inputShapes.size() == 2 && inputRanks.size() == 2);

        unsigned outRank = outShape.size();
        llvm::SmallVector<mlir::Value> dynSizes;

        for (unsigned i = 0; i < outRank; ++i)
        {
            if (outShape[i] != mlir::ShapedType::kDynamic) continue;

            int idx0 = static_cast<int>(i) - static_cast<int>(outRank - inputRanks[0]);
            int idx1 = static_cast<int>(i) - static_cast<int>(outRank - inputRanks[1]);

            mlir::Value source;
            unsigned srcDimIdx = 0;

            if (idx0 >= 0 && inputShapes[0][idx0] == mlir::ShapedType::kDynamic)
            {
                source = inputs[0];
                srcDimIdx = static_cast<unsigned>(idx0);
            }
            
            
            else if (idx1 >= 0 && inputShapes[1][idx1] == mlir::ShapedType::kDynamic)
            {
                source = inputs[1];
                srcDimIdx = static_cast<unsigned>(idx1);
            }
            
            else if (inputRanks[0] == 0)
            {
                source = inputs[1];
                srcDimIdx = static_cast<unsigned>(idx1 >= 0 ? idx1 : 0);
            }
            
            else if (inputRanks[1] == 0)
            {
                source = inputs[0];
                srcDimIdx = static_cast<unsigned>(idx0 >= 0 ? idx0 : 0);
            }
            
            else
            {
                source = inputs[0];
                srcDimIdx = 0;
            }

            auto dim = mlir::tensor::DimOp::create(builder, loc, source, srcDimIdx);
            dynSizes.push_back(dim);
        }





        return dynSizes;
    }




    template<typename ArithOp>
    struct BodyEmitter
    {
        static void emit(mlir::OpBuilder& b, mlir::Location loc, mlir::Block* block)
        {
            auto result = ArithOp::create(b, loc, block->getArgument(0), block->getArgument(1));
            mlir::linalg::YieldOp::create(b, loc, mlir::ValueRange{result});
        }
    };

    void emitBody(OpType opType, mlir::OpBuilder& builder, mlir::Location loc, mlir::Block* block, mlir::Type elemType)
    {
        // float
        if (mlir::isa<mlir::FloatType>(elemType))
        {
            if (opType == OpType::Add)
                BodyEmitter<mlir::arith::AddFOp>::emit(builder, loc, block);

            else if (opType == OpType::Mul)
                BodyEmitter<mlir::arith::MulFOp>::emit(builder, loc, block);

            else
                throw std::runtime_error("emitBody: unsupported OpType for float");
        }

        // integer
        else if (mlir::isa<mlir::IntegerType>(elemType))
        {
            if (opType == OpType::Add)
                BodyEmitter<mlir::arith::AddIOp>::emit(builder, loc, block);

            else if (opType == OpType::Mul)
                BodyEmitter<mlir::arith::MulIOp>::emit(builder, loc, block);

            else
                throw std::runtime_error("emitBody: unsupported OpType for integer");
        }

        else
        {
            throw std::runtime_error("emitBody: unsupported element type");
        }
    }

    mlir::Value buildElementwiseGeneric(
        OpType opType,
        mlir::OpBuilder& builder,
        mlir::Location loc,
        mlir::Value lhs,
        mlir::Value rhs,
        mlir::MLIRContext* ctx)                                  
    {

        if (opType != OpType::Add && opType != OpType::Mul)
            throw std::runtime_error("buildElementwiseGeneric only supports Add and Mul");

        auto lhsType = mlir::cast<mlir::RankedTensorType>(lhs.getType());
        auto rhsType = mlir::cast<mlir::RankedTensorType>(rhs.getType());
        auto elemType = lhsType.getElementType();

        // output shape
        auto outShape = inferBroadcastShape(lhsType.getShape(), rhsType.getShape());
        auto outType = mlir::RankedTensorType::get(outShape, elemType);

        // dynamic dimensions
        auto dynSizes = collectDynamicSizesForBinary(builder, loc, outShape,
                                                    {lhs, rhs},
                                                    {lhsType.getShape(), rhsType.getShape()},
                                                    {lhsType.getRank(), rhsType.getRank()});

        auto emptyOut = mlir::tensor::EmptyOp::create(builder, loc, outType, dynSizes);

        // affine maps
        auto lhsMap = makeBroadcastMap(outShape.size(), lhsType.getRank(), lhsType.getShape(), ctx);
        auto rhsMap = makeBroadcastMap(outShape.size(), rhsType.getRank(), rhsType.getShape(), ctx);
        auto outMap = mlir::AffineMap::getMultiDimIdentityMap(outShape.size(), ctx);

        llvm::SmallVector<mlir::utils::IteratorType> iterators(outShape.size(), mlir::utils::IteratorType::parallel);

        auto generic = mlir::linalg::GenericOp::create(builder, loc, outType,
                                                    mlir::ValueRange{lhs, rhs},
                                                    mlir::ValueRange{emptyOut},
                                                    {lhsMap, rhsMap, outMap},
                                                    iterators);


        mlir::OpBuilder::InsertionGuard guard(builder);

        mlir::Block* block = builder.createBlock(&generic.getRegion());
        block->addArguments({elemType, elemType, elemType}, {loc, loc, loc});

        builder.setInsertionPointToStart(block);

        emitBody(opType, builder, loc, block, elemType);

        return generic->getResult(0);
    }







    mlir::Value MLIRGen::buildMatmulGeneric(mlir::OpBuilder& builder,
                                mlir::Location loc,
                                mlir::Value A,
                                mlir::Value B,
                                const bool transA, const bool transB) const
    {

        auto Atype = mlir::cast<mlir::RankedTensorType>(A.getType());
        auto Btype = mlir::cast<mlir::RankedTensorType>(B.getType());

        unsigned rankA = Atype.getRank(), rankB = Btype.getRank();

        if (rankA < 2 || rankB < 2)
            throw std::runtime_error("MatMul: inputs must have rank at least 2");

        // dimensions of transposed
        int64_t A0 = Atype.getDimSize(rankA - 2), A1 = Atype.getDimSize(rankA - 1);
        int64_t B0 = Btype.getDimSize(rankB - 2), B1 = Btype.getDimSize(rankB - 1);

        int64_t M   = transA ? A1 : A0;
        int64_t K   = transA ? A0 : A1;
        int64_t N   = transB ? B0 : B1;
        int64_t K2  = transB ? B1 : B0;

        if (K != K2 && K != mlir::ShapedType::kDynamic && K2 != mlir::ShapedType::kDynamic)
            throw std::runtime_error("MatMul: inner dimension mismatch");

        // batch dimensions
        llvm::SmallVector<int64_t> batchA(rankA - 2), batchB(rankB - 2);
        for (unsigned i = 0; i < rankA - 2; ++i) batchA[i] = Atype.getDimSize(i);
        for (unsigned i = 0; i < rankB - 2; ++i) batchB[i] = Btype.getDimSize(i);

        // broadcasting
        auto outBatchShape = inferBroadcastShape(batchA, batchB);
        unsigned outBatchRank = outBatchShape.size();
        llvm::SmallVector<int64_t> outShape = outBatchShape;
        outShape.push_back(M);
        outShape.push_back(N);
        auto outType = mlir::RankedTensorType::get(outShape, Atype.getElementType());

        // dynamic dimensions
        llvm::SmallVector<mlir::Value> dynOutSizes;
        unsigned totalRank = outShape.size();
        for (unsigned i = 0; i < totalRank; ++i)
        {
            if (outShape[i] != mlir::ShapedType::kDynamic) continue;

            if (i < outBatchRank)
            {
                int idxA = static_cast<int>(i) - static_cast<int>(outBatchRank - batchA.size());
                int idxB = static_cast<int>(i) - static_cast<int>(outBatchRank - batchB.size());

                if (idxA >= 0 && batchA[idxA] == mlir::ShapedType::kDynamic)
                {
                    auto dim = mlir::tensor::DimOp::create(builder, loc, A, idxA);
                    dynOutSizes.push_back(dim);
                }
                
                else if (idxB >= 0 && batchB[idxB] == mlir::ShapedType::kDynamic)
                {
                    auto dim = mlir::tensor::DimOp::create(builder, loc, B, idxB);
                    dynOutSizes.push_back(dim);
                }
                
                else
                {
                    dynOutSizes.push_back(mlir::tensor::DimOp::create(builder, loc, A, 0));
                }
            }
            
            else if (i == totalRank - 2)
            {
                // M
                unsigned srcIdx = transA ? rankA - 1 : rankA - 2;
                auto dim = mlir::tensor::DimOp::create(builder, loc, A, srcIdx);

                dynOutSizes.push_back(dim);
            }
            
            else if (i == totalRank - 1)
            {
                // N
                unsigned srcIdx = transB ? rankB - 2 : rankB - 1;
                auto dim = mlir::tensor::DimOp::create(builder, loc, B, srcIdx);

                dynOutSizes.push_back(dim);
            }
        }

        auto initOut = createConstantTensor(builder, loc, outType, dynOutSizes, 0.0);


        // in linalg.generic all dimensions must be iterated
        // so reduction dimension K is added
        unsigned iterCount = totalRank + 1; // + K (reduction)
        auto d = [&](unsigned pos) { return mlir::getAffineDimExpr(pos, &ctx_); };
        auto cst0 = mlir::getAffineConstantExpr(0, &ctx_);


        // affine map for A
        llvm::SmallVector<mlir::AffineExpr> Aexprs;
        for (unsigned i = 0; i < outBatchRank; ++i)
        {
            int idx = static_cast<int>(i) - static_cast<int>(outBatchRank - batchA.size());

            if (idx < 0 || (idx < (int)batchA.size() && batchA[idx] == 1))
                Aexprs.push_back(cst0);

            else
                Aexprs.push_back(d(i));
        }

        if (!transA)
        {
            Aexprs.push_back(d(totalRank - 2)); // M
            Aexprs.push_back(d(iterCount - 1)); // K
        }
        
        else
        {
            Aexprs.push_back(d(iterCount - 1)); // K
            Aexprs.push_back(d(totalRank - 2)); // M
        }
        auto Amap = mlir::AffineMap::get(iterCount, 0, Aexprs, &ctx_);

        // affine map for B
        llvm::SmallVector<mlir::AffineExpr> Bexprs;
        for (unsigned i = 0; i < outBatchRank; ++i)
        {
            int idx = static_cast<int>(i) - static_cast<int>(outBatchRank - batchB.size());
            if (idx < 0 || (idx < (int)batchB.size() && batchB[idx] == 1))
                Bexprs.push_back(cst0);
            else
                Bexprs.push_back(d(i));
        }

        if (!transB)
        {
            Bexprs.push_back(d(iterCount - 1)); // K
            Bexprs.push_back(d(totalRank - 1)); // N
        }
        
        else
        {
            Bexprs.push_back(d(totalRank - 1)); // N
            Bexprs.push_back(d(iterCount - 1)); // K
        }
        auto Bmap = mlir::AffineMap::get(iterCount, 0, Bexprs, &ctx_);

        // output map
        llvm::SmallVector<mlir::AffineExpr> outExprs;

        for (unsigned i = 0; i < totalRank; ++i) outExprs.push_back(d(i));
        auto outMap = mlir::AffineMap::get(iterCount, 0, outExprs, &ctx_);

        llvm::SmallVector<mlir::utils::IteratorType> iterators(iterCount, mlir::utils::IteratorType::parallel);
        iterators.back() = mlir::utils::IteratorType::reduction;

        auto generic = mlir::linalg::GenericOp::create(builder, loc, outType,
                                                    mlir::ValueRange{A, B},
                                                    mlir::ValueRange{initOut},
                                                    {Amap, Bmap, outMap},
                                                    iterators);


        
        mlir::OpBuilder::InsertionGuard guard(builder);

        auto* block = builder.createBlock(&generic.getRegion());
        auto elemType = outType.getElementType();
        block->addArgument(elemType, loc);
        block->addArgument(elemType, loc);
        block->addArgument(elemType, loc);

        builder.setInsertionPointToStart(block);

        mlir::Value mul, add;

        if (mlir::isa<mlir::FloatType>(elemType))
        {
            mul = mlir::arith::MulFOp::create(builder, loc, block->getArgument(0), block->getArgument(1));
            add = mlir::arith::AddFOp::create(builder, loc, block->getArgument(2), mul);
        }
        
        else if (mlir::isa<mlir::IntegerType>(elemType))
        {
            mul = mlir::arith::MulIOp::create(builder, loc, block->getArgument(0), block->getArgument(1));
            add = mlir::arith::AddIOp::create(builder, loc, block->getArgument(2), mul);
        }
        
        else
        {
            throw std::runtime_error("MatMul: unsupported element type");
        }
        mlir::linalg::YieldOp::create(builder, loc, add);
        
        return generic->getResult(0);
    }







    static mlir::Value makeZeroConstant(mlir::OpBuilder& builder, mlir::Location loc, mlir::Type elemType)
    {
        if (llvm::isa<mlir::FloatType>(elemType))
        {
            auto attr = builder.getFloatAttr(elemType, 0.0);
            return mlir::arith::ConstantOp::create(builder, loc, elemType, attr);
        }
        
        else if (llvm::isa<mlir::IntegerType>(elemType))
        {
            auto attr = builder.getIntegerAttr(elemType, 0);
            return mlir::arith::ConstantOp::create(builder, loc, elemType, attr);
        }
        
        else
        {
            throw std::runtime_error("makeZeroConstant: unsupported element type");
        }
    }


    mlir::Value buildReLUGeneric(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value input, mlir::MLIRContext* ctx)
    {
        auto inputType = llvm::cast<mlir::RankedTensorType>(input.getType());
        auto elemType = inputType.getElementType();
        unsigned rank = inputType.getRank();

        auto outType = inputType;

        // dynamic dimensions
        llvm::SmallVector<mlir::Value> dynSizes;
        for (unsigned i = 0; i < rank; ++i)
        {
            if (inputType.isDynamicDim(i))
            {
                dynSizes.push_back(mlir::tensor::DimOp::create(builder, loc, input, i));
            }
        }

        auto emptyOut = mlir::tensor::EmptyOp::create(builder, loc, outType, dynSizes);

        // affine maps
        auto identityMap = mlir::AffineMap::getMultiDimIdentityMap(rank, ctx);
        llvm::SmallVector<mlir::utils::IteratorType> iterators(rank, mlir::utils::IteratorType::parallel);

        auto generic = mlir::linalg::GenericOp::create(builder,
                                                        loc,
                                                        outType,
                                                        mlir::ValueRange{input},
                                                        mlir::ValueRange{emptyOut},
                                                        {identityMap, identityMap},
                                                        iterators);

        
        mlir::OpBuilder::InsertionGuard guard(builder);
        
        auto* block = builder.createBlock(&generic.getRegion());
        block->addArgument(elemType, loc);
        block->addArgument(elemType, loc);

        builder.setInsertionPointToStart(block);

        auto zero = makeZeroConstant(builder, loc, elemType);
        mlir::Value result;

        if (llvm::isa<mlir::FloatType>(elemType))
        {
            result = mlir::arith::MaximumFOp::create(builder, loc, block->getArgument(0), zero);
        }
        
        else if (llvm::isa<mlir::IntegerType>(elemType))
        {
            result = mlir::arith::MaxSIOp::create(builder, loc, block->getArgument(0), zero);
        }
        
        else
        {
            throw std::runtime_error("ReLU: unsupported element type");
        }

        mlir::linalg::YieldOp::create(builder, loc, mlir::ValueRange{result});
        

        return generic.getResult(0);
    }

















    mlir::Value buildShapeOp(mlir::OpBuilder& builder,
                            mlir::Location loc,
                            mlir::Value input,
                            int64_t start = 0,
                            int64_t end   = std::numeric_limits<int64_t>::max())
    {
        auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());
        int64_t rank   = static_cast<int64_t>(inputType.getRank());

        // normalizing
        if (start < 0) start += rank;
        if (end != std::numeric_limits<int64_t>::max() && end < 0) end += rank;
        if (end == std::numeric_limits<int64_t>::max()) end = rank;

        start = std::clamp(start, int64_t(0), rank);
        end = std::clamp(end,   int64_t(0), rank);

        int64_t numDims = std::max(int64_t(0), end - start);

        auto i64Type = builder.getI64Type();
        auto outType = mlir::RankedTensorType::get({numDims}, i64Type);

        if (numDims == 0)
        {
            return mlir::tensor::EmptyOp::create(builder, loc, outType, mlir::ValueRange{}).getResult();
        }

        llvm::SmallVector<mlir::Value> dimValues;
        dimValues.reserve(numDims);

        for (int64_t i = start; i < end; ++i)
        {
            auto dimOp  = mlir::tensor::DimOp::create(builder, loc, input, static_cast<unsigned>(i));
            auto castOp = mlir::arith::IndexCastOp::create(builder, loc, i64Type, dimOp.getResult());
            dimValues.push_back(castOp.getResult());
        }

        return mlir::tensor::FromElementsOp::create(builder, loc, outType, dimValues).getResult();
    }



    // ── Shape inference ───────────────────────────────────────────────────────────────────



    // folding Value to int64 through operation graph (arith + tensor)
    static std::optional<int64_t> foldToInt(mlir::Value v)
    {
        if (mlir::APInt val; mlir::matchPattern(v, mlir::m_ConstantInt(&val)))
            return val.getSExtValue();

        if (auto constIdx = v.getDefiningOp<mlir::arith::ConstantIndexOp>())
            return constIdx.value();

        // arith.addi both operands must be const
        if (auto add = v.getDefiningOp<mlir::arith::AddIOp>())
        {
            auto lhs = foldToInt(add.getLhs());
            auto rhs = foldToInt(add.getRhs());

            if (lhs && rhs) return *lhs + *rhs;
        }

        // arith.index_cast
        if (auto cast = v.getDefiningOp<mlir::arith::IndexCastOp>())
            return foldToInt(cast.getIn());

        // arith.index_castui
        if (auto cast = v.getDefiningOp<mlir::arith::IndexCastUIOp>())
            return foldToInt(cast.getIn());

        return std::nullopt;
    }

    // infer an in64_t value from a tensor, std::nullopt if can't
    static std::optional<int64_t> tryGetConstShapeElem(mlir::Value tensor, int64_t elemIdx)
    {
        // dense constant
        if (mlir::DenseIntElementsAttr dense; mlir::matchPattern(tensor, mlir::m_Constant(&dense)))
        {
            auto vals = dense.getValues<int64_t>();
            if (elemIdx >= 0 && elemIdx < (int64_t)vals.size())
                return vals[elemIdx];
            return std::nullopt;
        }

        // tensor.from_elements
        if (auto fe = tensor.getDefiningOp<mlir::tensor::FromElementsOp>())
        {
            auto elems = fe.getElements();
            if (elemIdx >= 0 && elemIdx < (int64_t)elems.size())
                return foldToInt(elems[elemIdx]);

            return std::nullopt;
        }

        // tensor.insert_slice
        if (auto ins = tensor.getDefiningOp<mlir::tensor::InsertSliceOp>())
        {
            int64_t offset = mlir::ShapedType::kDynamic;
            {
                auto staticOff = ins.getStaticOffsets();

                if (!staticOff.empty() && staticOff[0] != mlir::ShapedType::kDynamic)
                {
                    offset = staticOff[0];
                }

                else
                {
                    // dynamic offset
                    auto dynOff = ins.getOffsets(); // ValueRange

                    if (!dynOff.empty())
                        if (auto v = foldToInt(dynOff[0])) offset = *v;
                }
            }

            if (offset == mlir::ShapedType::kDynamic)
                return std::nullopt;

            int64_t sliceSize = mlir::ShapedType::kDynamic;
            {
                auto staticSz = ins.getStaticSizes();
                if (!staticSz.empty() &&
                    staticSz[0] != mlir::ShapedType::kDynamic)
                {
                    sliceSize = staticSz[0];
                }
            }
            if (sliceSize == mlir::ShapedType::kDynamic)
                return std::nullopt;

            if (elemIdx >= offset && elemIdx < offset + sliceSize)
                return tryGetConstShapeElem(ins.getSource(), elemIdx - offset); // element is inside the inserted

            else
                return tryGetConstShapeElem(ins.getDest(), elemIdx); // element is from the initial tensor
        }


        // tensor.concat
        if (auto concat = tensor.getDefiningOp<mlir::tensor::ConcatOp>())
        {
            int64_t offset = 0;
            for (mlir::Value inp : concat.getInputs())
            {
                auto inpType = mlir::cast<mlir::RankedTensorType>(inp.getType());
                int64_t inpSize = inpType.getDimSize(0);

                if (inpSize == mlir::ShapedType::kDynamic)
                    return std::nullopt;

                if (elemIdx < offset + inpSize)
                    return tryGetConstShapeElem(inp, elemIdx - offset);

                offset += inpSize;
            }
            return std::nullopt;
        }

        return std::nullopt;
    }



    mlir::Value buildReshapeOp(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value data, mlir::Value shape, bool allowZero = false)
    {
        auto dataType  = mlir::cast<mlir::RankedTensorType>(data.getType());
        auto shapeType = mlir::cast<mlir::RankedTensorType>(shape.getType());
        auto elemType  = dataType.getElementType();
        auto indexType = builder.getIndexType();

        if (shapeType.getRank() != 1)
            throw std::runtime_error("Reshape: shape must have exactly one dimension");

        int64_t inRank  = dataType.getRank();
        int64_t outRank = shapeType.getDimSize(0);

        if (outRank == mlir::ShapedType::kDynamic)
            throw std::runtime_error("Reshape: rank of the output tensor must be compile-time computed");


        // get consts from shape tensor
        enum class DimKind { Literal, Copy, Infer, PassThrough };

        struct DimInfo
        {
            DimKind kind;
            int64_t value = 0; // size for Literal, axis for Copy
        };

        llvm::SmallVector<DimInfo> dimInfos(outRank);
        int64_t inferIdx = -1;

        for (int64_t i = 0; i < outRank; ++i)
        {
            auto maybeVal = tryGetConstShapeElem(shape, i);

            if (!maybeVal)
            {
                // failed to fold
                dimInfos[i] = {DimKind::PassThrough, 0};
                continue;
            }

            int64_t v = *maybeVal;

            if (v == -1)
            {
                if (inferIdx >= 0)
                    throw std::runtime_error("Reshape: no more than one '-1' dimension");

                inferIdx = i;

                dimInfos[i] = {DimKind::Infer, -1};
            }
            
            else if (v == 0 && !allowZero)
            {
                if (i >= inRank)
                    throw std::runtime_error("Reshape: '0' dim index bigger than input rank");

                dimInfos[i] = {DimKind::Copy, i};
            }

            else if (v >= 0)
            {
                dimInfos[i] = {DimKind::Literal, v};
            }

            else
            {
                throw std::runtime_error("Reshape: invalid value " + std::to_string(v));
            }
        }


        // static calculation of output shape
        llvm::SmallVector<bool> inputConsumed(inRank, false);
        int64_t literalProduct = 1;

        for (int64_t i = 0; i < outRank; ++i)
        {
            const auto& d = dimInfos[i];

            if (d.kind == DimKind::Copy)
                inputConsumed[d.value] = true;

            else if (d.kind == DimKind::Literal)
                literalProduct *= d.value;
        }

        // all consumed dimensions
        for (int64_t i = 0; i < outRank; ++i)
        {
            if (dimInfos[i].kind == DimKind::PassThrough && i < inRank)
                inputConsumed[i] = true;
        }

        int64_t unconsumedStaticProduct = 1;
        bool hasUncancelledDynamic   = false;

        for (int64_t d = 0; d < inRank; ++d)
        {
            if (inputConsumed[d]) continue;

            int64_t s = dataType.getDimSize(d);

            if (s == mlir::ShapedType::kDynamic)
                hasUncancelledDynamic = true;

            else
                unconsumedStaticProduct *= s;
        }

        // output type
        llvm::SmallVector<int64_t> outShape(outRank, mlir::ShapedType::kDynamic);

        for (int64_t i = 0; i < outRank; ++i)
        {
            switch (dimInfos[i].kind)
            {

                case DimKind::Literal:
                    outShape[i] = dimInfos[i].value;
                    break;

                case DimKind::Copy:
                    outShape[i] = dataType.getDimSize(dimInfos[i].value);
                    break;

                case DimKind::Infer:
                    if (!hasUncancelledDynamic)
                        outShape[i] = (literalProduct != 0) ? (unconsumedStaticProduct / literalProduct) : 0;
                    break;

                case DimKind::PassThrough:
                    outShape[i] = mlir::ShapedType::kDynamic;

                    break;
            }
        }

        auto outType = mlir::RankedTensorType::get(outShape, elemType);


        // build index-values for tensor.reshape
        auto shapeIdxType = mlir::RankedTensorType::get({outRank}, indexType);
        llvm::SmallVector<mlir::Value> idxVals(outRank, nullptr);

        auto buildTotalElems = [&]() -> mlir::Value
        {
            mlir::Value total = mlir::arith::ConstantIndexOp::create(builder, loc, 1).getResult();

            for (int64_t d = 0; d < inRank; ++d)
            {
                mlir::Value dv = mlir::tensor::DimOp::create(builder, loc, data, static_cast<unsigned>(d)).getResult();
                total = mlir::arith::MulIOp::create(builder, loc, total, dv).getResult();
            }

            return total;
        };

        // first pass: Literal, Copy, PassThrough
        mlir::Value dynKnownProd = mlir::arith::ConstantIndexOp::create(builder, loc, 1).getResult();

        for (int64_t i = 0; i < outRank; ++i)
        {
            const auto& d = dimInfos[i];

            switch (d.kind)
            {
                case DimKind::Literal:
                    idxVals[i] = mlir::arith::ConstantIndexOp::create(builder, loc, d.value).getResult();
                    break;

                case DimKind::Copy:
                    idxVals[i] = mlir::tensor::DimOp::create(builder, loc, data, static_cast<unsigned>(d.value)).getResult();
                    break;

                case DimKind::PassThrough:
                {
                    // from shape tensor
                    mlir::Value idx = mlir::arith::ConstantIndexOp::create(builder, loc, i).getResult();

                    mlir::Value elem = mlir::tensor::ExtractOp::create(builder, loc, shape,mlir::ValueRange{idx}).getResult();

                    idxVals[i] = mlir::arith::IndexCastOp::create(builder, loc, indexType, elem).getResult();
                    break;
                }

                case DimKind::Infer:
                    break;
            }

            if (d.kind != DimKind::Infer)
            {
                dynKnownProd = mlir::arith::MulIOp::create(
                                builder, loc, dynKnownProd, idxVals[i]).getResult();
            }
        }

        // second pass
        if (inferIdx >= 0)
        {
            if (!hasUncancelledDynamic)
            {
                // static
                idxVals[inferIdx] = mlir::arith::ConstantIndexOp::create(builder, loc, outShape[inferIdx]).getResult();
            }
            
            else
            {
                // dynamic: totalElems / dynLnownProd
                mlir::Value total = buildTotalElems();
                idxVals[inferIdx] = mlir::arith::DivUIOp::create(builder, loc, total, dynKnownProd).getResult();
            }
        }

        // build tensor.reshape
        mlir::Value shapeIdxTensor = mlir::tensor::FromElementsOp::create(builder, loc, shapeIdxType, idxVals).getResult();

        return mlir::tensor::ReshapeOp::create(builder, loc, outType, data, shapeIdxTensor).getResult();
    }










    mlir::Value buildConcatOp(mlir::OpBuilder& builder, mlir::Location loc, llvm::ArrayRef<mlir::Value> inputs, int64_t axis)
    {
        auto firstType = mlir::cast<mlir::RankedTensorType>(inputs[0].getType());
        int64_t rank = firstType.getRank();

        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank)
            throw std::runtime_error("Concat: invalid axis");

        for (auto inp : inputs)
        {
            auto t = mlir::cast<mlir::RankedTensorType>(inp.getType());
            if (t.getRank() != rank)
                throw std::runtime_error("Concat: all inputs must share same rank");
        }

        // tensor.concat
        auto concatOp = mlir::tensor::ConcatOp::create(builder, loc, axis, inputs);
        return concatOp.getResult();
    }












    
    void MLIRGen::processNode(mlir::OpBuilder& builder,
                            const Node&      node,
                            ValueMap&        vmap,
                            const Graph&     graph) const
    {
        auto loc = builder.getUnknownLoc();
        auto nodeType = node.getOpType();

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
        if (nodeType == OpType::Add || nodeType == OpType::Mul)
        {
            auto lhs = resolve(node.getInputs()[0]);
            auto rhs = resolve(node.getInputs()[1]);


            //always use generic build //TODO - straight generation when same shapes


            auto result = buildElementwiseGeneric(nodeType, builder, loc, lhs, rhs, &ctx_);
            vmap[node.getOutputs()[0]] = result;
        
            return;
        }








        // ── MatMul ────────────────────────────────────────────────────────────────
        if (nodeType == OpType::MatMul)
        {
            auto A = resolve(node.getInputs()[0]);
            auto B = resolve(node.getInputs()[1]);

            //TODO - if 2D or 3D use linalg.batch_matmul
            auto result = buildMatmulGeneric(builder, loc, A, B, false, false);
            vmap[node.getOutputs()[0]] = result;
            return;
        }




        // ── Gemm ──────────────────────────────────────────────────────────────────
        if (nodeType == OpType::Gemm)
        {
            auto A = resolve(node.getInputs()[0]);
            auto B = resolve(node.getInputs()[1]);

            mlir::Value C = nullptr;

            if (node.getInputs().size() >= 3 && !node.getInputs()[2].empty())
                C = resolve(node.getInputs()[2]);

            float alpha = 1.0f, beta = 1.0f;
            bool transA = false, transB = false;

            if (node.hasAttribute("alpha"))     alpha   = node.getAttribute("alpha").asFloat();
            if (node.hasAttribute("beta"))      beta    = node.getAttribute("beta").asFloat();
            if (node.hasAttribute("transA"))    transA  = node.getAttribute("transA").asInt() != 0;
            if (node.hasAttribute("transB"))    transB  = node.getAttribute("transB").asInt() != 0;

            // A[] * B[]
            mlir::Value result = buildMatmulGeneric(builder, loc, A, B, transA, transB);
            auto resultType = llvm::cast<mlir::RankedTensorType>(result.getType());

            // dynamic dimensions
            llvm::SmallVector<mlir::Value> dynSizes;
            for (unsigned i = 0; i < resultType.getRank(); ++i)
            {
                if (resultType.isDynamicDim(i))
                {
                    dynSizes.push_back(mlir::tensor::DimOp::create(builder, loc, result, i));
                }
            }

            // scale by alpha
            if (std::abs(alpha - 1.0f) > 1e-6)
            {
                auto alphaTensor = createConstantTensor(builder, loc, resultType, dynSizes, alpha);
                result = buildElementwiseGeneric(OpType::Mul, builder, loc, result, alphaTensor, &ctx_);
            }

            // + С * beta
            if (C)
            {
                mlir::Value Cval = broadcastToShape(builder, loc, C, resultType, dynSizes);

                // scaling C
                if (std::abs(beta - 1.0f) > 1e-6)
                {
                    auto betaTensor = createConstantTensor(builder, loc, resultType, dynSizes, beta);
                    Cval = buildElementwiseGeneric(OpType::Mul, builder, loc, Cval, betaTensor, &ctx_);
                }

                // result + C
                result = buildElementwiseGeneric(OpType::Add, builder, loc, result, Cval, &ctx_);
            }

            vmap[node.getOutputs()[0]] = result;
            return;
        }
        

        // ── Relu ──────────────────────────────────────────────────────────────────
        if (nodeType == OpType::Relu)
        {
            auto input = resolve(node.getInputs()[0]);
            auto result = buildReLUGeneric(builder, loc, input, &ctx_);
            vmap[node.getOutputs()[0]] = result;

            return;
        }
        


        // ── Shape ─────────────────────────────────────────────────────────────────
        if (nodeType == OpType::Shape)
        {
            auto input = resolve(node.getInputs()[0]);

            int64_t start = 0;
            int64_t end = std::numeric_limits<int64_t>::max();

            if (node.hasAttribute("start"))
                start = node.getAttribute("start").asInt();

            if (node.hasAttribute("end"))
                end = node.getAttribute("end").asInt();

            auto result = buildShapeOp(builder, loc, input, start, end);
            vmap[node.getOutputs()[0]] = result;

            return;
        }


        // ── Reshape ───────────────────────────────────────────────────────────────
        if (nodeType == OpType::Reshape)
        {
            auto data = resolve(node.getInputs()[0]);
            auto shape = resolve(node.getInputs()[1]);

            bool allowZero = false;

            if (node.hasAttribute("allowzero"))
                allowZero = node.getAttribute("allowzero").asInt() != 0;

            auto result = buildReshapeOp(builder, loc, data, shape, allowZero);
            vmap[node.getOutputs()[0]] = result;

            return;
        }


        // ── Concat ────────────────────────────────────────────────────────────────
        if (nodeType == OpType::Concat)
        {
            int64_t axis = node.getAttribute("axis").asInt();

            llvm::SmallVector<mlir::Value> inputs;
            inputs.reserve(node.getInputs().size());

            for (const auto& inputName : node.getInputs())
                inputs.push_back(resolve(inputName));

            auto result = buildConcatOp(builder, loc, inputs, axis);
            vmap[node.getOutputs()[0]] = result;

            return;
        }


        // ── Conv2d ────────────────────────────────────────────────────────────────
        if (nodeType == OpType::Conv)
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

        



        if (opts.print_mlir)
        {
            llvm::outs() << "\nMLIR before optimization\n";
            module.print(llvm::outs());
            llvm::outs() << "\n";
        }

        if (mlir::failed(mlir::verify(module)))
            throw std::runtime_error("MLIR module verification failed");



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
