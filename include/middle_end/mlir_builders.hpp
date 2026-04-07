#ifndef MLIR_BUILDERS_HPP
#define MLIR_BUILDERS_HPP

#include "graph/graph.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include <optional>

namespace tc
{


    mlir::Value createConstantTensor(mlir::OpBuilder& builder,
                                    mlir::Location loc,
                                    mlir::RankedTensorType type,
                                    llvm::ArrayRef<mlir::Value> dynSizes,
                                    double value);

    llvm::SmallVector<int64_t> inferBroadcastShape( llvm::ArrayRef<int64_t> shape1, llvm::ArrayRef<int64_t> shape2);

    mlir::AffineMap makeBroadcastMap(unsigned outRank,
                                    unsigned operandRank,
                                    llvm::ArrayRef<int64_t> operandShape,
                                    mlir::MLIRContext* ctx);


    llvm::SmallVector<mlir::Value> collectDynamicSizesForBinary(
        mlir::OpBuilder& builder, mlir::Location loc,
        llvm::ArrayRef<int64_t> outShape,
        llvm::ArrayRef<mlir::Value> inputs,
        llvm::ArrayRef<llvm::ArrayRef<int64_t>> inputShapes,
        llvm::ArrayRef<int64_t> inputRanks);

    mlir::Value buildElementwiseGeneric(
        OpType opType,
        mlir::OpBuilder& builder,
        mlir::Location loc,
        mlir::Value lhs,
        mlir::Value rhs,
        mlir::MLIRContext* ctx);

    mlir::Value buildMatmulGeneric(mlir::OpBuilder& builder,
                                mlir::Location loc,
                                mlir::Value A,
                                mlir::Value B,
                                const bool transA,
                                const bool transB,
                                mlir::MLIRContext* ctx);

    mlir::Value buildReLUGeneric(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value input, mlir::MLIRContext* ctx);

    mlir::Value buildShapeOp(mlir::OpBuilder& builder,
                            mlir::Location loc,
                            mlir::Value input,
                            int64_t start,
                            int64_t end);

    mlir::Value buildReshapeOp(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value data, mlir::Value shape, bool allowZero = false);

    mlir::Value buildConcatOp(mlir::OpBuilder& builder, mlir::Location loc, llvm::ArrayRef<mlir::Value> inputs, int64_t axis);

    mlir::Value buildConv2dOp(mlir::OpBuilder& builder,
                                mlir::Location loc,
                                mlir::Value input,
                                mlir::Value weights,
                                std::optional<mlir::Value> bias,
                                llvm::ArrayRef<int64_t> kernelShape,
                                llvm::ArrayRef<int64_t> strides,
                                llvm::ArrayRef<int64_t> pads,
                                llvm::ArrayRef<int64_t> dilations,
                                int64_t group,
                                llvm::StringRef autoPad,
                                mlir::MLIRContext* ctx);

    mlir::Value makeZeroConstant(mlir::OpBuilder& builder, mlir::Location loc, mlir::Type elemType);
    std::optional<int64_t> foldToInt(mlir::Value v);
    std::optional<int64_t> tryGetConstShapeElem(mlir::Value tensor, int64_t elemIdx);

} // namespace tc

#endif // MLIR_BUILDERS_HPP
