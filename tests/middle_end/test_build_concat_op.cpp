#include <gtest/gtest.h>
#include "middle_end/mlir_builders.hpp"
#include "test_dimensions.hpp"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace tc;
using namespace mlir;
using namespace tc::test;

class ConcatTestBase : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ctx.loadDialect<arith::ArithDialect, linalg::LinalgDialect, tensor::TensorDialect, func::FuncDialect>();
    }

    MLIRContext ctx;
    OpBuilder builder = OpBuilder(&ctx);
    Location loc = UnknownLoc::get(&ctx);
};

using ConcatParam = ConcatCase;

class ConcatOpTest : public ConcatTestBase, public ::testing::WithParamInterface<ConcatParam> {};

TEST_P(ConcatOpTest, Concat)
{
    const auto& param = GetParam();
    size_t numInputs = param.input_shapes.size();
    llvm::SmallVector<mlir::RankedTensorType> inputTypes;
    
    for (const auto& shape : param.input_shapes)
    {
        inputTypes.push_back(RankedTensorType::get(llvm::ArrayRef<int64_t>(shape), builder.getF32Type()));
    }
    auto outType = RankedTensorType::get(llvm::ArrayRef<int64_t>(param.expected_output_shape), builder.getF32Type());

    auto module = ModuleOp::create(loc);
    llvm::SmallVector<mlir::Type> argTypes(inputTypes.begin(), inputTypes.end());
    auto funcType = builder.getFunctionType(argTypes, {outType});
    auto func = func::FuncOp::create(loc, "test", funcType);
    func.addEntryBlock();
    builder.setInsertionPointToStart(&func.getBody().front());

    llvm::SmallVector<mlir::Value> inputs;
    for (unsigned i = 0; i < numInputs; ++i)
    {
        inputs.push_back(func.getArgument(i));
    }

    Value result = buildConcatOp(builder, loc, inputs, param.axis);
    ASSERT_TRUE(result);
    EXPECT_EQ(result.getType(), outType);

    auto concatOp = result.getDefiningOp<tensor::ConcatOp>();
    ASSERT_TRUE(concatOp);
    EXPECT_EQ(concatOp.getInputs().size(), numInputs);

    auto axisAttr = concatOp->getAttrOfType<IntegerAttr>("axis");
    if (axisAttr)
    {
        int64_t rank = inputTypes[0].getRank();
        int64_t normalizedAxis = param.axis;
        if (normalizedAxis < 0) normalizedAxis += rank;
        EXPECT_EQ(axisAttr.getInt(), normalizedAxis);
    }

    func::ReturnOp::create(builder, loc, result);
    module.push_back(func);
    EXPECT_TRUE(succeeded(verify(module)));
}

INSTANTIATE_TEST_SUITE_P(
    ConcatTests,
    ConcatOpTest,
    testing::ValuesIn(concatTests)
);
