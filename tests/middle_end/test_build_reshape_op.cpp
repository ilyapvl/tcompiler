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

class ReshapeTestBase : public ::testing::Test
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

using ReshapeParam = ReshapeCase;

class ReshapeOpTest : public ReshapeTestBase, public ::testing::WithParamInterface<ReshapeParam> {};

TEST_P(ReshapeOpTest, Reshape)
{
    const auto& param = GetParam();
    auto inputType = RankedTensorType::get(llvm::ArrayRef<int64_t>(param.input_shape), builder.getF32Type());
    auto outType = RankedTensorType::get(llvm::ArrayRef<int64_t>(param.expected_output_shape), builder.getF32Type());

    auto module = ModuleOp::create(loc);
    auto funcType = builder.getFunctionType({inputType}, {outType});
    auto func = func::FuncOp::create(loc, "test", funcType);
    func.addEntryBlock();
    builder.setInsertionPointToStart(&func.getBody().front());
    Value input = func.getArgument(0);

    auto shapeType = RankedTensorType::get({static_cast<int64_t>(param.shape_tensor.size())}, builder.getI64Type());
    llvm::SmallVector<Value> shapeValues;

    for (int64_t val : param.shape_tensor)
    {
        shapeValues.push_back(arith::ConstantOp::create(builder, loc, builder.getI64Type(), builder.getI64IntegerAttr(val)));
    }

    Value shape = tensor::FromElementsOp::create(builder, loc, shapeType, shapeValues);

    Value result = buildReshapeOp(builder, loc, input, shape, param.allowzero);
    ASSERT_TRUE(result);
    EXPECT_EQ(result.getType(), outType);




    auto reshapeOp = result.getDefiningOp<tensor::ReshapeOp>();
    ASSERT_TRUE(reshapeOp);
    EXPECT_EQ(reshapeOp.getSource(), input);



    func::ReturnOp::create(builder, loc, result);
    module.push_back(func);
    EXPECT_TRUE(succeeded(verify(module)));
}

INSTANTIATE_TEST_SUITE_P(
    ReshapeTests,
    ReshapeOpTest,
    testing::ValuesIn(reshapeTests)
);








class ReshapeErrorTest : public ReshapeTestBase, public ::testing::WithParamInterface<ReshapeCase> {};

TEST_P(ReshapeErrorTest, Throws)
{
    const auto& param = GetParam();
    auto inputType = RankedTensorType::get(llvm::ArrayRef<int64_t>(param.input_shape), builder.getF32Type());

    auto module = ModuleOp::create(loc);
    auto funcType = builder.getFunctionType({inputType}, {inputType});
    auto func = func::FuncOp::create(loc, "test", funcType);

    func.addEntryBlock();
    builder.setInsertionPointToStart(&func.getBody().front());
    Value input = func.getArgument(0);

    auto shapeType = RankedTensorType::get({static_cast<int64_t>(param.shape_tensor.size())}, builder.getI64Type());
    llvm::SmallVector<Value> shapeValues;

    for (int64_t val : param.shape_tensor)
    {
        shapeValues.push_back(arith::ConstantOp::create(builder, loc, builder.getI64Type(), builder.getI64IntegerAttr(val)));
    }

    Value shape = tensor::FromElementsOp::create(builder, loc, shapeType, shapeValues);

    EXPECT_THROW(buildReshapeOp(builder, loc, input, shape, param.allowzero), std::runtime_error);

    func::ReturnOp::create(builder, loc, input);
    module.push_back(func);
}

INSTANTIATE_TEST_SUITE_P(
    ReshapeErrorTests,
    ReshapeErrorTest,
    testing::ValuesIn(reshapeErrorTests)
);
