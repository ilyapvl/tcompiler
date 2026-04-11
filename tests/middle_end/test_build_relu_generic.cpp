#include <gtest/gtest.h>
#include "middle_end/mlir_builders.hpp"
#include "test_dimensions.hpp"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace tc;
using namespace mlir;
using namespace tc::test;

class ReLUTest : public ::testing::Test
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

using ShapeParam = std::vector<int64_t>;

class ReLUParamTest : public ReLUTest, public ::testing::WithParamInterface<ShapeParam> {};

TEST_P(ReLUParamTest, ReLU)
{
    auto shapeVec = GetParam();
    auto shape = llvm::ArrayRef<int64_t>(shapeVec);
    auto inputType = RankedTensorType::get(shape, builder.getF32Type());
    auto outType = inputType;

    auto module = ModuleOp::create(loc);

    auto funcType = builder.getFunctionType({inputType}, {outType});
    auto func = func::FuncOp::create(loc, "test", funcType);
    func.addEntryBlock();
    builder.setInsertionPointToStart(&func.getBody().front());
    Value input = func.getArgument(0);





    Value result = buildReLU(builder, loc, input, &ctx);
    ASSERT_TRUE(result);


    auto maxOp = result.getDefiningOp<linalg::MaxOp>();
    ASSERT_TRUE(maxOp) << "Expected linalg.max operation";
    EXPECT_EQ(maxOp.getNumDpsInputs(), 2);
    EXPECT_EQ(maxOp.getNumDpsInits(), 1);



    func::ReturnOp::create(builder, loc, result);
    module.push_back(func);
    EXPECT_TRUE(succeeded(verify(module)));
}

INSTANTIATE_TEST_SUITE_P(
    ReLUTests,
    ReLUParamTest,
    testing::ValuesIn(unaryShapes)
);
