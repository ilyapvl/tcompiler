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

class MatMulTestBase : public ::testing::Test
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

using MatMulParam = MatMulCase;

class MatMulTest : public MatMulTestBase, public ::testing::WithParamInterface<MatMulParam> {};

TEST_P(MatMulTest, MatMul)
{
    const auto& param = GetParam();
    auto aType = RankedTensorType::get(llvm::ArrayRef<int64_t>(param.a_shape), builder.getF32Type());
    auto bType = RankedTensorType::get(llvm::ArrayRef<int64_t>(param.b_shape), builder.getF32Type());
    auto outType = RankedTensorType::get(llvm::ArrayRef<int64_t>(param.expected_shape), builder.getF32Type());

    auto module = ModuleOp::create(loc);
    auto funcType = builder.getFunctionType({aType, bType}, {outType});
    auto func = func::FuncOp::create(loc, "test", funcType);

    func.addEntryBlock();
    builder.setInsertionPointToStart(&func.getBody().front());

    Value A = func.getArgument(0);
    Value B = func.getArgument(1);

    Value result = buildMatmulGeneric(builder, loc, A, B, param.transA, param.transB, &ctx);
    ASSERT_TRUE(result);

    

    auto generic = result.getDefiningOp<linalg::GenericOp>();
    ASSERT_TRUE(generic);
    EXPECT_EQ(generic.getNumDpsInputs(), 2);
    EXPECT_EQ(generic.getNumDpsInits(), 1);
    EXPECT_EQ(result.getType(), outType);



    func::ReturnOp::create(builder, loc, result);
    module.push_back(func);
    EXPECT_TRUE(succeeded(verify(module)));
}

INSTANTIATE_TEST_SUITE_P(
    MatMulTests,
    MatMulTest,
    testing::ValuesIn(MatMulTests)
);
