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

    Value result = buildMatMul(builder, loc, A, B, param.transA, param.transB, &ctx);
    ASSERT_TRUE(result);



    bool noTranspose = !param.transA && !param.transB;
    bool rank2 = (aType.getRank() == 2 && bType.getRank() == 2);
    bool rank3 = (aType.getRank() == 3 && bType.getRank() == 3);
    bool sameBatch = true;

    if (rank3)
    {
        auto batchA = aType.getShape().drop_back(2);
        auto batchB = bType.getShape().drop_back(2);
        sameBatch = (batchA == batchB);
    }

    if (noTranspose && rank2)
    {
        auto namedOp = result.getDefiningOp<linalg::MatmulOp>();
        ASSERT_TRUE(namedOp) << "Expected linalg.matmul for rank 2";
        EXPECT_EQ(namedOp.getNumDpsInputs(), 2);
        EXPECT_EQ(namedOp.getNumDpsInits(), 1);
    }
    
    else if (noTranspose && rank3 && sameBatch)
    {
        auto namedOp = result.getDefiningOp<linalg::BatchMatmulOp>();
        ASSERT_TRUE(namedOp) << "Expected linalg.batch_matmul for rank 3";
        EXPECT_EQ(namedOp.getNumDpsInputs(), 2);
        EXPECT_EQ(namedOp.getNumDpsInits(), 1);
    }
    
    else
    {
        auto generic = result.getDefiningOp<linalg::GenericOp>();
        ASSERT_TRUE(generic) << "Expected linalg.generic for other cases";
        EXPECT_EQ(generic.getNumDpsInputs(), 2);
        EXPECT_EQ(generic.getNumDpsInits(), 1);
    }



    func::ReturnOp::create(builder, loc, result);
    module.push_back(func);
    EXPECT_TRUE(succeeded(verify(module)));
}

INSTANTIATE_TEST_SUITE_P(
    MatMulTests,
    MatMulTest,
    testing::ValuesIn(MatMulTests)
);
