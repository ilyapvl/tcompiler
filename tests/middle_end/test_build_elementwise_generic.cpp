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

class ElementwiseTest : public ::testing::Test
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






using CompatibleParam = std::tuple<OpType, CompatiblePair>;

class ElementwiseCompatibleTest : public ElementwiseTest, public ::testing::WithParamInterface<CompatibleParam> {};

TEST_P(ElementwiseCompatibleTest, AddMul)
{
    auto [op, pair] = GetParam();
    auto lhsType = RankedTensorType::get(llvm::ArrayRef<int64_t>(pair.lhs), builder.getF32Type());
    auto rhsType = RankedTensorType::get(llvm::ArrayRef<int64_t>(pair.rhs), builder.getF32Type());
    auto outType = RankedTensorType::get(llvm::ArrayRef<int64_t>(pair.expected), builder.getF32Type());

    auto module = ModuleOp::create(loc);

    auto funcType = builder.getFunctionType({lhsType, rhsType}, {outType});
    auto func = func::FuncOp::create(loc, "test", funcType);
    func.addEntryBlock();
    builder.setInsertionPointToStart(&func.getBody().front());

    Value lhs = func.getArgument(0);
    Value rhs = func.getArgument(1);

    Value result = buildElementwise(op, builder, loc, lhs, rhs, &ctx);
    ASSERT_TRUE(result);

    bool sameShape = (lhsType.getShape() == rhsType.getShape());

    if (sameShape)
    {
        if (op == OpType::Add)
        {
            auto namedOp = result.getDefiningOp<linalg::AddOp>();
            ASSERT_TRUE(namedOp) << "Expected linalg.add for same shapes";
            EXPECT_EQ(namedOp.getNumDpsInputs(), 2);
            EXPECT_EQ(namedOp.getNumDpsInits(), 1);
        }

        else if (op == OpType::Mul)
        {
            auto namedOp = result.getDefiningOp<linalg::MulOp>();
            ASSERT_TRUE(namedOp) << "Expected linalg.mul for same shapes";
            EXPECT_EQ(namedOp.getNumDpsInputs(), 2);
            EXPECT_EQ(namedOp.getNumDpsInits(), 1);
        }
    }

    else
    {
        auto generic = result.getDefiningOp<linalg::GenericOp>();
        ASSERT_TRUE(generic) << "Expected linalg.generic for different shapes";
        EXPECT_EQ(generic.getNumDpsInputs(), 2);
        EXPECT_EQ(generic.getNumDpsInits(), 1);

        auto& region = generic.getRegion();
        ASSERT_FALSE(region.empty());
        auto& block = region.front();
        auto yield = dyn_cast<linalg::YieldOp>(block.getTerminator());
        ASSERT_TRUE(yield);

        if (op == OpType::Add)
        {
            auto addOp = yield.getOperand(0).getDefiningOp<arith::AddFOp>();
            ASSERT_TRUE(addOp);
        }
        else if (op == OpType::Mul)
        {
            auto mulOp = yield.getOperand(0).getDefiningOp<arith::MulFOp>();
            ASSERT_TRUE(mulOp);
        }
    }

    func::ReturnOp::create(builder, loc, result);
    module.push_back(func);
    EXPECT_TRUE(succeeded(verify(module)));
}

INSTANTIATE_TEST_SUITE_P(
    AddMulCompatible,
    ElementwiseCompatibleTest,
    testing::Combine(
        testing::Values(OpType::Add, OpType::Mul),
        testing::ValuesIn(AddMulTests)
    )
);










class ElementwiseIncompatibleTest : public ElementwiseTest, public ::testing::WithParamInterface<IncompatiblePair> {};

TEST_P(ElementwiseIncompatibleTest, Throws)
{
    auto pair = GetParam();
    auto lhsType = RankedTensorType::get(llvm::ArrayRef<int64_t>(pair.lhs), builder.getF32Type());
    auto rhsType = RankedTensorType::get(llvm::ArrayRef<int64_t>(pair.rhs), builder.getF32Type());

    Value lhs = createConstantTensor(builder, loc, lhsType, {}, 1.0);
    Value rhs = createConstantTensor(builder, loc, rhsType, {}, 2.0);

    EXPECT_THROW(buildElementwise(OpType::Add, builder, loc, lhs, rhs, &ctx), std::runtime_error);
}

INSTANTIATE_TEST_SUITE_P(
    AddMulIncompatible,
    ElementwiseIncompatibleTest,
    testing::ValuesIn(AddMulErrorTests)
);
