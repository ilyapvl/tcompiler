#include <gtest/gtest.h>
#include "middle_end/mlir_builders.hpp"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/AffineExpr.h"

using namespace tc;
using namespace mlir;

class MakeBroadcastMapTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ctx = std::make_unique<MLIRContext>();
    }

    std::unique_ptr<MLIRContext> ctx;
};

TEST_F(MakeBroadcastMapTest, ZeroRankOperand)
{
    SmallVector<int64_t> operandShape = {};
    auto map = makeBroadcastMap(2, 0, operandShape, ctx.get());

    EXPECT_EQ(map.getNumDims(), 2);
    EXPECT_EQ(map.getNumResults(), 0);
}

TEST_F(MakeBroadcastMapTest, SameRankNoBroadcast)
{
    SmallVector<int64_t> operandShape = {2, 3};

    auto map = makeBroadcastMap(2, 2, operandShape, ctx.get());
    EXPECT_EQ(map.getNumResults(), 2);

    auto d0 = getAffineDimExpr(0, ctx.get());
    auto d1 = getAffineDimExpr(1, ctx.get());

    EXPECT_EQ(map.getResult(0), d0);
    EXPECT_EQ(map.getResult(1), d1);
}

TEST_F(MakeBroadcastMapTest, SameRankWithOne)
{
    SmallVector<int64_t> operandShape = {1, 3};

    auto map = makeBroadcastMap(2, 2, operandShape, ctx.get());
    
    EXPECT_EQ(map.getNumResults(), 2);

    auto d1 = getAffineDimExpr(1, ctx.get());
    auto c0 = getAffineConstantExpr(0, ctx.get());
    EXPECT_EQ(map.getResult(0), c0);
    EXPECT_EQ(map.getResult(1), d1);
}

TEST_F(MakeBroadcastMapTest, DifferentRank)
{
    SmallVector<int64_t> operandShape = {3};

    auto map = makeBroadcastMap(2, 1, operandShape, ctx.get());

    EXPECT_EQ(map.getNumResults(), 1);

    auto d1 = getAffineDimExpr(1, ctx.get());
    EXPECT_EQ(map.getResult(0), d1);
}

TEST_F(MakeBroadcastMapTest, DifferentRankWithOne)
{
    SmallVector<int64_t> operandShape = {1, 4};

    auto map = makeBroadcastMap(3, 2, operandShape, ctx.get());
    EXPECT_EQ(map.getNumResults(), 2);

    auto d1 = getAffineDimExpr(1, ctx.get());
    auto d2 = getAffineDimExpr(2, ctx.get());
    EXPECT_EQ(map.getResult(0), d1);
    EXPECT_EQ(map.getResult(1), d2);
}

TEST_F(MakeBroadcastMapTest, OperandRankHigherThanOutRank)
{
    SmallVector<int64_t> operandShape = {2, 3, 4};

    auto map = makeBroadcastMap(2, 3, operandShape, ctx.get());
    EXPECT_EQ(map.getNumResults(), 3);


    auto d0 = getAffineDimExpr(0, ctx.get());
    auto d1 = getAffineDimExpr(1, ctx.get());
    auto c0 = getAffineConstantExpr(0, ctx.get());
    EXPECT_EQ(map.getResult(0), c0);
    EXPECT_EQ(map.getResult(1), d0);
    EXPECT_EQ(map.getResult(2), d1);
}
