#include <gtest/gtest.h>
#include "middle_end/mlir_builders.hpp"
#include "mlir/IR/BuiltinTypes.h"

using namespace tc;
using namespace mlir;

TEST(InferBroadcastShapeTest, SameRankSameDim)
{
    SmallVector<int64_t> shape1 = {2, 3};
    SmallVector<int64_t> shape2 = {2, 3};

    auto result = inferBroadcastShape(shape1, shape2);

    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result[0], 2);
    EXPECT_EQ(result[1], 3);
}

TEST(InferBroadcastShapeTest, SameRankBroadcastOne)
{
    SmallVector<int64_t> shape1 = {1, 3};
    SmallVector<int64_t> shape2 = {2, 3};

    auto result = inferBroadcastShape(shape1, shape2);

    EXPECT_EQ(result[0], 2);
    EXPECT_EQ(result[1], 3);
}

TEST(InferBroadcastShapeTest, SameRankBroadcastBoth)
{
    SmallVector<int64_t> shape1 = {1, 3, 1};
    SmallVector<int64_t> shape2 = {2, 1, 4};
    auto result = inferBroadcastShape(shape1, shape2);

    EXPECT_EQ(result.size(), 3);
    EXPECT_EQ(result[0], 2);
    EXPECT_EQ(result[1], 3);
    EXPECT_EQ(result[2], 4);
}

TEST(InferBroadcastShapeTest, DifferentRanks)
{
    SmallVector<int64_t> shape1 = {3};
    SmallVector<int64_t> shape2 = {2, 3};

    auto result = inferBroadcastShape(shape1, shape2);

    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result[0], 2);
    EXPECT_EQ(result[1], 3);
}

TEST(InferBroadcastShapeTest, DifferentRanksWithOnesIncompatible)
{
    SmallVector<int64_t> shape1 = {1, 4, 1};
    SmallVector<int64_t> shape2 = {5, 1};

    EXPECT_THROW(inferBroadcastShape(shape1, shape2), std::runtime_error);
}

TEST(InferBroadcastShapeTest, IncompatibleShapesThrows)
{
    SmallVector<int64_t> shape1 = {2, 3};
    SmallVector<int64_t> shape2 = {4, 5};

    EXPECT_THROW(inferBroadcastShape(shape1, shape2), std::runtime_error);
}


TEST(InferBroadcastShapeTest, DynamicDimensions)
{
    SmallVector<int64_t> shape1 = {ShapedType::kDynamic, 3};
    SmallVector<int64_t> shape2 = {2, ShapedType::kDynamic};

    auto result = inferBroadcastShape(shape1, shape2);

    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result[0], ShapedType::kDynamic);
    EXPECT_EQ(result[1], ShapedType::kDynamic);
}


TEST(InferBroadcastShapeTest, DynamicWithStatic)
{
    SmallVector<int64_t> shape1 = {ShapedType::kDynamic, 4};
    SmallVector<int64_t> shape2 = {2, 1};

    auto result = inferBroadcastShape(shape1, shape2);

    EXPECT_EQ(result[0], ShapedType::kDynamic);
    EXPECT_EQ(result[1], 4);
}
