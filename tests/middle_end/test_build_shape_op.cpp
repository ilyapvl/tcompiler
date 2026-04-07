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

class ShapeTestBase : public ::testing::Test
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

using ShapeParam = ShapeCase;

class ShapeOpTest : public ShapeTestBase, public ::testing::WithParamInterface<ShapeParam> {};

TEST_P(ShapeOpTest, Shape)
{
    const auto& param = GetParam();
    auto inputType = RankedTensorType::get(llvm::ArrayRef<int64_t>(param.input_shape), builder.getF32Type());
    auto outType = RankedTensorType::get(llvm::ArrayRef<int64_t>(param.expected_output_shape), builder.getI64Type());

    auto module = ModuleOp::create(loc);
    auto funcType = builder.getFunctionType({inputType}, {outType});
    auto func = func::FuncOp::create(loc, "test", funcType);

    func.addEntryBlock();
    builder.setInsertionPointToStart(&func.getBody().front());

    Value input = func.getArgument(0);
    Value result = buildShapeOp(builder, loc, input, param.start, param.end);
    ASSERT_TRUE(result);
    EXPECT_EQ(result.getType(), outType);




    if (param.expected_output_shape.empty() || param.expected_output_shape[0] == 0)
    {
        auto emptyOp = result.getDefiningOp<tensor::EmptyOp>();
        ASSERT_TRUE(emptyOp);
        EXPECT_EQ(emptyOp.getType().getDimSize(0), 0);


    }
    
    else
    {
        auto fromElements = result.getDefiningOp<tensor::FromElementsOp>();
        ASSERT_TRUE(fromElements);

        auto elements = fromElements.getElements();
        EXPECT_EQ(elements.size(), param.expected_output_shape[0]);


        int64_t expectedNumElements = param.expected_output_shape[0];
        for (int64_t i = 0; i < expectedNumElements; ++i)
        {
            auto castOp = elements[i].getDefiningOp<arith::IndexCastOp>();
            ASSERT_TRUE(castOp);

            auto dimOp = castOp.getIn().getDefiningOp<tensor::DimOp>();
            ASSERT_TRUE(dimOp);
            EXPECT_EQ(dimOp.getSource(), input);

            auto indexValue = dimOp.getIndex();
            auto constIndex = indexValue.getDefiningOp<arith::ConstantIndexOp>();
            ASSERT_TRUE(constIndex);
        }
    }

    func::ReturnOp::create(builder, loc, result);
    module.push_back(func);
    EXPECT_TRUE(succeeded(verify(module)));
}

INSTANTIATE_TEST_SUITE_P(
    ShapeTests,
    ShapeOpTest,
    testing::ValuesIn(shapeTests)
);
