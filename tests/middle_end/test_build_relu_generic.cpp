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





    Value result = buildReLUGeneric(builder, loc, input, &ctx);
    ASSERT_TRUE(result);


    auto generic = result.getDefiningOp<linalg::GenericOp>();
    ASSERT_TRUE(generic);
    EXPECT_EQ(generic.getNumDpsInputs(), 1);
    EXPECT_EQ(generic.getNumDpsInits(), 1);




    auto& region = generic.getRegion();
    ASSERT_FALSE(region.empty());



    auto& block = region.front();
    auto yield = dyn_cast<linalg::YieldOp>(block.getTerminator());
    ASSERT_TRUE(yield);



    auto maxOp = yield.getOperand(0).getDefiningOp<arith::MaximumFOp>();
    ASSERT_TRUE(maxOp);
    EXPECT_EQ(maxOp.getLhs(), block.getArgument(0));




    auto zeroConst = maxOp.getRhs().getDefiningOp<arith::ConstantOp>();
    ASSERT_TRUE(zeroConst);



    auto zeroAttr = mlir::cast<FloatAttr>(zeroConst.getValue());
    EXPECT_FLOAT_EQ(zeroAttr.getValueAsDouble(), 0.0);



    func::ReturnOp::create(builder, loc, result);
    module.push_back(func);
    EXPECT_TRUE(succeeded(verify(module)));
}

INSTANTIATE_TEST_SUITE_P(
    ReLUTests,
    ReLUParamTest,
    testing::ValuesIn(unaryShapes)
);
