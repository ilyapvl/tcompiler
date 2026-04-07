#include <gtest/gtest.h>
#include "graph/node.hpp"

using namespace tc;


TEST(NodeTest, ConstructionAndGetters)
{
    Node node("add1", OpType::Add, "Add",
            {"input_a", "input_b"},
            {"output"},
            {});




    EXPECT_EQ(node.getName(), "add1");
    EXPECT_EQ(node.getOpType(), OpType::Add);
    EXPECT_EQ(node.getOpStr(), "Add");

    const auto& inputs = node.getInputs();
    ASSERT_EQ(inputs.size(), 2);
    EXPECT_EQ(inputs[0], "input_a");
    EXPECT_EQ(inputs[1], "input_b");

    const auto& outputs = node.getOutputs();
    ASSERT_EQ(outputs.size(), 1);
    EXPECT_EQ(outputs[0], "output");
    EXPECT_TRUE(node.getAttributes().empty());

    EXPECT_FALSE(node.hasAttribute("dummy"));
    EXPECT_THROW((void)node.getAttribute("dummy"), std::out_of_range);
}

TEST(NodeTest, Attributes) //TODO - Tensor, Graph attributes
{
    Node node("conv1", OpType::Conv, "Conv",
            {"x", "w"}, {"y"}, {});

    Attribute a1("kernel_shape", AttributeType::INTS, std::vector<int64_t>{3, 3});
    node.addAttribute(std::move(a1));

    Attribute a2("strides", AttributeType::INTS, std::vector<int64_t>{1, 1});
    node.addAttribute(std::move(a2));

    EXPECT_TRUE(node.hasAttribute("kernel_shape"));
    EXPECT_TRUE(node.hasAttribute("strides"));
    EXPECT_FALSE(node.hasAttribute("dilations"));

    const Attribute& ks = node.getAttribute("kernel_shape");
    EXPECT_EQ(ks.getType(), AttributeType::INTS);
    auto vals = ks.asInts();
    ASSERT_EQ(vals.size(), 2);
    EXPECT_EQ(vals[0], 3);
    EXPECT_EQ(vals[1], 3);

    const Attribute& ss = node.getAttribute("strides");
    EXPECT_EQ(ss.getType(), AttributeType::INTS);
    auto svals = ss.asInts();
    ASSERT_EQ(svals.size(), 2);
    EXPECT_EQ(svals[0], 1);
    EXPECT_EQ(svals[1], 1);

    std::string str = node.toString();
    EXPECT_NE(str.find("Conv"), std::string::npos);
    EXPECT_NE(str.find("conv1"), std::string::npos);
    EXPECT_NE(str.find("kernel_shape"), std::string::npos);
}

TEST(NodeTest, OpTypeConversion)
{
    EXPECT_EQ(opTypeFromString("Add"), OpType::Add);
    EXPECT_EQ(opTypeFromString("Mul"), OpType::Mul);
    EXPECT_EQ(opTypeFromString("MatMul"), OpType::MatMul);
    EXPECT_EQ(opTypeFromString("Gemm"), OpType::Gemm);
    EXPECT_EQ(opTypeFromString("Conv"), OpType::Conv);
    EXPECT_EQ(opTypeFromString("Relu"), OpType::Relu);
    EXPECT_EQ(opTypeFromString("Unknown"), OpType::Other);
    EXPECT_EQ(opTypeFromString(""), OpType::Other);

    EXPECT_EQ(opTypeToString(OpType::Add), "Add");
    EXPECT_EQ(opTypeToString(OpType::Relu), "Relu");
    EXPECT_EQ(opTypeToString(OpType::Other), "Other");
}
