#include <gtest/gtest.h>
#include "graph/attribute.hpp"
#include "graph/tensor.hpp"
#include "graph/graph.hpp"

using namespace tc;

TEST(AttributeTest, FloatAttribute)
{
    Attribute attr("alpha", AttributeType::FLOAT, 3.14f);

    EXPECT_EQ(attr.getName(), "alpha");
    EXPECT_EQ(attr.getType(), AttributeType::FLOAT);
    EXPECT_FLOAT_EQ(attr.asFloat(), 3.14f);
    EXPECT_THROW((void)attr.asInt(), std::bad_variant_access);
    EXPECT_EQ(attr.toString(), "alpha=3.14");
}

TEST(AttributeTest, IntAttribute)
{
    Attribute attr("size", AttributeType::INT, 4);

    EXPECT_EQ(attr.getName(), "size");
    EXPECT_EQ(attr.getType(), AttributeType::INT);
    EXPECT_EQ(attr.asInt(), 4);
    EXPECT_THROW((void)attr.asFloat(), std::bad_variant_access);
    EXPECT_EQ(attr.toString(), "size=4");
}

TEST(AttributeTest, StringAttribute)
{
    Attribute attr("name", AttributeType::STRING, std::string("my_op"));

    EXPECT_EQ(attr.getName(), "name");
    EXPECT_EQ(attr.getType(), AttributeType::STRING);
    EXPECT_EQ(attr.asString(), "my_op");
    EXPECT_THROW((void)attr.asInt(), std::bad_variant_access);
    EXPECT_EQ(attr.toString(), "name=\"my_op\"");
}

TEST(AttributeTest, FloatsAttribute)
{
    std::vector<float> vals = {1.0f, 2.5f, 3.7f};

    Attribute attr("kernel", AttributeType::FLOATS, vals);

    EXPECT_EQ(attr.getName(), "kernel");
    EXPECT_EQ(attr.getType(), AttributeType::FLOATS);
    EXPECT_EQ(vals, attr.asFloats());
    EXPECT_THROW((void)attr.asInts(), std::bad_variant_access);
    EXPECT_EQ(attr.toString(), "kernel=[1,2.5,3.7]");
}

TEST(AttributeTest, IntsAttribute)
{
    std::vector<int64_t> vals = {1, 2, 3};

    Attribute attr("dims", AttributeType::INTS, vals);

    EXPECT_EQ(attr.getName(), "dims");
    EXPECT_EQ(attr.getType(), AttributeType::INTS);
    EXPECT_EQ(vals, attr.asInts());
    EXPECT_THROW((void)attr.asFloats(), std::bad_variant_access);
    EXPECT_EQ(attr.toString(), "dims=[1,2,3]");
}

TEST(AttributeTest, StringsAttribute)
{
    std::vector<std::string> vals = {"x", "y", "z"};

    Attribute attr("names", AttributeType::STRINGS, vals);

    EXPECT_EQ(attr.getName(), "names");
    EXPECT_EQ(attr.getType(), AttributeType::STRINGS);
    EXPECT_EQ(vals, attr.asStrings());
    EXPECT_THROW((void)attr.asInts(), std::bad_variant_access);
    EXPECT_EQ(attr.toString(), "names=[\"x\",\"y\",\"z\"]");
}

TEST(AttributeTest, TensorAttribute)
{
    auto tensor = std::make_shared<Tensor>("w", DataType::FLOAT, TensorShape{{2, 3}});

    Attribute attr("weight", AttributeType::TENSOR, tensor);

    EXPECT_EQ(attr.getName(), "weight");
    EXPECT_EQ(attr.getType(), AttributeType::TENSOR);
    EXPECT_EQ(attr.asTensor(), tensor);
    EXPECT_THROW((void)attr.asInt(), std::bad_variant_access);

    EXPECT_EQ(attr.toString(), "weight=Tensor(w : float32[2x3])");
}

TEST(AttributeTest, GraphAttribute)
{
    auto graph = std::make_shared<Graph>("sub");

    Attribute attr("subgraph", AttributeType::GRAPH, graph);

    EXPECT_EQ(attr.getName(), "subgraph");
    EXPECT_EQ(attr.getType(), AttributeType::GRAPH);
    EXPECT_EQ(attr.asGraph(), graph);
    EXPECT_THROW((void)attr.asInt(), std::bad_variant_access);
}

TEST(AttributeTest, TensorsAttribute)
{
    auto t1 = std::make_shared<Tensor>("w1", DataType::FLOAT, TensorShape{{2, 2}});
    auto t2 = std::make_shared<Tensor>("w2", DataType::INT32, TensorShape{{1}});

    std::vector<std::shared_ptr<Tensor>> tensors = {t1, t2};
    Attribute attr("weights", AttributeType::TENSORS, tensors);

    EXPECT_EQ(attr.getName(), "weights");
    EXPECT_EQ(attr.getType(), AttributeType::TENSORS);
    EXPECT_EQ(attr.asTensors().size(), 2);
    EXPECT_EQ(attr.asTensors()[0], t1);
    EXPECT_EQ(attr.asTensors()[1], t2);
    EXPECT_THROW((void)attr.asTensor(), std::bad_variant_access);
    EXPECT_EQ(attr.toString(), "weights=[2 tensors]");
}
