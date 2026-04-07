#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include "frontend/onnx_loader.hpp"
#include "graph/graph.hpp"
#include "graph/node.hpp"
#include "graph/tensor.hpp"


#include "onnx.pb.h"

using namespace tc;


class OnnxLoaderTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        temp_path = std::filesystem::temp_directory_path() / "test_model.onnx";
    }

    void TearDown() override
    {
        if (std::filesystem::exists(temp_path))
        {
            std::filesystem::remove(temp_path);
        }
    }

    std::filesystem::path temp_path;
};


static void writeModelToFile(const onnx::ModelProto& model, const std::filesystem::path& path)
{
    std::ofstream ofs(path, std::ios::binary);

    if (!ofs) throw std::runtime_error("Cannot open file for writing");

    if (!model.SerializeToOstream(&ofs))
    {
        throw std::runtime_error("Failed to serialize model");
    }
}

// model with 1 Add node: input0 + input1 -> output
static onnx::ModelProto createSimpleAddModel()
{
    onnx::ModelProto model;
    model.set_ir_version(8);
    model.set_producer_name("test");
    model.set_producer_version("1.0");
    model.set_domain("");
    model.set_model_version(1);
    model.set_doc_string("Test model with Add node");

    auto* graph = model.mutable_graph();
    graph->set_name("test_graph");

    // inputs
    onnx::ValueInfoProto* input0 = graph->add_input();
    input0->set_name("input0");
    auto* type0 = input0->mutable_type()->mutable_tensor_type();
    type0->set_elem_type(onnx::TensorProto::FLOAT);
    auto* shape0 = type0->mutable_shape();
    shape0->add_dim()->set_dim_value(1);
    shape0->add_dim()->set_dim_value(1);

    onnx::ValueInfoProto* input1 = graph->add_input();
    input1->set_name("input1");
    auto* type1 = input1->mutable_type()->mutable_tensor_type();
    type1->set_elem_type(onnx::TensorProto::FLOAT);
    auto* shape1 = type1->mutable_shape();
    shape1->add_dim()->set_dim_value(1);
    shape1->add_dim()->set_dim_value(1);

    // output
    onnx::ValueInfoProto* output = graph->add_output();
    output->set_name("output");
    auto* out_type = output->mutable_type()->mutable_tensor_type();
    out_type->set_elem_type(onnx::TensorProto::FLOAT);
    auto* out_shape = out_type->mutable_shape();
    out_shape->add_dim()->set_dim_value(1);
    out_shape->add_dim()->set_dim_value(1);

    // Add
    onnx::NodeProto* node = graph->add_node();
    node->set_op_type("Add");
    node->set_name("add_node");
    node->add_input("input0");
    node->add_input("input1");
    node->add_output("output");

    return model;
}

TEST_F(OnnxLoaderTest, LoadSimpleAddModel)
{
    // create and store
    auto model = createSimpleAddModel();
    ASSERT_NO_THROW(writeModelToFile(model, temp_path));

    // load
    OnnxLoader loader;
    auto graph = loader.load(temp_path);

    // metadata
    EXPECT_EQ(graph->getName(), "test_graph");

    // tensors
    const auto& tensors = graph->getTensors();
    EXPECT_GE(tensors.size(), 3);

    // in/out
    const auto& inputs = graph->getInputs();
    EXPECT_EQ(inputs.size(), 2);
    EXPECT_EQ(inputs[0], "input0");
    EXPECT_EQ(inputs[1], "input1");

    const auto& outputs = graph->getOutputs();
    EXPECT_EQ(outputs.size(), 1);
    EXPECT_EQ(outputs[0], "output");

    // nodes
    const auto& nodes = graph->getNodes();
    ASSERT_EQ(nodes.size(), 1);

    const auto& node = nodes[0];
    EXPECT_EQ(node->getOpStr(), "Add");
    EXPECT_EQ(node->getOpType(), OpType::Add);
    EXPECT_EQ(node->getName(), "add_node");

    const auto& node_inputs = node->getInputs();
    EXPECT_EQ(node_inputs.size(), 2);
    EXPECT_EQ(node_inputs[0], "input0");
    EXPECT_EQ(node_inputs[1], "input1");

    const auto& node_outputs = node->getOutputs();
    EXPECT_EQ(node_outputs.size(), 1);
    EXPECT_EQ(node_outputs[0], "output");
}

TEST_F(OnnxLoaderTest, ReadModelInfo)
{
    auto model = createSimpleAddModel();
    ASSERT_NO_THROW(writeModelToFile(model, temp_path));

    auto info = OnnxLoader::readModelInfo(temp_path);
    EXPECT_EQ(info.ir_version, 8);
    EXPECT_EQ(info.producer_name, "test");
    EXPECT_EQ(info.producer_version, "1.0");
    EXPECT_EQ(info.domain, "");
    EXPECT_EQ(info.model_version, 1);
    EXPECT_EQ(info.doc_string, "Test model with Add node");
    EXPECT_EQ(info.graph_name, "test_graph");
}

TEST_F(OnnxLoaderTest, LoadWithAttributes)
{
    onnx::ModelProto model;
    model.set_ir_version(8);
    auto* graph = model.mutable_graph();
    graph->set_name("conv_test");

    // inputs X and W
    onnx::ValueInfoProto* inputX = graph->add_input();
    inputX->set_name("X");
    auto* typeX = inputX->mutable_type()->mutable_tensor_type();
    typeX->set_elem_type(onnx::TensorProto::FLOAT);
    auto* shapeX = typeX->mutable_shape();


    shapeX->add_dim()->set_dim_value(1); // batch
    shapeX->add_dim()->set_dim_value(3); // channels
    shapeX->add_dim()->set_dim_value(5); // height
    shapeX->add_dim()->set_dim_value(5); // width


    onnx::ValueInfoProto* inputW = graph->add_input();
    inputW->set_name("W");
    auto* typeW = inputW->mutable_type()->mutable_tensor_type();
    typeW->set_elem_type(onnx::TensorProto::FLOAT);
    auto* shapeW = typeW->mutable_shape();


    shapeW->add_dim()->set_dim_value(3); // out channels
    shapeW->add_dim()->set_dim_value(3); // in channels
    shapeW->add_dim()->set_dim_value(3); // kernel H
    shapeW->add_dim()->set_dim_value(3); // kernel W


    // oputput Y
    onnx::ValueInfoProto* output = graph->add_output();
    output->set_name("Y");
    auto* typeY = output->mutable_type()->mutable_tensor_type();
    typeY->set_elem_type(onnx::TensorProto::FLOAT);
    auto* shapeY = typeY->mutable_shape();
    shapeY->add_dim()->set_dim_value(1);
    shapeY->add_dim()->set_dim_value(3);
    shapeY->add_dim()->set_dim_value(3);
    shapeY->add_dim()->set_dim_value(3);

    // Conv
    onnx::NodeProto* node = graph->add_node();
    node->set_op_type("Conv");
    node->set_name("conv_node");
    node->add_input("X");
    node->add_input("W");
    node->add_output("Y");

    auto* attr_kernel = node->add_attribute();
    attr_kernel->set_name("kernel_shape");
    attr_kernel->set_type(onnx::AttributeProto::INTS);
    attr_kernel->add_ints(3);
    attr_kernel->add_ints(3);

    auto* attr_strides = node->add_attribute();
    attr_strides->set_name("strides");
    attr_strides->set_type(onnx::AttributeProto::INTS);
    attr_strides->add_ints(1);
    attr_strides->add_ints(1);

    auto* attr_pads = node->add_attribute();
    attr_pads->set_name("pads");
    attr_pads->set_type(onnx::AttributeProto::INTS);
    attr_pads->add_ints(1);
    attr_pads->add_ints(1);
    attr_pads->add_ints(1);
    attr_pads->add_ints(1);

    auto* attr_group = node->add_attribute();
    attr_group->set_name("group");
    attr_group->set_type(onnx::AttributeProto::INT);
    attr_group->set_i(1);

    ASSERT_NO_THROW(writeModelToFile(model, temp_path));

    OnnxLoader loader;
    auto graphPtr = loader.load(temp_path);

    // nodes
    const auto& nodes = graphPtr->getNodes();
    ASSERT_EQ(nodes.size(), 1);
    const auto& convNode = nodes[0];
    EXPECT_EQ(convNode->getOpType(), OpType::Conv);
    EXPECT_TRUE(convNode->hasAttribute("kernel_shape"));
    EXPECT_TRUE(convNode->hasAttribute("strides"));
    EXPECT_TRUE(convNode->hasAttribute("pads"));
    EXPECT_TRUE(convNode->hasAttribute("group"));

    const auto& kernel = convNode->getAttribute("kernel_shape");
    EXPECT_EQ(kernel.getType(), AttributeType::INTS);
    auto kernelVals = kernel.asInts();
    ASSERT_EQ(kernelVals.size(), 2);
    EXPECT_EQ(kernelVals[0], 3);
    EXPECT_EQ(kernelVals[1], 3);

    const auto& strides = convNode->getAttribute("strides");
    auto strideVals = strides.asInts();
    ASSERT_EQ(strideVals.size(), 2);
    EXPECT_EQ(strideVals[0], 1);
    EXPECT_EQ(strideVals[1], 1);

    const auto& group = convNode->getAttribute("group");
    EXPECT_EQ(group.asInt(), 1);
}
