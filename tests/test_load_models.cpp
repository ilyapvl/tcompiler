#include <gtest/gtest.h>
#include <graph/graph.h>
#include <graph/exceptions.h>
#include <filesystem>
#include <fstream>

using namespace graph;

static bool file_exists(const std::string& path)
{
    std::ifstream f(path);
    return f.good();
}

const std::string MODELS_DIR = "../models/";
TEST(LoadModelTest, SixOpsFixed)
{
    std::string path = MODELS_DIR + "test.onnx";
    if (!file_exists(path))
    {
        GTEST_SKIP() << "Model file not found: " << path;
    }

    Graph actual;
    actual.load_from_onnx(path);

    Graph expected;

    auto add_tensor = [&](const std::string& name, DataType type,
                          const std::vector<int64_t>& dims, bool constant)
    {
        TensorInfo* ti = expected.add_tensor(name);
        ti->data_type = type;
        ti->dims = dims;
        ti->is_constant = constant;
    };

    add_tensor("X1", DataType::FLOAT, {1, 3, 32, 32}, false);
    add_tensor("X2", DataType::FLOAT, {1, 256}, false);

    add_tensor("C1", DataType::FLOAT, {1, 16, 32, 32}, true);
    add_tensor("C2", DataType::FLOAT, {1, 16, 32, 32}, true);
    add_tensor("W_conv", DataType::FLOAT, {16, 3, 3, 3}, true);
    add_tensor("B_conv", DataType::FLOAT, {16}, true);
    add_tensor("W_matmul", DataType::FLOAT, {256, 128}, true);
    add_tensor("W_gemm", DataType::FLOAT, {128, 64}, true);
    add_tensor("B_gemm", DataType::FLOAT, {64}, true);

    add_tensor("Y1", DataType::FLOAT, {1, 16, 32, 32}, false);
    add_tensor("Z1", DataType::FLOAT, {1, 16, 32, 32}, false);
    add_tensor("A1", DataType::FLOAT, {1, 16, 32, 32}, false);
    add_tensor("Y2", DataType::FLOAT, {1, 128}, false);
    add_tensor("Out1", DataType::FLOAT, {1, 16, 32, 32}, false);
    add_tensor("Out2", DataType::FLOAT, {1, 64}, false);

    Node* conv = expected.add_node("conv", "Conv");
    conv->add_input("X1");
    conv->add_input("W_conv");
    conv->add_input("B_conv");
    conv->add_output("Y1");
    conv->set_attribute("kernel_shape", std::vector<int64_t>{3, 3});
    conv->set_attribute("pads", std::vector<int64_t>{1, 1, 1, 1});
    conv->set_attribute("strides", std::vector<int64_t>{1, 1});

    Node* relu = expected.add_node("relu", "Relu");
    relu->add_input("Y1");
    relu->add_output("Z1");

    Node* add = expected.add_node("add", "Add");
    add->add_input("Z1");
    add->add_input("C1");
    add->add_output("A1");

    Node* mul = expected.add_node("mul", "Mul");
    mul->add_input("A1");
    mul->add_input("C2");
    mul->add_output("Out1");

    Node* matmul = expected.add_node("matmul", "MatMul");
    matmul->add_input("X2");
    matmul->add_input("W_matmul");
    matmul->add_output("Y2");

    Node* gemm = expected.add_node("gemm", "Gemm");
    gemm->add_input("Y2");
    gemm->add_input("W_gemm");
    gemm->add_input("B_gemm");
    gemm->add_output("Out2");
    gemm->set_attribute("alpha", 1.0f);
    gemm->set_attribute("beta", 1.0f);
    gemm->set_attribute("transB", int64_t(0));

    expected.add_input("X1");
    expected.add_input("X2");
    expected.add_output("Out1");
    expected.add_output("Out2");

    expected.build_connections();

    EXPECT_EQ(actual, expected);

    //raw data
    const TensorInfo* c1 = actual.get_tensor("C1");
    ASSERT_NE(c1, nullptr);
    EXPECT_EQ(c1->raw_data.size(), 1*16*32*32 * sizeof(float));
    const float* c1_data = reinterpret_cast<const float*>(c1->raw_data.data());
    for (size_t i = 0; i < 1*16*32*32; ++i)
    {
        EXPECT_FLOAT_EQ(c1_data[i], 2.0f);
    }

    const TensorInfo* c2 = actual.get_tensor("C2");
    ASSERT_NE(c2, nullptr);
    EXPECT_EQ(c2->raw_data.size(), 1*16*32*32 * sizeof(float));
    const float* c2_data = reinterpret_cast<const float*>(c2->raw_data.data());
    for (size_t i = 0; i < 1*16*32*32; ++i)
    {
        EXPECT_FLOAT_EQ(c2_data[i], 0.5f);
    }

    const TensorInfo* w_conv = actual.get_tensor("W_conv");
    ASSERT_NE(w_conv, nullptr);
    EXPECT_EQ(w_conv->raw_data.size(), 16*3*3*3 * sizeof(float));
    const float* wc_data = reinterpret_cast<const float*>(w_conv->raw_data.data());
    for (size_t i = 0; i < 16*3*3*3; ++i)
    {
        EXPECT_FLOAT_EQ(wc_data[i], 1.0f);
    }

    const TensorInfo* b_conv = actual.get_tensor("B_conv");
    ASSERT_NE(b_conv, nullptr);
    EXPECT_EQ(b_conv->raw_data.size(), 16 * sizeof(float));
    const float* bc_data = reinterpret_cast<const float*>(b_conv->raw_data.data());
    for (int i = 0; i < 16; ++i)
    {
        EXPECT_FLOAT_EQ(bc_data[i], 0.0f);
    }

    const TensorInfo* w_matmul = actual.get_tensor("W_matmul");
    ASSERT_NE(w_matmul, nullptr);
    EXPECT_EQ(w_matmul->raw_data.size(), 256*128 * sizeof(float));
    const float* wm_data = reinterpret_cast<const float*>(w_matmul->raw_data.data());
    for (size_t i = 0; i < 256*128; ++i)
    {
        EXPECT_FLOAT_EQ(wm_data[i], 0.1f);
    }

    const TensorInfo* w_gemm = actual.get_tensor("W_gemm");
    ASSERT_NE(w_gemm, nullptr);
    EXPECT_EQ(w_gemm->raw_data.size(), 128*64 * sizeof(float));
    const float* wg_data = reinterpret_cast<const float*>(w_gemm->raw_data.data());
    for (size_t i = 0; i < 128*64; ++i)
    {
        EXPECT_FLOAT_EQ(wg_data[i], 0.2f);
    }

    const TensorInfo* b_gemm = actual.get_tensor("B_gemm");
    ASSERT_NE(b_gemm, nullptr);
    EXPECT_EQ(b_gemm->raw_data.size(), 64 * sizeof(float));
    const float* bg_data = reinterpret_cast<const float*>(b_gemm->raw_data.data());
    for (int i = 0; i < 64; ++i)
    {
        EXPECT_FLOAT_EQ(bg_data[i], 0.05f);
    }
}
