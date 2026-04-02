#include <gtest/gtest.h>
#include "graph/tensor.hpp"
#include <vector>
#include <cstring>

using namespace tc;



TEST(TensorTest, ConstructionAndGetters)
{
    Tensor t("x", DataType::FLOAT, TensorShape{{2, 3, 4}});

    EXPECT_EQ(t.getName(), "x");
    EXPECT_EQ(t.getDtype(), DataType::FLOAT);
    EXPECT_EQ(t.getShape().dims.size(), 3);
    EXPECT_EQ(t.getShape().dims[0], 2);
    EXPECT_EQ(t.getShape().dims[1], 3);
    EXPECT_EQ(t.getShape().dims[2], 4);
    EXPECT_FALSE(t.hasData());




    EXPECT_EQ(t.toString(), "x : float32[2x3x4]");
}

TEST(TensorTest, SetRawData)
{
    Tensor t("weights", DataType::FLOAT, TensorShape{{2, 2}});
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<uint8_t> raw;

    raw.resize(data.size() * sizeof(float));
    std::memcpy(raw.data(), data.data(), raw.size());
    t.setRawData(std::move(raw));

    EXPECT_TRUE(t.hasData());
    EXPECT_EQ(t.getRawData().size(), 4 * sizeof(float));
}

TEST(TensorTest, NumElements)
{
    Tensor t1("scalar", DataType::FLOAT, TensorShape{});
    EXPECT_EQ(t1.numElements(), 1);
    Tensor t2("vector", DataType::FLOAT, TensorShape{{5}});
    EXPECT_EQ(t2.numElements(), 5);
    Tensor t3("matrix", DataType::FLOAT, TensorShape{{3, 4}});
    EXPECT_EQ(t3.numElements(), 12);
    Tensor t4("dynamic", DataType::FLOAT, TensorShape{{-1, 2}});

    EXPECT_EQ(t4.numElements(), 2);
}

TEST(TensorTest, HasValidData)
{
    Tensor t("w", DataType::FLOAT, TensorShape{{2, 2}});
    EXPECT_FALSE(t.hasValidData());

    // (4 float)
    std::vector<uint8_t> raw(4 * sizeof(float));
    t.setRawData(std::move(raw));
    EXPECT_TRUE(t.hasValidData());

    // wrong (3 float)
    std::vector<uint8_t> raw2(3 * sizeof(float));
    t.setRawData(std::move(raw2));
    EXPECT_FALSE(t.hasValidData());
}

TEST(TensorTest, GetDataAsFloat)
{
    Tensor t("w", DataType::FLOAT, TensorShape{{2, 2}});
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<uint8_t> raw(data.size() * sizeof(float));

    std::memcpy(raw.data(), data.data(), raw.size());
    t.setRawData(std::move(raw));

    auto span = t.getDataAs<float>();
    ASSERT_EQ(span.size(), 4);
    EXPECT_FLOAT_EQ(span[0], 1.0f);
    EXPECT_FLOAT_EQ(span[1], 2.0f);
    EXPECT_FLOAT_EQ(span[2], 3.0f);
    EXPECT_FLOAT_EQ(span[3], 4.0f);
}

TEST(TensorTest, GetDataAsWrongSize)
{
    Tensor t("w", DataType::FLOAT, TensorShape{{2, 2}});
    std::vector<uint8_t> raw(3 * sizeof(float));
    t.setRawData(std::move(raw));

    // getDataAs<float> should return span size 3 and validData = false
    EXPECT_FALSE(t.hasValidData());
    auto span = t.getDataAs<float>();
    EXPECT_EQ(span.size(), 3);
}

TEST(TensorTest, DataTypeSize)
{
    EXPECT_EQ(Tensor::dataTypeSize(DataType::FLOAT), sizeof(float));
    EXPECT_EQ(Tensor::dataTypeSize(DataType::INT32), sizeof(int32_t));
    EXPECT_EQ(Tensor::dataTypeSize(DataType::INT64), sizeof(int64_t));
    EXPECT_EQ(Tensor::dataTypeSize(DataType::UNDEFINED), 0);
}

TEST(TensorTest, DataTypeConversion)
{
    EXPECT_EQ(dataTypeFromOnnx(1), DataType::FLOAT);
    EXPECT_EQ(dataTypeFromOnnx(2), DataType::UINT8);
    EXPECT_EQ(dataTypeFromOnnx(100), DataType::UNDEFINED);
    EXPECT_EQ(dataTypeToString(DataType::FLOAT), "float32");
    EXPECT_EQ(dataTypeToString(DataType::INT64), "int64");
}

