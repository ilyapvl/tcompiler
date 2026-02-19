#include <gtest/gtest.h>
#include <reader/reader.h>
#include <graph/types.h>
#include <vector>
#include <cstring>
#include <variant>

using namespace graph;
using namespace graph::proto;


static void write_varint(std::vector<uint8_t>& buf, uint64_t v)
{
    do
    {
        uint8_t byte = v & 0x7F;
        v >>= 7;
        if (v) byte |= 0x80;
        buf.push_back(byte);
    } while (v);
}


TEST(ParseAttributeTest, FloatAttr)
{
    std::vector<uint8_t> buffer;

    write_varint(buffer, (1 << 3) | 2);
    std::string name = "alpha";
    write_varint(buffer, name.size());
    buffer.insert(buffer.end(), name.begin(), name.end());

    write_varint(buffer, (20 << 3) | 0);
    write_varint(buffer, 1);

    write_varint(buffer, (2 << 3) | 5);
    float f = 3.14f;
    uint32_t f_bits;
    std::memcpy(&f_bits, &f, sizeof(f));

    for (size_t i = 0; i < 4; ++i)
    {
        buffer.push_back((f_bits >> (i * 8)) & 0xFF);
    }

    auto [attr_name, attr] = parse_AttributeProto(buffer.data(), buffer.size());

    EXPECT_EQ(attr_name, "alpha");
    ASSERT_TRUE(std::holds_alternative<float>(attr));
    EXPECT_FLOAT_EQ(std::get<float>(attr), 3.14f);
}

// varint
TEST(ParseAttributeTest, IntAttr)
{
    std::vector<uint8_t> buffer;


    write_varint(buffer, (1 << 3) | 2);
    std::string name = "axis";
    write_varint(buffer, name.size());
    buffer.insert(buffer.end(), name.begin(), name.end());

 
    write_varint(buffer, (20 << 3) | 0);
    write_varint(buffer, 2);


    write_varint(buffer, (3 << 3) | 0);
    write_varint(buffer, 42);

    auto [attr_name, attr] = parse_AttributeProto(buffer.data(), buffer.size());

    EXPECT_EQ(attr_name, "axis");
    ASSERT_TRUE(std::holds_alternative<int64_t>(attr));
    EXPECT_EQ(std::get<int64_t>(attr), 42);
}




TEST(ParseAttributeTest, TensorAttr)
{
    std::vector<uint8_t> buffer;

    write_varint(buffer, (1 << 3) | 2);
    std::string name = "value";
    write_varint(buffer, name.size());
    buffer.insert(buffer.end(), name.begin(), name.end());

    write_varint(buffer, (20 << 3) | 0);
    write_varint(buffer, 4);

    std::vector<uint8_t> tensor_buf;
    {
        // [2,2]
        write_varint(tensor_buf, (1 << 3) | 2);
        std::vector<uint8_t> dims_data;
        write_varint(dims_data, 2);
        write_varint(dims_data, 2);
        write_varint(tensor_buf, dims_data.size());
        tensor_buf.insert(tensor_buf.end(), dims_data.begin(), dims_data.end());

        // float
        write_varint(tensor_buf, (2 << 3) | 0);
        write_varint(tensor_buf, 1);

        // raw 4 float
        write_varint(tensor_buf, (13 << 3) | 2);
        float vals[4] = {1.0f, 0.0f, 0.0f, 1.0f};
        write_varint(tensor_buf, sizeof(vals));
        tensor_buf.insert(tensor_buf.end(), reinterpret_cast<uint8_t*>(vals),
                          reinterpret_cast<uint8_t*>(vals) + sizeof(vals));
    }
    write_varint(buffer, (5 << 3) | 2);
    write_varint(buffer, tensor_buf.size());
    buffer.insert(buffer.end(), tensor_buf.begin(), tensor_buf.end());

    auto [attr_name, attr] = parse_AttributeProto(buffer.data(), buffer.size());

    EXPECT_EQ(attr_name, "value");
    ASSERT_TRUE(std::holds_alternative<TensorInfo>(attr));
    const TensorInfo& ti = std::get<TensorInfo>(attr);
    EXPECT_EQ(ti.data_type, DataType::FLOAT);
    ASSERT_EQ(ti.dims.size(), 2);
    EXPECT_EQ(ti.dims[0], 2);
    EXPECT_EQ(ti.dims[1], 2);
    ASSERT_EQ(ti.raw_data.size(), 4 * sizeof(float));
    const float* data = reinterpret_cast<const float*>(ti.raw_data.data());
    EXPECT_FLOAT_EQ(data[0], 1.0f);
    EXPECT_FLOAT_EQ(data[1], 0.0f);
    EXPECT_FLOAT_EQ(data[2], 0.0f);
    EXPECT_FLOAT_EQ(data[3], 1.0f);
}


TEST(ParseAttributeTest, FloatsPacked)
{
    std::vector<uint8_t> buffer;

    // name = "floats"
    write_varint(buffer, (1 << 3) | 2);
    std::string name = "floats";
    write_varint(buffer, name.size());
    buffer.insert(buffer.end(), name.begin(), name.end());

    write_varint(buffer, (20 << 3) | 0);
    write_varint(buffer, 6);

    // floats [1.0, 2.0, 3.0]
    write_varint(buffer, (6 << 3) | 2);
    float vals[3] = {1.0f, 2.0f, 3.0f};
    write_varint(buffer, sizeof(vals));
    buffer.insert(buffer.end(), reinterpret_cast<uint8_t*>(vals),
                  reinterpret_cast<uint8_t*>(vals) + sizeof(vals));

    auto [attr_name, attr] = parse_AttributeProto(buffer.data(), buffer.size());

    EXPECT_EQ(attr_name, "floats");
    ASSERT_TRUE(std::holds_alternative<std::vector<float>>(attr));
    const auto& vec = std::get<std::vector<float>>(attr);
    ASSERT_EQ(vec.size(), 3);
    EXPECT_FLOAT_EQ(vec[0], 1.0f);
    EXPECT_FLOAT_EQ(vec[1], 2.0f);
    EXPECT_FLOAT_EQ(vec[2], 3.0f);
}


TEST(ParseAttributeTest, IntsPacked)
{
    std::vector<uint8_t> buffer;

    write_varint(buffer, (1 << 3) | 2);
    std::string name = "ints";
    write_varint(buffer, name.size());
    buffer.insert(buffer.end(), name.begin(), name.end());

    write_varint(buffer, (20 << 3) | 0);
    write_varint(buffer, 7);

    // ints [3, 3]
    write_varint(buffer, (7 << 3) | 2);
    std::vector<uint8_t> ints_data;
    write_varint(ints_data, 3);
    write_varint(ints_data, 3);
    write_varint(buffer, ints_data.size());
    buffer.insert(buffer.end(), ints_data.begin(), ints_data.end());

    auto [attr_name, attr] = parse_AttributeProto(buffer.data(), buffer.size());

    EXPECT_EQ(attr_name, "ints");
    ASSERT_TRUE(std::holds_alternative<std::vector<int64_t>>(attr));
    const auto& vec = std::get<std::vector<int64_t>>(attr);
    ASSERT_EQ(vec.size(), 2);
    EXPECT_EQ(vec[0], 3);
    EXPECT_EQ(vec[1], 3);
}


TEST(ParseAttributeTest, UnsupportedType)
{
    std::vector<uint8_t> buffer;

    write_varint(buffer, (1 << 3) | 2);
    std::string name = "unsupported";
    write_varint(buffer, name.size());
    buffer.insert(buffer.end(), name.begin(), name.end());


    write_varint(buffer, (20 << 3) | 0);
    write_varint(buffer, 9);

    EXPECT_THROW(parse_AttributeProto(buffer.data(), buffer.size()), unsupported_error);
}


TEST(ParseAttributeTest, MissingValue)
{
    std::vector<uint8_t> buffer;

    write_varint(buffer, (1 << 3) | 2);
    std::string name = "nofield";
    write_varint(buffer, name.size());
    buffer.insert(buffer.end(), name.begin(), name.end());

    write_varint(buffer, (20 << 3) | 0);
    write_varint(buffer, 2);

    EXPECT_THROW(parse_AttributeProto(buffer.data(), buffer.size()), parse_error);
}
