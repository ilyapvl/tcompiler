#include <gtest/gtest.h>
#include <reader/reader.h>
#include <graph/types.h>
#include <cstring>
#include <vector>

using namespace graph;
using namespace graph::proto;

// создание varint
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

TEST(ParseTensorTest, RawData)
{
    std::vector<uint8_t> buffer;

    // [2,3]
    write_varint(buffer, (1 << 3) | 2); 
    std::vector<uint8_t> dims_data;
    write_varint(dims_data, 2);
    write_varint(dims_data, 3);
    write_varint(buffer, dims_data.size());
    buffer.insert(buffer.end(), dims_data.begin(), dims_data.end());

    // float
    write_varint(buffer, (2 << 3) | 0);
    write_varint(buffer, 1);

    // name=test
    write_varint(buffer, (8 << 3) | 2);
    std::string name = "test";
    write_varint(buffer, name.size());
    buffer.insert(buffer.end(), name.begin(), name.end());

    // raw
    write_varint(buffer, (13 << 3) | 2);
    float fdata[2] = {1.25f, 2.5f};
    write_varint(buffer, sizeof(fdata));
    buffer.insert(buffer.end(), reinterpret_cast<uint8_t*>(fdata), 
                  reinterpret_cast<uint8_t*>(fdata) + sizeof(fdata));

    TensorInfo ti = parse_TensorInfo(buffer.data(), buffer.size());

    EXPECT_EQ(ti.name, "test");
    EXPECT_EQ(ti.data_type, DataType::FLOAT);
    ASSERT_EQ(ti.dims.size(), 2);
    EXPECT_EQ(ti.dims[0], 2);
    EXPECT_EQ(ti.dims[1], 3);
    ASSERT_EQ(ti.raw_data.size(), 8);
    const float* parsed = reinterpret_cast<const float*>(ti.raw_data.data());
    EXPECT_FLOAT_EQ(parsed[0], 1.25f);
    EXPECT_FLOAT_EQ(parsed[1], 2.5f);
    EXPECT_TRUE(ti.is_constant);
}

TEST(ParseTensorTest, FloatDataPacked)
{
    std::vector<uint8_t> buffer;

    // [4]
    write_varint(buffer, (1 << 3) | 2);
    std::vector<uint8_t> dims_data;
    write_varint(dims_data, 4);
    write_varint(buffer, dims_data.size());
    buffer.insert(buffer.end(), dims_data.begin(), dims_data.end());

    // float
    write_varint(buffer, (2 << 3) | 0);
    write_varint(buffer, 1);

    // name=float_tensor
    write_varint(buffer, (8 << 3) | 2);
    std::string name = "float_tensor";
    write_varint(buffer, name.size());
    buffer.insert(buffer.end(), name.begin(), name.end());

    // data=[1.0, 2.0, 3.0, 4.0]
    write_varint(buffer, (4 << 3) | 2);
    float vals[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    write_varint(buffer, sizeof(vals));
    buffer.insert(buffer.end(), reinterpret_cast<uint8_t*>(vals), 
                  reinterpret_cast<uint8_t*>(vals) + sizeof(vals));

    TensorInfo ti = parse_TensorInfo(buffer.data(), buffer.size());

    EXPECT_EQ(ti.name, "float_tensor");
    EXPECT_EQ(ti.data_type, DataType::FLOAT);
    ASSERT_EQ(ti.dims.size(), 1);
    EXPECT_EQ(ti.dims[0], 4);
    ASSERT_EQ(ti.raw_data.size(), 16);
    const float* parsed = reinterpret_cast<const float*>(ti.raw_data.data());
    for (int i = 0; i < 4; ++i)
    {
        EXPECT_FLOAT_EQ(parsed[i], static_cast<float>(i+1));
    }
}

TEST(ParseTensorTest, Int32DataPacked)
{
    std::vector<uint8_t> buffer;

    // [3]
    write_varint(buffer, (1 << 3) | 2);
    std::vector<uint8_t> dims_data;
    write_varint(dims_data, 3);
    write_varint(buffer, dims_data.size());
    buffer.insert(buffer.end(), dims_data.begin(), dims_data.end());

    // int32
    write_varint(buffer, (2 << 3) | 0);
    write_varint(buffer, 6);

    // name=int32_tensor
    write_varint(buffer, (8 << 3) | 2);
    std::string name = "int32_tensor";
    write_varint(buffer, name.size());
    buffer.insert(buffer.end(), name.begin(), name.end());

    // [100, 200, 300] как varint
    write_varint(buffer, (5 << 3) | 2);
    std::vector<uint8_t> ints_data;
    write_varint(ints_data, 100);
    write_varint(ints_data, 200);
    write_varint(ints_data, 300);
    write_varint(buffer, ints_data.size());
    buffer.insert(buffer.end(), ints_data.begin(), ints_data.end());

    TensorInfo ti = parse_TensorInfo(buffer.data(), buffer.size());

    EXPECT_EQ(ti.data_type, DataType::INT32);
    ASSERT_EQ(ti.dims.size(), 1);
    EXPECT_EQ(ti.dims[0], 3);
    ASSERT_EQ(ti.raw_data.size(), 3 * sizeof(int32_t));
    const int32_t* parsed = reinterpret_cast<const int32_t*>(ti.raw_data.data());
    EXPECT_EQ(parsed[0], 100);
    EXPECT_EQ(parsed[1], 200);
    EXPECT_EQ(parsed[2], 300);
}


TEST(ParseTensorTest, Int64DataPacked)
{
    std::vector<uint8_t> buffer;

    // [2]
    write_varint(buffer, (1 << 3) | 2);
    std::vector<uint8_t> dims_data;
    write_varint(dims_data, 2);
    write_varint(buffer, dims_data.size());
    buffer.insert(buffer.end(), dims_data.begin(), dims_data.end());

    // int64
    write_varint(buffer, (2 << 3) | 0);
    write_varint(buffer, 7);

    // name=int64_tensor
    write_varint(buffer, (8 << 3) | 2);
    std::string name = "int64_tensor";
    write_varint(buffer, name.size());
    buffer.insert(buffer.end(), name.begin(), name.end());

    // [1000000, 2000000] как varint
    write_varint(buffer, (6 << 3) | 2);
    std::vector<uint8_t> ints_data;
    write_varint(ints_data, 1000000);
    write_varint(ints_data, 2000000);
    write_varint(buffer, ints_data.size());
    buffer.insert(buffer.end(), ints_data.begin(), ints_data.end());

    TensorInfo ti = parse_TensorInfo(buffer.data(), buffer.size());

    EXPECT_EQ(ti.data_type, DataType::INT64);
    ASSERT_EQ(ti.dims.size(), 1);
    EXPECT_EQ(ti.dims[0], 2);
    ASSERT_EQ(ti.raw_data.size(), 2 * sizeof(int64_t));
    const int64_t* parsed = reinterpret_cast<const int64_t*>(ti.raw_data.data());
    EXPECT_EQ(parsed[0], 1000000);
    EXPECT_EQ(parsed[1], 2000000);
}


TEST(ParseTensorTest, MissingName)
{
    std::vector<uint8_t> buffer;

    // [1]
    write_varint(buffer, (1 << 3) | 2);
    std::vector<uint8_t> dims_data;
    write_varint(dims_data, 1);
    write_varint(buffer, dims_data.size());
    buffer.insert(buffer.end(), dims_data.begin(), dims_data.end());

    // float
    write_varint(buffer, (2 << 3) | 0);
    write_varint(buffer, 1);

    // raw
    write_varint(buffer, (13 << 3) | 2);
    float f = 42.0f;
    write_varint(buffer, sizeof(f));
    buffer.insert(buffer.end(), reinterpret_cast<uint8_t*>(&f), 
                  reinterpret_cast<uint8_t*>(&f) + sizeof(f));

    TensorInfo ti = parse_TensorInfo(buffer.data(), buffer.size());
    EXPECT_TRUE(ti.name.empty());
    EXPECT_EQ(ti.data_type, DataType::FLOAT);
    ASSERT_EQ(ti.dims.size(), 1);
    EXPECT_EQ(ti.dims[0], 1);
    ASSERT_EQ(ti.raw_data.size(), 4);
}


TEST(ParseTensorTest, UnsupportedDataType)
{
    std::vector<uint8_t> buffer;

    write_varint(buffer, (2 << 3) | 0);
    write_varint(buffer, 0);

    TensorInfo ti = parse_TensorInfo(buffer.data(), buffer.size());
    EXPECT_EQ(ti.data_type, DataType::UNDEFINED);
}
