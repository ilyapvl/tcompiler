#include <gtest/gtest.h>
#include <reader/reader.h>
#include <graph/types.h>
#include <vector>
#include <cstring>

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


TEST(ParseValueInfoTest, FixedDims)
{
    std::vector<uint8_t> buffer;

    // name="input"
    write_varint(buffer, (1 << 3) | 2);
    std::string name = "input";
    write_varint(buffer, name.size());
    buffer.insert(buffer.end(), name.begin(), name.end());

    // type=TypeProto
    std::vector<uint8_t> type_buf;
    {
        // tensor_type=TensorTypeProto
        std::vector<uint8_t> tensor_buf;
        {
            // elem_type=float
            write_varint(tensor_buf, (1 << 3) | 0);
            write_varint(tensor_buf, 1);

            // shape=TensorShapeProto
            std::vector<uint8_t> shape_buf;
            {
                int64_t dims[] = {1, 3, 224, 224};
                for (int64_t d : dims)
                {
                    std::vector<uint8_t> dim_buf;
                    {
                        write_varint(dim_buf, (1 << 3) | 0);
                        write_varint(dim_buf, static_cast<uint64_t>(d));
                    }
                    write_varint(shape_buf, (1 << 3) | 2);
                    write_varint(shape_buf, dim_buf.size());
                    shape_buf.insert(shape_buf.end(), dim_buf.begin(), dim_buf.end());
                }
            }
            write_varint(tensor_buf, (2 << 3) | 2);
            write_varint(tensor_buf, shape_buf.size());
            tensor_buf.insert(tensor_buf.end(), shape_buf.begin(), shape_buf.end());
        }
        write_varint(type_buf, (1 << 3) | 2);
        write_varint(type_buf, tensor_buf.size());
        type_buf.insert(type_buf.end(), tensor_buf.begin(), tensor_buf.end());
    }
    write_varint(buffer, (2 << 3) | 2);
    write_varint(buffer, type_buf.size());
    buffer.insert(buffer.end(), type_buf.begin(), type_buf.end());

    TensorInfo info = parse_ValueInfoProto(buffer.data(), buffer.size());

    EXPECT_EQ(info.name, "input");
    EXPECT_EQ(info.data_type, DataType::FLOAT);
    ASSERT_EQ(info.dims.size(), 4);
    EXPECT_EQ(info.dims[0], 1);
    EXPECT_EQ(info.dims[1], 3);
    EXPECT_EQ(info.dims[2], 224);
    EXPECT_EQ(info.dims[3], 224);
}


TEST(ParseValueInfoTest, DynamicDim)
{
    std::vector<uint8_t> buffer;

    // name="dynamic"
    write_varint(buffer, (1 << 3) | 2);
    std::string name = "dynamic";
    write_varint(buffer, name.size());
    buffer.insert(buffer.end(), name.begin(), name.end());

    // type
    std::vector<uint8_t> type_buf;
    {
        std::vector<uint8_t> tensor_buf;
        {
            // elem_type=float
            write_varint(tensor_buf, (1 << 3) | 0);
            write_varint(tensor_buf, 1);

            // shape
            std::vector<uint8_t> shape_buf;
            {
                // dim_param="dynamic"
                std::vector<uint8_t> dim_buf;
                {
                    write_varint(dim_buf, (2 << 3) | 2);
                    std::string param = "dynamic";
                    write_varint(dim_buf, param.size());
                    dim_buf.insert(dim_buf.end(), param.begin(), param.end());
                }
                write_varint(shape_buf, (1 << 3) | 2);
                write_varint(shape_buf, dim_buf.size());
                shape_buf.insert(shape_buf.end(), dim_buf.begin(), dim_buf.end());
            }
            write_varint(tensor_buf, (2 << 3) | 2);
            write_varint(tensor_buf, shape_buf.size());
            tensor_buf.insert(tensor_buf.end(), shape_buf.begin(), shape_buf.end());
        }
        write_varint(type_buf, (1 << 3) | 2);
        write_varint(type_buf, tensor_buf.size());
        type_buf.insert(type_buf.end(), tensor_buf.begin(), tensor_buf.end());
    }
    write_varint(buffer, (2 << 3) | 2);
    write_varint(buffer, type_buf.size());
    buffer.insert(buffer.end(), type_buf.begin(), type_buf.end());

    TensorInfo info = parse_ValueInfoProto(buffer.data(), buffer.size());

    EXPECT_EQ(info.name, "dynamic");
    EXPECT_EQ(info.data_type, DataType::FLOAT);
    ASSERT_EQ(info.dims.size(), 1);
    EXPECT_EQ(info.dims[0], -1);
}


TEST(ParseValueInfoTest, UnknownDataType)
{
    std::vector<uint8_t> buffer;

    // name
    write_varint(buffer, (1 << 3) | 2);
    std::string name = "unknown";
    write_varint(buffer, name.size());
    buffer.insert(buffer.end(), name.begin(), name.end());

    // type
    std::vector<uint8_t> type_buf;
    {
        std::vector<uint8_t> tensor_buf;
        {
            // elem_type=undefined
            write_varint(tensor_buf, (1 << 3) | 0);
            write_varint(tensor_buf, 0);
        }
        write_varint(type_buf, (1 << 3) | 2);
        write_varint(type_buf, tensor_buf.size());
        type_buf.insert(type_buf.end(), tensor_buf.begin(), tensor_buf.end());
    }
    write_varint(buffer, (2 << 3) | 2);
    write_varint(buffer, type_buf.size());
    buffer.insert(buffer.end(), type_buf.begin(), type_buf.end());

    TensorInfo info = parse_ValueInfoProto(buffer.data(), buffer.size());

    EXPECT_EQ(info.name, "unknown");
    EXPECT_EQ(info.data_type, DataType::UNDEFINED);
    EXPECT_TRUE(info.dims.empty());
}


TEST(ParseValueInfoTest, NoType)
{
    std::vector<uint8_t> buffer;

    // name
    write_varint(buffer, (1 << 3) | 2);
    std::string name = "notype";
    write_varint(buffer, name.size());
    buffer.insert(buffer.end(), name.begin(), name.end());

    TensorInfo info = parse_ValueInfoProto(buffer.data(), buffer.size());

    EXPECT_EQ(info.name, "notype");
    EXPECT_EQ(info.data_type, DataType::UNDEFINED);
    EXPECT_TRUE(info.dims.empty());
}


TEST(ParseValueInfoTest, MixedDims)
{
    std::vector<uint8_t> buffer;

    write_varint(buffer, (1 << 3) | 2);
    std::string name = "mixed";
    write_varint(buffer, name.size());
    buffer.insert(buffer.end(), name.begin(), name.end());

    std::vector<uint8_t> type_buf;
    {
        std::vector<uint8_t> tensor_buf;
        {
            // elem_type=int64
            write_varint(tensor_buf, (1 << 3) | 0);
            write_varint(tensor_buf, 7);

            // shape [2, ?, 128]
            std::vector<uint8_t> shape_buf;
            {
                // dim1: dim_value = 2
                {
                    std::vector<uint8_t> dim_buf;
                    write_varint(dim_buf, (1 << 3) | 0);
                    write_varint(dim_buf, 2);
                    write_varint(shape_buf, (1 << 3) | 2);
                    write_varint(shape_buf, dim_buf.size());
                    shape_buf.insert(shape_buf.end(), dim_buf.begin(), dim_buf.end());
                }
                // dim2: dim_param = "dynamic"
                {
                    std::vector<uint8_t> dim_buf;
                    write_varint(dim_buf, (2 << 3) | 2);
                    std::string param = "dynamic";
                    write_varint(dim_buf, param.size());
                    dim_buf.insert(dim_buf.end(), param.begin(), param.end());
                    write_varint(shape_buf, (1 << 3) | 2);
                    write_varint(shape_buf, dim_buf.size());
                    shape_buf.insert(shape_buf.end(), dim_buf.begin(), dim_buf.end());
                }
                // dim3: dim_value = 128
                {
                    std::vector<uint8_t> dim_buf;
                    write_varint(dim_buf, (1 << 3) | 0);
                    write_varint(dim_buf, 128);
                    write_varint(shape_buf, (1 << 3) | 2);
                    write_varint(shape_buf, dim_buf.size());
                    shape_buf.insert(shape_buf.end(), dim_buf.begin(), dim_buf.end());
                }
            }
            write_varint(tensor_buf, (2 << 3) | 2);
            write_varint(tensor_buf, shape_buf.size());
            tensor_buf.insert(tensor_buf.end(), shape_buf.begin(), shape_buf.end());
        }
        write_varint(type_buf, (1 << 3) | 2);
        write_varint(type_buf, tensor_buf.size());
        type_buf.insert(type_buf.end(), tensor_buf.begin(), tensor_buf.end());
    }
    write_varint(buffer, (2 << 3) | 2);
    write_varint(buffer, type_buf.size());
    buffer.insert(buffer.end(), type_buf.begin(), type_buf.end());

    TensorInfo info = parse_ValueInfoProto(buffer.data(), buffer.size());

    EXPECT_EQ(info.name, "mixed");
    EXPECT_EQ(info.data_type, DataType::INT64);
    ASSERT_EQ(info.dims.size(), 3);
    EXPECT_EQ(info.dims[0], 2);
    EXPECT_EQ(info.dims[1], -1);
    EXPECT_EQ(info.dims[2], 128);
}
