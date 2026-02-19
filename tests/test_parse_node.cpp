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

TEST(ParseNodeTest, MultipleAttributes)
{
    std::vector<uint8_t> buffer;

    write_varint(buffer, (4 << 3) | 2);
    std::string op_type = "test";
    write_varint(buffer, op_type.size());
    buffer.insert(buffer.end(), op_type.begin(), op_type.end());

    // int attr=1
    {
        std::vector<uint8_t> attr_buf;
        write_varint(attr_buf, (1 << 3) | 2);
        std::string attr_name = "int_attr";
        write_varint(attr_buf, attr_name.size());
        attr_buf.insert(attr_buf.end(), attr_name.begin(), attr_name.end());

        write_varint(attr_buf, (20 << 3) | 0);
        write_varint(attr_buf, 2);

        write_varint(attr_buf, (3 << 3) | 0);
        write_varint(attr_buf, 1);

        write_varint(buffer, (5 << 3) | 2);
        write_varint(buffer, attr_buf.size());
        buffer.insert(buffer.end(), attr_buf.begin(), attr_buf.end());
    }

    // float attr=1.0f
    {
        std::vector<uint8_t> attr_buf;
        write_varint(attr_buf, (1 << 3) | 2);
        std::string attr_name = "float_attr";
        write_varint(attr_buf, attr_name.size());
        attr_buf.insert(attr_buf.end(), attr_name.begin(), attr_name.end());

        write_varint(attr_buf, (20 << 3) | 0);
        write_varint(attr_buf, 1);

        write_varint(attr_buf, (2 << 3) | 5);
        float f = 1.0f;
        uint32_t bits = 0;
        std::memcpy(&bits, &f, sizeof(f));
        for (size_t i = 0; i < 4; ++i) attr_buf.push_back((bits >> (i * 8)) & 0xFF);

        write_varint(buffer, (5 << 3) | 2);
        write_varint(buffer, attr_buf.size());
        buffer.insert(buffer.end(), attr_buf.begin(), attr_buf.end());
    }

    NodeProtoInfo node = parse_NodeProto(buffer.data(), buffer.size());

    EXPECT_EQ(node.op_type, "test");
    ASSERT_EQ(node.attributes.size(), 2);

    bool found_int = false, found_float = false;
    for (const auto& [key, val] : node.attributes)
    {
        if (key == "int_attr")
        {
            found_int = true;
            ASSERT_TRUE(std::holds_alternative<int64_t>(val));
            EXPECT_EQ(std::get<int64_t>(val), 1);
        }
        
        else if (key == "float_attr")
        {
            found_float = true;
            ASSERT_TRUE(std::holds_alternative<float>(val));
            EXPECT_FLOAT_EQ(std::get<float>(val), 1.0f);
        }
    }
    EXPECT_TRUE(found_int);
    EXPECT_TRUE(found_float);
}



TEST(ParseNodeTest, CorruptedAttribute)
{
    std::vector<uint8_t> buffer;

    write_varint(buffer, (4 << 3) | 2);
    std::string op_type = "test";
    write_varint(buffer, op_type.size());
    buffer.insert(buffer.end(), op_type.begin(), op_type.end());


    write_varint(buffer, (5 << 3) | 2);
    write_varint(buffer, 3); 
    buffer.push_back(0x012); // random so should be ignored
    buffer.push_back(0x023); //
    buffer.push_back(0x067); //


    NodeProtoInfo node = parse_NodeProto(buffer.data(), buffer.size());

    EXPECT_EQ(node.op_type, "test");
    EXPECT_TRUE(node.attributes.empty());
}
