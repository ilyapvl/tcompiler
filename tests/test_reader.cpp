#include <gtest/gtest.h>
#include <reader/reader.h>
#include <vector>

using namespace graph::proto;

TEST(ProtoReaderTest, ReadVarint)
{
    std::vector<uint8_t> data = {0xAC, 0x02}; // varint 300
    ProtoReader reader(data.data(), data.size());
    EXPECT_EQ(reader.read_varint(), 300);
    EXPECT_TRUE(reader.eof());
}


TEST(ProtoReaderTest, ReadKey)
{
    std::vector<uint8_t> data = {0x08}; // field 1, wire type 0
    ProtoReader reader(data.data(), data.size());
    auto [field, wire] = reader.read_key();
    EXPECT_EQ(field, 1);
    EXPECT_EQ(wire, 0);
    EXPECT_TRUE(reader.eof());
}


TEST(ProtoReaderTest, SkipField)
{
    std::vector<uint8_t> data = {0x08, 0x96, 0x01}; // varint 150
    ProtoReader reader(data.data(), data.size());
    reader.read_key(); 
    reader.skip_field(0);
    EXPECT_TRUE(reader.eof());
}


TEST(ProtoReaderTest, OutOfBounds)
{
    std::vector<uint8_t> data = {0x08}; 
    ProtoReader reader(data.data(), data.size());
    EXPECT_NO_THROW(reader.read_varint());
    EXPECT_THROW(reader.read_varint(), graph::parse_error);
}
