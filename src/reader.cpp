#include <reader/reader.h>
#include <iostream>
#include <cstring>
#include <limits>


static graph::DataType onnx_data_type_to_enum(int32_t onnx_type)
{
    switch (onnx_type)
    {
        case 1:
            return graph::DataType::FLOAT;
        case 6:
            return graph::DataType::INT32;
        case 7:
            return graph::DataType::INT64;
        default:
            return graph::DataType::UNDEFINED;
    }
}

namespace graph
{
namespace proto
{

ProtoReader::ProtoReader(const uint8_t* data, size_t size)
    : data_(data), pos_(0), end_(size) {}

ProtoReader::ProtoReader(const std::vector<uint8_t>& data)
    : data_(data.data()), pos_(0), end_(data.size()) {}

void ProtoReader::check_bound(size_t needed) const
{
    if (pos_ + needed > end_)
        throw parse_error("Unexpected end of protobuf data");
}

std::pair<uint32_t, int> ProtoReader::read_key()
{
    uint64_t value = read_varint();
    uint32_t field_number = static_cast<uint32_t>(value >> 3);
    int wire_type = static_cast<int>(value & 0x07);
    
    return {field_number, wire_type};
}

uint64_t ProtoReader::read_varint()
{
    uint64_t result = 0;
    int shift = 0;
    while (true)
    {
        check_bound(1);
        uint8_t byte = data_[pos_++];
        result |= (static_cast<uint64_t>(byte & 0x7F) << shift);
        if ((byte & 0x80) == 0) break;
        shift += 7;
        if (shift > 63)
            throw parse_error("Varint too long");
    }
    return result;
}

std::pair<const uint8_t*, size_t> ProtoReader::read_length_delimited()
{
    uint64_t length = read_varint();
    if (length > std::numeric_limits<size_t>::max() || length > (end_ - pos_))
        throw parse_error("Invalid length in length-delimited field");
    check_bound(static_cast<size_t>(length));
    const uint8_t* ptr = data_ + pos_;
    pos_ += static_cast<size_t>(length);
    
    return {ptr, static_cast<size_t>(length)};
}

std::string ProtoReader::read_string()
{
    auto [ptr, len] = read_length_delimited();
    return std::string(reinterpret_cast<const char*>(ptr), len);
}

uint32_t ProtoReader::read_fixed32()
{
    check_bound(4);
    uint32_t value;
    std::memcpy(&value, data_ + pos_, 4);
    pos_ += 4;
    return value;
}

uint64_t ProtoReader::read_fixed64()
{
    check_bound(8);
    uint64_t value;
    std::memcpy(&value, data_ + pos_, 8);
    pos_ += 8;
    return value;
}

void ProtoReader::skip_field(int wire_type)
{
    switch (wire_type)
    {
        case 0:
            read_varint();
            break;
        case 1:
            check_bound(8);
            pos_ += 8;
            break;
        case 2:
        {
            uint64_t length = read_varint();
            if (length > (end_ - pos_))
                throw parse_error("Invalid length in skip_field");
            pos_ += static_cast<size_t>(length);
            break;
        }
        case 5:
            check_bound(4);
            pos_ += 4;
            break;
        default:
            throw parse_error("Unsupported wire type for skipping: " + std::to_string(wire_type));
    }
}



void parse_message(ProtoReader& reader,
                   const SimpleFieldHandler& on_simple,
                   const LengthDelimitedHandler& on_length_delimited)
{
    while (!reader.eof())
    {
        auto [field_number, wire_type] = reader.read_key();
        switch (wire_type)
        {
            case 0:
            {
                uint64_t value = reader.read_varint();
                if (on_simple) on_simple(field_number, wire_type, value);
                break;
            }
            case 1:
            {
                uint64_t value = reader.read_fixed64();
                if (on_simple) on_simple(field_number, wire_type, value);
                break;
            }
            case 2:
            {
                auto [data, size] = reader.read_length_delimited();
                if (on_length_delimited) on_length_delimited(field_number, data, size);
                break;
            }
            case 5:
            {
                uint32_t value = reader.read_fixed32();
                if (on_simple) on_simple(field_number, wire_type, static_cast<uint64_t>(value));
                break;
            }
            case 3:
            {
                reader.skip_group();
                break;
            }
            case 4:
                break;
            default:
                throw parse_error("Unsupported wire type in parse_message: " + std::to_string(wire_type));
        }
    }
}


void ProtoReader::skip_group()
{
    int depth = 1;
    while (depth > 0 && !eof())
    {
        auto [field_number, wire_type] = read_key();
        if (wire_type == 3)
        {
            depth++;
        }
        else if (wire_type == 4)
        {
            depth--;
        }
        else
        {
            skip_field(wire_type);
        }
    }
}



TensorInfo parse_TensorInfo(const uint8_t* data, size_t size)
{
    ProtoReader reader(data, size);
    TensorInfo ti;
    ti.data_type = DataType::UNDEFINED;
    ti.is_constant = true;

    std::vector<float> float_data;
    std::vector<int32_t> int32_data;
    std::vector<int64_t> int64_data;

    auto on_simple = [&](uint32_t field_number, int wire_type, uint64_t value)
    {
        switch (field_number)
        {
            case 2: //type
                if (wire_type == 0)
                {
                    ti.data_type = onnx_data_type_to_enum(static_cast<int32_t>(value));
                }
                else
                {
                    throw parse_error("data_type field has wrong wire type");
                }
                break;
            case 4: //float
                if (wire_type == 5)
                {
                    float_data.push_back(*reinterpret_cast<const float*>(&value));
                }
                else
                {
                    throw parse_error("float_data field has wrong wire type");
                }
                break;
            case 5: //int32
                if (wire_type == 0)
                {
                    int32_data.push_back(static_cast<int32_t>(value));
                }
                else
                {
                    throw parse_error("int32_data field has wrong wire type");
                }
                break;
            case 6: //int64
                if (wire_type == 0)
                {
                    int64_data.push_back(static_cast<int64_t>(value));
                }
                else
                {
                    throw parse_error("int64_data field has wrong wire type");
                }
                break;
            
            default:
                break;
        }
    };

    auto on_length_delimited = [&](uint32_t field_number, const uint8_t* data, size_t size)
    {
        switch (field_number)
        {
            case 1: //dims
            {
                ProtoReader dims_reader(data, size);
                while (!dims_reader.eof())
                {
                    ti.dims.push_back(static_cast<int64_t>(dims_reader.read_varint()));
                }
                break;
            }
            case 4: //float
            {
                ProtoReader float_reader(data, size);
                while (!float_reader.eof())
                {
                    uint32_t bits = float_reader.read_fixed32();
                    float f;
                    std::memcpy(&f, &bits, sizeof(f));
                    float_data.push_back(f);
                }
                break;
            }
            case 5: //int32
            {
                ProtoReader int32_reader(data, size);
                while (!int32_reader.eof())
                {
                    int32_data.push_back(static_cast<int32_t>(int32_reader.read_varint()));
                }
                break;
            }
            case 6: //int64
            {
                ProtoReader int64_reader(data, size);
                while (!int64_reader.eof())
                {
                    int64_data.push_back(static_cast<int64_t>(int64_reader.read_varint()));
                }
                break;
            }
            case 7: //name
                ti.name.assign(reinterpret_cast<const char*>(data), size);
                break;
            case 13: //raw
                ti.raw_data.assign(data, data + size);
                break;
            default:
                break;
        }
    };

    parse_message(reader, on_simple, on_length_delimited);

    if (ti.raw_data.empty())
    {
        if (!float_data.empty())
        {
            ti.raw_data.resize(float_data.size() * sizeof(float));
            std::memcpy(ti.raw_data.data(), float_data.data(), ti.raw_data.size());
        }
        else if (!int32_data.empty())
        {
            ti.raw_data.resize(int32_data.size() * sizeof(int32_t));
            std::memcpy(ti.raw_data.data(), int32_data.data(), ti.raw_data.size());
        }
        else if (!int64_data.empty())
        {
            ti.raw_data.resize(int64_data.size() * sizeof(int64_t));
            std::memcpy(ti.raw_data.data(), int64_data.data(), ti.raw_data.size());
        }
    }

    return ti;
}




void parse_TensorShapeProto(ProtoReader& reader, std::vector<int64_t>& dims)
{
    while (!reader.eof())
    {
        auto [field_number, wire_type] = reader.read_key();
        if (field_number == 1 && wire_type == 2) //dims
        {
            auto [dim_data, dim_size] = reader.read_length_delimited();
            ProtoReader dim_reader(dim_data, dim_size);
            int64_t dim_val = -1;
            while (!dim_reader.eof())
            {
                auto [dim_field, dim_wire] = dim_reader.read_key();
                if (dim_field == 1 && dim_wire == 0) //dimval
                {
                    dim_val = static_cast<int64_t>(dim_reader.read_varint());
                }
                else if (dim_field == 2 && dim_wire == 2) //dimpar
                {
                    dim_reader.read_string();
                    dim_val = -1;
                }
                else
                {
                    dim_reader.skip_field(dim_wire);
                }
            }
            dims.push_back(dim_val);
        }
        else
        {
            reader.skip_field(wire_type);
        }
    }
}

void parse_TypeProto(ProtoReader& reader, DataType& elem_type, std::vector<int64_t>& dims)
{
    while (!reader.eof())
    {
        auto [field_number, wire_type] = reader.read_key();
        if (field_number == 1 && wire_type == 2) //ttype
        {
            auto [tensor_data, tensor_size] = reader.read_length_delimited();
            ProtoReader tensor_reader(tensor_data, tensor_size);
            while (!tensor_reader.eof())
            {
                auto [tensor_field, tensor_wire] = tensor_reader.read_key();
                if (tensor_field == 1 && tensor_wire == 0) //eltype
                {
                    elem_type = onnx_data_type_to_enum(static_cast<int32_t>(tensor_reader.read_varint()));
                }
                else if (tensor_field == 2 && tensor_wire == 2) //shape
                {
                    auto [shape_data, shape_size] = tensor_reader.read_length_delimited();
                    ProtoReader shape_reader(shape_data, shape_size);
                    parse_TensorShapeProto(shape_reader, dims);
                }
                else
                {
                    tensor_reader.skip_field(tensor_wire);
                }
            }
        }
        else
        {
            reader.skip_field(wire_type);
        }
    }
}

TensorInfo parse_ValueInfoProto(const uint8_t* data, size_t size)
{
    ProtoReader reader(data, size);
    TensorInfo info;
    info.data_type = DataType::UNDEFINED;

    auto on_simple = [&](uint32_t field_number, int wire_type, uint64_t value) {};

    auto on_length_delimited = [&](uint32_t field_number, const uint8_t* data, size_t size)
    {
        if (field_number == 1) //name
        {
            info.name.assign(reinterpret_cast<const char*>(data), size);
        }
        else if (field_number == 2) //type
        {
            ProtoReader type_reader(data, size);
            parse_TypeProto(type_reader, info.data_type, info.dims);
        }
    };

    parse_message(reader, on_simple, on_length_delimited);

    return info;
}



std::pair<std::string, Attribute> parse_AttributeProto(const uint8_t* data, size_t size)
{
    ProtoReader reader(data, size);

    struct RawAttribute
    {
        std::string name;
        int type = 0;
        std::optional<float> f;
        std::optional<int64_t> i;
        std::optional<std::string> s;
        std::optional<std::pair<const uint8_t*, size_t>> t;
        std::vector<float> floats;
        std::vector<int64_t> ints;
        std::vector<std::string> strings;
    } raw;

    auto on_simple = [&](uint32_t field_number, int wire_type, uint64_t value)
    {
        switch (field_number)
        {
            case 2: //float
                if (wire_type == 5)
                {
                    uint32_t bits = static_cast<uint32_t>(value);
                    float f;
                    std::memcpy(&f, &bits, sizeof(f));
                    raw.f = f;
                }
                else
                {
                    throw parse_error("Field f has wrong wire type");
                }

                break;

            case 3: //int
                if (wire_type == 0)
                {
                    raw.i = static_cast<int64_t>(value);
                }

                else
                {
                    throw parse_error("Field i has wrong wire type");
                }

                break;

            case 6: //floats

                if (wire_type == 5)
                {
                    uint32_t bits = static_cast<uint32_t>(value);
                    float f;
                    std::memcpy(&f, &bits, sizeof(f));
                    raw.floats.push_back(f);
                }

                else
                {
                    throw parse_error("Unpacked floats field has wrong wire type");
                }

                break;
            
            case 8: //ints
                if (wire_type == 0)
                {
                    raw.ints.push_back(static_cast<int64_t>(value));
                }
            case 20: //type
                raw.type = static_cast<int>(value);

                break;
            default:
                break;
        }
    };

    auto on_length_delimited = [&](uint32_t field_number, const uint8_t* data, size_t size)
    {
        switch (field_number)
        {
            case 1: //name
                raw.name.assign(reinterpret_cast<const char*>(data), size);

                break;

            case 4: //str
                raw.s = std::string(reinterpret_cast<const char*>(data), size);

                break;

            case 5: //tensor
                raw.t = std::make_pair(data, size);

                break;

            case 6: //floats
            {
                ProtoReader float_reader(data, size);
                while (!float_reader.eof())
                {
                    uint32_t bits = float_reader.read_fixed32();
                    float f;
                    std::memcpy(&f, &bits, sizeof(f));
                    raw.floats.push_back(f);
                }

                break;
            }
            case 7: //ints
            {
                {
                    ProtoReader int_reader(data, size);
                    while (!int_reader.eof())
                    {
                        uint64_t v = int_reader.read_varint();
                        raw.ints.push_back(static_cast<int64_t>(v));
                    }
                }

                break;
            }
            
            default:
                break;
        }
    };

    parse_message(reader, on_simple, on_length_delimited);

    Attribute attr;
    switch (raw.type)
    {
        case 1:
            if (raw.f.has_value()) attr = *raw.f;
            else throw parse_error("Attribute of type FLOAT missing value");

            break;
        case 2:
            if (raw.i.has_value()) attr = *raw.i;
            else throw parse_error("Attribute of type INT missing value");

            break;
        case 3:
            if (raw.s.has_value()) attr = *raw.s;
            else throw parse_error("Attribute of type STRING missing value");

            break;
        case 4:
            if (raw.t.has_value())
            {
                attr = parse_TensorInfo(raw.t->first, raw.t->second);
            }
            else throw parse_error("Attribute of type TENSOR missing value");
            
            break;
        case 6:
            attr = raw.floats;

            break;
        case 7:
            attr = raw.ints;

            break;
        case 8:
            attr = raw.strings;

            break;
        default:
            throw unsupported_error("Unsupported attribute type: " + std::to_string(raw.type));
    }

    return {raw.name, attr};
}








NodeProtoInfo parse_NodeProto(const uint8_t* data, size_t size)
{
    ProtoReader reader(data, size);
    NodeProtoInfo info;

    auto on_simple = [&](uint32_t field_number, int wire_type, uint64_t value){};

    auto on_length_delimited = [&](uint32_t field_number, const uint8_t* data, size_t size)
    {
        switch (field_number)
        {
            case 1:
                info.inputs.emplace_back(reinterpret_cast<const char*>(data), size);
                break;
            case 2:
                info.outputs.emplace_back(reinterpret_cast<const char*>(data), size);
                break;
            case 3:
                info.name.assign(reinterpret_cast<const char*>(data), size);
                break;
            case 4:
                info.op_type.assign(reinterpret_cast<const char*>(data), size);
                break;
            case 5:
                try
                {
                    auto [attr_name, attr] = parse_AttributeProto(data, size);
                    info.attributes.emplace_back(std::move(attr_name), std::move(attr));
                }
                catch (graph::parse_error& e)
                {
                    std::cout << e.what() << std::endl;
                }
            
                break;
            default:
                break;
        }
    };

    parse_message(reader, on_simple, on_length_delimited);
    return info;
}





GraphProtoInfo parse_GraphProto(const uint8_t* data, size_t size)
{
    ProtoReader reader(data, size);
    GraphProtoInfo info;

    auto on_simple = [&](uint32_t field_number, int wire_type, uint64_t value) {};

    auto on_length_delimited = [&](uint32_t field_number, const uint8_t* data, size_t size)
    {
        switch (field_number)
        {
            case 1:
                try
                {
                    info.nodes.push_back(parse_NodeProto(data, size));
                }
                catch (const unsupported_error& e)
                {
                    throw;
                }
                break;
            case 5:
                info.initializers.push_back(parse_TensorInfo(data, size));
                break;
            case 11:
                info.inputs.push_back(parse_ValueInfoProto(data, size));
                break;
            case 12:
                info.outputs.push_back(parse_ValueInfoProto(data, size));
                break;
            case 13:
                info.value_infos.push_back(parse_ValueInfoProto(data, size));
                break;
            default:
                break;
        }
    };

    parse_message(reader, on_simple, on_length_delimited);
    return info;
}




GraphProtoInfo parse_ModelProto(const uint8_t* data, size_t size)
{
    ProtoReader reader(data, size);
    std::optional<GraphProtoInfo> graph_info;

    auto on_simple = [&](uint32_t field_number, int wire_type, uint64_t value) {};

    auto on_length_delimited = [&](uint32_t field_number, const uint8_t* data, size_t size)
    {
        //std::cout << "ModelProto: field=" << field_number << " size=" << size << std::endl;
        if (field_number == 7) //graph
        {
            graph_info = parse_GraphProto(data, size);
        }
    };

    parse_message(reader, on_simple, on_length_delimited);

    if (!graph_info.has_value())
    {
        throw parse_error("ModelProto does not contain a graph");
    }

    return std::move(*graph_info);
}



} //namespeca proto
} //namespace graph
