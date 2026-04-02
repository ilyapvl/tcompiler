#include "graph/tensor.hpp"
#include <sstream>
#include <stdexcept>

namespace tc
{

    std::string dataTypeToString(DataType dt)
    {
        switch (dt)
        {
            case DataType::UNDEFINED:  return "undefined";
            case DataType::FLOAT:      return "float32";
            case DataType::UINT8:      return "uint8";
            case DataType::INT8:       return "int8";
            case DataType::UINT16:     return "uint16";
            case DataType::INT16:      return "int16";
            case DataType::INT32:      return "int32";
            case DataType::INT64:      return "int64";
            case DataType::STRING_T:   return "string";
            case DataType::BOOL:       return "bool";
            case DataType::FLOAT16:    return "float16";
            case DataType::DOUBLE:     return "float64";
            case DataType::UINT32:     return "uint32";
            case DataType::UINT64:     return "uint64";
            case DataType::COMPLEX64:  return "complex64";
            case DataType::COMPLEX128: return "complex128";
            default:                   return "unknown";
        }
    }

    DataType dataTypeFromOnnx(int onnx_type)
    {
        if (onnx_type >= 0 && onnx_type <= 15)
            return static_cast<DataType>(onnx_type);
        return DataType::UNDEFINED;
    }

    bool TensorShape::isDynamic() const
    {
        for (const auto& d : dims)
            if (d < 0) return true;
        return false;
    }

    std::string TensorShape::toString() const
    {
        if (dims.empty()) return "scalar";
        std::ostringstream oss;
        oss << '[';
        for (size_t i = 0; i < dims.size(); ++i)
        {
            if (i) oss << 'x';
            if (dims[i] < 0) oss << '?';
            else              oss << dims[i];
        }
        oss << ']';
        return oss.str();
    }

    Tensor::Tensor(std::string name, DataType dtype, TensorShape shape)
        : name_(std::move(name)),
        dtype_(dtype),
        shape_(std::move(shape)) {}

    std::string Tensor::toString() const
    {
        return name_ + " : " + dataTypeToString(dtype_) + shape_.toString();
    }

    size_t Tensor::numElements() const
    {
        if (shape_.dims.empty()) return 1;

        size_t n = 1;
        for (auto d : shape_.dims) {
            if (d > 0) n *= static_cast<size_t>(d);
        }

        return n;
    }

    bool Tensor::hasValidData() const
    {
        if (!hasData()) return false;

        size_t expected = numElements() * dataTypeSize(dtype_);
        return raw_data_.size() == expected;
    }

    size_t Tensor::dataTypeSize(DataType dt)
    {
        switch (dt)
        {
            case DataType::FLOAT:      return sizeof(float);
            case DataType::DOUBLE:     return sizeof(double);
            case DataType::INT8:       return sizeof(int8_t);
            case DataType::INT16:      return sizeof(int16_t);
            case DataType::INT32:      return sizeof(int32_t);
            case DataType::INT64:      return sizeof(int64_t);
            case DataType::UINT8:      return sizeof(uint8_t);
            case DataType::UINT16:     return sizeof(uint16_t);
            case DataType::UINT32:     return sizeof(uint32_t);
            case DataType::UINT64:     return sizeof(uint64_t);
            case DataType::BOOL:       return sizeof(bool);
            default:                   return 0;
        }
    }

} // namespace tc
