#ifndef TENSOR_HPP
#define TENSOR_HPP


#include <cstdint>
#include <string>
#include <vector>
#include <optional>
#include <span>

namespace tc
{

    enum class DataType : int
    {
        UNDEFINED  = 0,
        FLOAT      = 1,
        UINT8      = 2,
        INT8       = 3,
        UINT16     = 4,
        INT16      = 5,
        INT32      = 6,
        INT64      = 7,
        STRING_T   = 8,
        BOOL       = 9,
        FLOAT16    = 10,
        DOUBLE     = 11,
        UINT32     = 12,
        UINT64     = 13,
        COMPLEX64  = 14,
        COMPLEX128 = 15,
    };

    [[nodiscard]] std::string dataTypeToString(DataType dt);
    [[nodiscard]] DataType    dataTypeFromOnnx(int onnx_type);

    struct TensorShape
    {
        std::vector<int64_t> dims;

        [[nodiscard]] bool isDynamic() const;
        [[nodiscard]] size_t rank() const { return dims.size(); }
        [[nodiscard]] std::string toString() const;
    };

    class Tensor
    {
    public:
        Tensor() = default;
        Tensor(std::string name, DataType dtype, TensorShape shape);

        [[nodiscard]] const std::string&  getName()  const { return name_; }
        [[nodiscard]] DataType            getDtype()  const { return dtype_; }
        [[nodiscard]] const TensorShape&  getShape()  const { return shape_; }

        [[nodiscard]] bool hasData() const { return !raw_data_.empty(); }
        [[nodiscard]] const std::vector<uint8_t>& getRawData()  const { return raw_data_; }
        void setRawData(std::vector<uint8_t> data) { raw_data_ = std::move(data); }

        void setShape(TensorShape shape) { shape_ = std::move(shape); }
        void setDtype(DataType dtype)    { dtype_ = dtype; }

        [[nodiscard]] std::string toString() const;



        template<typename T>
        [[nodiscard]] std::span<const T> getDataAs() const
        {
            static_assert(std::is_arithmetic_v<T>, "T must be arithmetic");
            if (!hasData()) return {};

            size_t elem_count = raw_data_.size() / sizeof(T);

            if (elem_count * sizeof(T) != raw_data_.size())
            {
                throw std::runtime_error("Data size not aligned with type");
            }

            return std::span<const T>(
                reinterpret_cast<const T*>(raw_data_.data()),
                elem_count
            );
        }

        template<typename T>
        [[nodiscard]] bool isDataTypeCompatible() const
        {
            if (!hasData()) return false;
            size_t elem_size = sizeof(T);
            size_t elem_count = raw_data_.size() / elem_size;
            return elem_count * elem_size == raw_data_.size();
        }


        [[nodiscard]] size_t numElements() const;
        [[nodiscard]] bool hasValidData() const;
        static size_t dataTypeSize(DataType dt);

    private:
        std::string          name_;
        DataType             dtype_{DataType::UNDEFINED};
        TensorShape          shape_;
        std::vector<uint8_t> raw_data_;
    };

} // namespace tc




#endif // TENSOR_HPP
