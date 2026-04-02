#ifndef ATTRIBUTE_HPP
#define ATTRIBUTE_HPP

#include <string>
#include <variant>
#include <vector>
#include <stdexcept>



namespace tc
{

    class Tensor;
    class Graph;

    enum class AttributeType
    {
        UNDEFINED      = 0,
        FLOAT          = 1,
        INT            = 2,
        STRING         = 3,
        TENSOR         = 4,
        GRAPH          = 5,
        FLOATS         = 6,
        INTS           = 7,
        STRINGS        = 8,
        TENSORS        = 9,
        GRAPHS         = 10,
        SPARSE_TENSOR  = 11,
        SPARSE_TENSORS = 12,
    };

    using AttributeValue = std::variant<
        float,                                     // FLOAT
        int64_t,                                   // INT
        std::string,                               // STRING
        std::shared_ptr<Tensor>,                   // TENSOR
        std::shared_ptr<Graph>,                    // GRAPH
        std::vector<uint8_t>,                      // SPARSE_TENSOR
        std::vector<float>,                        // FLOATS
        std::vector<int64_t>,                      // INTS
        std::vector<std::string>,                  // STRINGS
        std::vector<std::shared_ptr<Tensor>>,      // TENSORS
        std::vector<std::shared_ptr<Graph>>,       // GRAPHS
        std::vector<std::vector<uint8_t>>          // SPARSE_TENSORS
    >;

    class Attribute
    {
    public:
        Attribute() = default;
        Attribute(std::string name, AttributeType type, AttributeValue value)
            : name_(std::move(name)), type_(type), value_(std::move(value)) {}

        [[nodiscard]] const std::string&    getName()  const { return name_; }
        [[nodiscard]] AttributeType         getType()  const { return type_; }
        [[nodiscard]] const AttributeValue& getValue() const { return value_; }

        [[nodiscard]] float                                         asFloat()           const;
        [[nodiscard]] int64_t                                       asInt()             const;
        [[nodiscard]] const std::string&                            asString()          const;
        [[nodiscard]] const std::vector<float>&                     asFloats()          const;
        [[nodiscard]] const std::vector<int64_t>&                   asInts()            const;
        [[nodiscard]] const std::vector<std::string>&               asStrings()         const;
        [[nodiscard]] const std::vector<std::shared_ptr<Tensor>>&   asTensors()         const;
        [[nodiscard]] const std::vector<std::shared_ptr<Graph>>&    asGraphs()          const;
        [[nodiscard]] const std::vector<std::vector<uint8_t>>&      asSparseTensors()   const;
        [[nodiscard]] const std::shared_ptr<Tensor>&                asTensor()          const;
        [[nodiscard]] const std::shared_ptr<Graph>&                 asGraph()           const;
        [[nodiscard]] const std::vector<uint8_t>&                   asSparseTensor()    const;

        [[nodiscard]] std::string toString() const;

    private:
        std::string    name_;
        AttributeType  type_{AttributeType::UNDEFINED};
        AttributeValue value_;
    };

} // namespace ts




#endif // ATTRIBUTE_HPP
