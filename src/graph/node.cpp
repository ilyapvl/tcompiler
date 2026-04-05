#include "graph/node.hpp"
#include <sstream>
#include <stdexcept>
#include <algorithm>

namespace tc
{

    OpType opTypeFromString(const std::string& s)
    {
        static const std::unordered_map<std::string, OpType> table = {
            {"Add",         OpType::Add},
            {"Mul",         OpType::Mul},
            {"MatMul",      OpType::MatMul},
            {"Gemm",        OpType::Gemm},
            {"Conv",        OpType::Conv},
            {"Relu",        OpType::Relu},
            {"Shape",       OpType::Shape},
            {"Reshape",     OpType::Reshape},
            {"Concat",     OpType::Concat},
        };

        
        auto it = table.find(s);
        return (it != table.end()) ? it->second : OpType::Other;
    }

    std::string opTypeToString(OpType op)
    {
        switch (op)
        {
            case OpType::Add:       return "Add";
            case OpType::Mul:       return "Mul";
            case OpType::MatMul:    return "MatMul";
            case OpType::Gemm:      return "Gemm";
            case OpType::Conv:      return "Conv";
            case OpType::Relu:      return "Relu";
            case OpType::Shape:     return "Shape";
            case OpType::Reshape:   return "Reshape";
            case OpType::Concat:    return "Concat";
            case OpType::Other:     return "Other";
            default:                return "Unknown";
        }
    }

    Node::Node(
        std::string name,
        OpType op,
        std::string op_str,
        std::vector<std::string> inputs,
        std::vector<std::string> outputs,
        AttributeMap attributes)

        : name_(std::move(name)),
        op_(op),
        op_str_(std::move(op_str)),
        inputs_(std::move(inputs)),
        outputs_(std::move(outputs)),
        attrs_(std::move(attributes))
    {}

    bool Node::hasAttribute(const std::string& key) const
    {
        return attrs_.contains(key);
    }

    const Attribute& Node::getAttribute(const std::string& key) const
    {
        auto it = attrs_.find(key);
        if (it == attrs_.end())
            throw std::out_of_range("Attribute not found: " + key);
        return it->second;
    }

    void Node::addAttribute(Attribute attr)
    {
        auto key = attr.getName();
        attrs_.emplace(std::move(key), std::move(attr));
    }

    std::string Node::toString() const
    {
        std::ostringstream oss;
        oss << op_str_ << " [" << name_ << "]\n";

        oss << "  inputs : ";
        for (const auto& inp : inputs_) oss << inp << " ";
        oss << "\n";

        oss << "  outputs: ";
        for (const auto& out : outputs_) oss << out << " ";
        oss << "\n";

        if (!attrs_.empty())
        {
            oss << "  attrs  : ";
            for (const auto& [k, a] : attrs_)
                oss << a.toString() << "  ";
            oss << "\n";
        }
        return oss.str();
    }

} // namespace tc
