#ifndef NODE_HPP
#define NODE_HPP


#include "graph/attribute.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <optional>

namespace tc
{

    enum class OpType
    {
        Unknown,
        Add,
        Mul,
        MatMul,
        Gemm,
        Conv,
        Relu,
        Other,
    };

    [[nodiscard]] OpType      opTypeFromString(const std::string& s);
    [[nodiscard]] std::string opTypeToString(OpType op);

    class Node
    {
    public:
        using AttributeMap = std::unordered_map<std::string, Attribute>;

        Node() = default;
        Node(std::string name,
            OpType op,
            std::string op_str,
            std::vector<std::string> inputs,
            std::vector<std::string> outputs,
            AttributeMap attributes = {});

        [[nodiscard]] const std::string& getName()    const { return name_; }
        [[nodiscard]] OpType             getOpType()  const { return op_; }
        [[nodiscard]] const std::string& getOpStr()   const { return op_str_; }

        [[nodiscard]] const std::vector<std::string>& getInputs()  const { return inputs_; }
        [[nodiscard]] const std::vector<std::string>& getOutputs() const { return outputs_; }

        [[nodiscard]] const AttributeMap& getAttributes() const  { return attrs_; }
        [[nodiscard]] bool hasAttribute(const std::string& key) const;
        [[nodiscard]] const Attribute& getAttribute(const std::string& key) const;
        void addAttribute(Attribute attr);

        [[nodiscard]] std::string toString() const;

    private:
        std::string              name_;
        OpType                   op_ { OpType::Unknown };
        std::string              op_str_;
        std::vector<std::string> inputs_;
        std::vector<std::string> outputs_;
        AttributeMap             attrs_;
    };

} // namespace tc

#endif // NODE_HPP
