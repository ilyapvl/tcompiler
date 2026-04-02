#ifndef GRAPH_HPP
#define GRAPH_HPP

#include "graph/node.hpp"
#include "graph/tensor.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <optional>
#include <stdexcept>

namespace tc
{

    class Graph
    {
    public:
        explicit Graph(std::string name = "graph");

        [[nodiscard]] const std::string& getName() const { return name_; }
        void setName(std::string n) { name_ = std::move(n); }

        void addNode(std::shared_ptr<Node> node);
        [[nodiscard]] const std::vector<std::shared_ptr<Node>>& getNodes() const;
        [[nodiscard]] std::optional<std::shared_ptr<Node>> findNode(const std::string& name) const;

        void addTensor(std::shared_ptr<Tensor> tensor);
        [[nodiscard]] std::optional<std::shared_ptr<Tensor>> findTensor(const std::string& name) const;
        [[nodiscard]] const std::unordered_map<std::string, std::shared_ptr<Tensor>>& getTensors() const;

        void addInput (const std::string& name);
        void addOutput(const std::string& name);
        [[nodiscard]] const std::vector<std::string>& getInputs()  const { return inputs_; }
        [[nodiscard]] const std::vector<std::string>& getOutputs() const { return outputs_; }

        [[nodiscard]] std::vector<std::shared_ptr<Node>> topologicalSort() const;

        [[nodiscard]] std::string summary() const;

    private:
        std::string name_;
        std::vector<std::shared_ptr<Node>>                          nodes_;
        std::unordered_map<std::string, std::shared_ptr<Node>>      node_map_;
        std::unordered_map<std::string, std::shared_ptr<Tensor>>    tensor_map_;
        std::vector<std::string>                                    inputs_;
        std::vector<std::string>                                    outputs_;
    };

} // namespace ts




#endif //GRAPH_HPP
