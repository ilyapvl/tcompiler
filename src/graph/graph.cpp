#include "graph/graph.hpp"
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <queue>

namespace tc
{

    Graph::Graph(std::string name) : name_(std::move(name)) {}

    void Graph::addNode(std::shared_ptr<Node> node)
    {
        if (!node) throw std::invalid_argument("null node");
        node_map_[node->getName()] = node;
        nodes_.push_back(std::move(node));
    }

    const std::vector<std::shared_ptr<Node>>& Graph::getNodes() const
    {
        return nodes_;
    }

    std::optional<std::shared_ptr<Node>> Graph::findNode(const std::string& name) const
    {
        auto it = node_map_.find(name);
        if (it == node_map_.end()) return std::nullopt;
        return it->second;
    }

    void Graph::addTensor(std::shared_ptr<Tensor> tensor)
    {
        if (!tensor) throw std::invalid_argument("null tensor");
        tensor_map_[tensor->getName()] = std::move(tensor);
    }

    std::optional<std::shared_ptr<Tensor>> Graph::findTensor(const std::string& name) const
    {
        auto it = tensor_map_.find(name);
        if (it == tensor_map_.end()) return std::nullopt;


        return it->second;
    }

    const std::unordered_map<std::string, std::shared_ptr<Tensor>>& Graph::getTensors() const
    {
        return tensor_map_;
    }

    void Graph::addInput(const std::string& name)  { inputs_.push_back(name); }
    void Graph::addOutput(const std::string& name) { outputs_.push_back(name); }

    std::vector<std::shared_ptr<Node>> Graph::topologicalSort() const
    {
        std::unordered_map<std::string, std::vector<std::shared_ptr<Node>>> consumers;
        std::unordered_map<std::shared_ptr<Node>, int> in_degree;

        std::unordered_set<std::string> available;
        for (const auto& inp : inputs_) available.insert(inp);
        for (const auto& [name, t] : tensor_map_)
            if (t->hasData()) available.insert(name);

        for (const auto& node : nodes_)
        {
            int deg = 0;
            for (const auto& inp : node->getInputs())
            {
                if (!available.contains(inp))
                {
                    consumers[inp].push_back(node);
                    ++deg;
                }
            }
            in_degree[node] = deg;
        }



        

        std::queue<std::shared_ptr<Node>> q;


        for (const auto& node : nodes_)
            if (in_degree[node] == 0)
                q.push(node);

        std::vector<std::shared_ptr<Node>> sorted;
        sorted.reserve(nodes_.size());


        while (!q.empty())
        {
            auto cur = q.front(); q.pop();
            sorted.push_back(cur);

            for (const auto& out : cur->getOutputs())
            {
                auto it = consumers.find(out);
                if (it == consumers.end()) continue;


                for (auto& consumer : it->second)
                {
                    if (--in_degree[consumer] == 0)
                        q.push(consumer);
                }
            }
        }

        return sorted;
    }

    std::string Graph::summary() const
    {
        std::unordered_map<std::string, int> op_counts;
        
        for (const auto& n : nodes_)
            ++op_counts[n->getOpStr()];

        std::ostringstream oss;
        oss << "Graph: " << name_ << "\n";
        oss << "  nodes  : " << nodes_.size()   << "\n";
        oss << "  tensors: " << tensor_map_.size() << "\n";
        oss << "  inputs : ";
        for (const auto& i : inputs_)  oss << i << " ";
        oss << "\n  outputs: ";
        for (const auto& o : outputs_) oss << o << " ";
        oss << "\n  op breakdown:\n";
        for (const auto& [op, cnt] : op_counts)
            oss << "    " << op << ": " << cnt << "\n";
        return oss.str();
    }

} // namespace tc
