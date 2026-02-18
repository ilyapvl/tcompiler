#include <graph/graph.h>
#include <graph/exceptions.h>
#include <reader/reader.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>
#include <fstream>
#include <map>





namespace graph
{

Node* Graph::add_node(std::string name, std::string op_type)
{
    auto [it, inserted] = nodes_.try_emplace(name, std::move(name), std::move(op_type));
    if (!inserted)
    {
        throw validation_error("Node with name '" + name + "' already exists");
    }
    return &it->second;
}

Node* Graph::get_node(const std::string& name)
{
    auto it = nodes_.find(name);
    if (it == nodes_.end()) return nullptr;
    return &it->second;
}

TensorInfo* Graph::add_tensor(const std::string& name)
{
    auto [it, inserted] = tensors_.try_emplace(name);
    if (!inserted)
    {
        throw validation_error("Tensor with name '" + name + "' already exists");
    }
    it->second.name = name;
    return &it->second;
}

TensorInfo* Graph::get_tensor(const std::string& name)
{
    auto it = tensors_.find(name);
    if (it == tensors_.end()) return nullptr;
    return &it->second;
}

void Graph::add_input(const std::string& tensor_name)
{
    inputs_.push_back(tensor_name);
}

void Graph::add_output(const std::string& tensor_name)
{
    outputs_.push_back(tensor_name);
}

void Graph::build_connections()
{
    producer_.clear();
    consumers_.clear();

    for (const auto& [node_name, node] : nodes_)
    {
        for (const auto& out_tensor : node.outputs())
        {
            auto [it, inserted] = producer_.try_emplace(out_tensor, node_name);
            if (!inserted)
            {
                throw validation_error("Tensor '" + out_tensor + "' produced by multiple nodes");
            }
        }
        for (const auto& in_tensor : node.inputs())
        {
            consumers_[in_tensor].push_back(node_name);
        }
    }
}


bool Graph::load_from_onnx(const std::string& filename)
{



    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) 
    {
        throw io_error("Cannot open file: " + filename);
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> buffer(size);

    if (!file.read(reinterpret_cast<char*>(buffer.data()), size))
    {
        throw io_error("Failed to read file: " + filename);
    }



    proto::GraphProtoInfo graph_info = proto::parse_ModelProto(buffer.data(), buffer.size());

    
    nodes_.clear();
    tensors_.clear();
    inputs_.clear();
    outputs_.clear();





    
    for (const auto& tensor : graph_info.initializers)
    {
        tensors_[tensor.name] = tensor;
    }

    
    for (const auto& input : graph_info.inputs)
    {
        if (tensors_.find(input.name) == tensors_.end())
        {
            TensorInfo ti;
            ti.name = input.name;
            ti.data_type = input.data_type;
            ti.dims = input.dims;
            ti.is_constant = false;
            tensors_[input.name] = ti;
        }
        add_input(input.name);
    }


    for (const auto& output : graph_info.outputs)
    {
        if (tensors_.find(output.name) == tensors_.end())
        {
            TensorInfo ti;
            ti.name = output.name;
            ti.data_type = output.data_type;
            ti.dims = output.dims;
            ti.is_constant = false;
            tensors_[output.name] = ti;
        }
        add_output(output.name);
    }


    for (const auto& vi : graph_info.value_infos)
    {
        if (tensors_.find(vi.name) == tensors_.end())
        {
            TensorInfo ti;
            ti.name = vi.name;
            ti.data_type = vi.data_type;
            ti.dims = vi.dims;
            ti.is_constant = false;
            tensors_[vi.name] = ti;
        }
    }


    for (const auto& node_proto : graph_info.nodes)
    {
        
        static int node_counter = 0;
        std::string node_name = node_proto.name;
        if (node_name.empty())
        {
            node_name = "node_" + std::to_string(node_counter++);
        }
        Node* node = add_node(node_name, node_proto.op_type);

        for (const auto& input_name : node_proto.inputs)
        {
            node->add_input(input_name);

            if (tensors_.find(input_name) == tensors_.end())
            {
                TensorInfo ti;
                ti.name = input_name;
                ti.is_constant = false;
                tensors_[input_name] = ti;
            }
        }
        for (const auto& output_name : node_proto.outputs)
        {
            node->add_output(output_name);
            if (tensors_.find(output_name) == tensors_.end())
            {
                TensorInfo ti;
                ti.name = output_name;
                ti.is_constant = false;
                tensors_[output_name] = ti;
            }
        }
        for (const auto& [key, val] : node_proto.attributes)
        {
            node->set_attribute(key, val);
        }
    }

  
    build_connections();


    for (const auto& out_name : outputs_)
    {
        if (producer_.find(out_name) == producer_.end())
        {
            throw validation_error("Output tensor '" + out_name + "' has no producer");
        }
    }

    return true;
}





//NOTE -    как я понял, ONNX может по разному создавать константы, из-за чего 
//          константные тензоры при визуализации могут отображаться и как
//          овал (initializer), и как прямоугольник (операция)
void Graph::dump_dot(std::ostream& os) const
{
    os << "digraph G {\n";
    os << "  rankdir=TB;\n";
    os << "  node [shape=record, fontname=\"Sans\"];\n";

    static const std::map<std::string, std::vector<std::string>> input_names = {
        {"Conv",            {"X", "W", "B"}},
        {"Gemm",            {"A", "B", "C"}},
        {"Add",             {"A", "B"}},
        {"Mul",             {"A", "B"}},
        {"Relu",            {"X"}},
        {"MatMul",          {"A", "B"}},
        {"Squeeze",         {"data", "axes"}},
        {"Unsqueeze",       {"data", "axes"}},
        {"Shape",           {"data"}},
        {"Reshape",         {"data", "shape"}},
    };

    auto escape_name = [](const std::string& s) -> std::string
    {
        std::string res = s;
        std::replace(res.begin(), res.end(), '-', '_');
        std::replace(res.begin(), res.end(), '.', '_');
        std::replace(res.begin(), res.end(), '/', '_');
        std::replace(res.begin(), res.end(), '\\', '_');
        std::replace(res.begin(), res.end(), ':', '_');
        return res;
    };

    auto attr_to_string = [](const Attribute& attr) -> std::string
    {
        struct Visitor
        {
            std::string operator()(int64_t v) { return std::to_string(v); }
            std::string operator()(float v) { return std::to_string(v); }
            std::string operator()(const std::string& v) { return v; }
            std::string operator()(const std::vector<int64_t>& v)
            {
                std::string res = "[";
                for (size_t i = 0; i < v.size(); ++i)
                {
                    if (i > 0) res += ",";
                    res += std::to_string(v[i]);
                }
                res += "]";
                return res;
            }
            std::string operator()(const std::vector<float>& v)
            {
                std::string res = "[";
                for (size_t i = 0; i < v.size(); ++i)
                {
                    if (i > 0) res += ",";
                    res += std::to_string(v[i]);
                }
                res += "]";
                return res;
            }
            std::string operator()(const std::vector<std::string>& v)
            {
                std::string res = "[";
                for (size_t i = 0; i < v.size(); ++i)
                {
                    if (i > 0) res += ",";
                    res += v[i];
                }
                res += "]";
                return res;
            }
            std::string operator()(const TensorInfo& ti)
            {
                std::string res = "tensor(" + ti.name;
                if (!ti.dims.empty())
                {
                    res += " [";
                    for (size_t i = 0; i < ti.dims.size(); ++i)
                    {
                        if (i > 0) res += ",";
                        if (ti.dims[i] == -1) res += "?";
                        else res += std::to_string(ti.dims[i]);
                    }
                    res += "]";
                }
                res += ")";
                return res;
            }
        };
        return std::visit(Visitor{}, attr);
    };

    std::map<std::string, std::string> tensor_id_map = {};
    int tensor_counter = 0;
    int node_counter = 0;

    auto get_tensor_id = [&](const std::string& name) -> std::string
    {
        auto it = tensor_id_map.find(name);
        if (it != tensor_id_map.end()) return it->second;
        std::string id;
        if (name.empty())
        {
            id = "tensor_" + std::to_string(tensor_counter++);
        }
        else
        {
            id = escape_name(name);
            if (id.empty()) id = "tensor_" + std::to_string(tensor_counter++);
        }
        tensor_id_map[name] = id;
        return id;
    };

    auto get_node_id = [&](const std::string& name) -> std::string
    {
        if (name.empty())
        {
            return "node_" + std::to_string(node_counter++);
        }
        std::string id = escape_name(name);
        if (id.empty()) id = "node_" + std::to_string(node_counter++);
        return id;
    };

    for (const auto& [node_name, node] : nodes_)
    {
        std::string node_id = get_node_id(node_name);
        std::string label = "{" + node.op_type();

        if (!node.inputs().empty())
        {
            label += "|inputs:";
            for (size_t i = 0; i < node.inputs().size(); ++i)
            {
                const auto& tensor_name = node.inputs()[i];
                std::string role;

                auto names_it = input_names.find(node.op_type());
                if (names_it != input_names.end() && i < names_it->second.size())
                {
                    role = names_it->second[i];
                }
                else
                {
                    role = tensor_name.empty() ? "?" : tensor_name;
                }

                label += "\\n  " + role + " : " + (tensor_name.empty() ? "?" : tensor_name);


                auto tensor_it = tensors_.find(tensor_name);
                if (tensor_it != tensors_.end() && !tensor_it->second.dims.empty())
                {
                    label += " [";
                    for (size_t d = 0; d < tensor_it->second.dims.size(); ++d)
                    {
                        if (d > 0) label += ",";
                        if (tensor_it->second.dims[d] == -1) label += "?";
                        else label += std::to_string(tensor_it->second.dims[d]);
                    }
                    label += "]";
                }
            }
        }

        if (!node.attributes().empty())
        {
            label += "|attributes:";
            bool first_attr = true;
            for (const auto& [key, val] : node.attributes())
            {
                if (!first_attr) label += "\\n";
                first_attr = false;
                label += "  " + key + "=" + attr_to_string(val);
            }
        }

        label += "}";
        os << "  \"" << node_id << "\" [label=\"" << label << "\"];\n";
    }

    for (const auto& [tensor_name, tensor] : tensors_)
    {
        if (producer_.find(tensor_name) != producer_.end()) continue;

        std::string tensor_id = get_tensor_id(tensor_name);
        if (tensor.name.empty()) continue;
        std::string label = tensor_name;
        if (!tensor.dims.empty())
        {
            label += " [";
            for (size_t i = 0; i < tensor.dims.size(); ++i)
            {
                if (i > 0) label += ",";
                if (tensor.dims[i] == -1) label += "?";
                else label += std::to_string(tensor.dims[i]);
            }
            label += "]";
        }
        os << "  \"" << tensor_id << "\" [label=\"" << label << "\", shape=ellipse";
        if (tensor.is_constant)
        {
            os << ", style=filled, fillcolor=lightgray";
        }
        os << "];\n";
    }

    for (const auto& out_name : outputs_) 
    {
        std::string out_node_id = get_tensor_id(out_name) + "_out";
        std::string label = "output: " + out_name;

        auto tensor_it = tensors_.find(out_name);
        if (tensor_it != tensors_.end() && !tensor_it->second.dims.empty())
        {
            label += " [";
            for (size_t i = 0; i < tensor_it->second.dims.size(); ++i)
            {
                if (i > 0) label += ",";
                if (tensor_it->second.dims[i] == -1) label += "?";
                else label += std::to_string(tensor_it->second.dims[i]);
            }
            label += "]";
        }

        os << "  \"" << out_node_id << "\" [label=\"" << label << "\", shape=ellipse];\n";
    }

    for (const auto& [tensor_name, producer_node] : producer_)
    {
        std::string tensor_id = get_tensor_id(tensor_name);
        std::string producer_id = get_node_id(producer_node);

        auto consumers_it = consumers_.find(tensor_name);
        if (consumers_it != consumers_.end())
        {
            for (const auto& consumer_node : consumers_it->second)
            {
                std::string consumer_id = get_node_id(consumer_node);
                os << "  \"" << producer_id << "\" -> \"" << consumer_id << "\";\n";
            }
        }
    }

    for (const auto& [tensor_name, tensor] : tensors_)
    {
        if (producer_.find(tensor_name) != producer_.end()) continue;
        auto consumers_it = consumers_.find(tensor_name);
        if (consumers_it == consumers_.end()) continue;

        std::string tensor_id = get_tensor_id(tensor_name);
        for (const auto& consumer_node : consumers_it->second)
        {
            std::string consumer_id = get_node_id(consumer_node);
            os << "  \"" << tensor_id << "\" -> \"" << consumer_id << "\";\n";
        }
    }

    for (const auto& out_name : outputs_)
    {
        std::string out_node_id = get_tensor_id(out_name) + "_out";
        auto prod_it = producer_.find(out_name);
        if (prod_it != producer_.end())
        {
            std::string prod_id = get_node_id(prod_it->second);
            os << "  \"" << prod_id << "\" -> \"" << out_node_id << "\";\n";
        }
        
        else
        {
            std::string tensor_id = get_tensor_id(out_name);
            os << "  \"" << tensor_id << "\" -> \"" << out_node_id << "\";\n";
        }
    }

    os << "}\n";
}

}
