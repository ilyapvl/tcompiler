#include "frontend/onnx_loader.hpp"
#include "graph/attribute.hpp"

#include "onnx.pb.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace tc
{

    static Attribute convertAttribute(const onnx::AttributeProto& ap);

    static TensorShape shapeFromTypeProto(const onnx::TypeProto& tp)
    {
        TensorShape shape;
        if (!tp.has_tensor_type()) return shape;

        const auto& ts = tp.tensor_type();
        if (!ts.has_shape()) return shape;

        for (const auto& dim : ts.shape().dim())
        {
            if (dim.has_dim_value()) shape.dims.push_back(dim.dim_value());
            else shape.dims.push_back(-1);
        }

        return shape;
    }

    static TensorShape shapeFromTensorProto(const onnx::TensorProto& tp)
    {
        TensorShape shape;

        for (const auto& d : tp.dims()) shape.dims.push_back(d);

        return shape;
    }

    static std::vector<uint8_t> rawDataFromTensorProto(const onnx::TensorProto& tp)
    {
        std::vector<uint8_t> data;
        if (!tp.raw_data().empty())
        {
            const auto& rd = tp.raw_data();
            data.assign(
                reinterpret_cast<const uint8_t*>(rd.data()),
                reinterpret_cast<const uint8_t*>(rd.data()) + rd.size());
        }

        return data;
    }




    static std::shared_ptr<Tensor> tensorFromProto(const onnx::TensorProto& tp)
    {
        auto shape = shapeFromTensorProto(tp);
        auto dtype = dataTypeFromOnnx(tp.data_type());
        auto tensor = std::make_shared<Tensor>(tp.name(), dtype, shape);

        tensor->setRawData(rawDataFromTensorProto(tp));
        return tensor;
    }




    static std::shared_ptr<Graph> graphFromProto(const onnx::GraphProto& gp)
    {
        auto graph = std::make_shared<Graph>(gp.name().empty() ? "subgraph" : gp.name());
        
        std::unordered_set<std::string> initializer_names;
        for (const auto& init : gp.initializer())
            initializer_names.insert(init.name());
        
        for (const auto& vi : gp.input())
        {
            if (initializer_names.contains(vi.name())) continue;
            graph->addInput(vi.name());

            auto shape = shapeFromTypeProto(vi.type());
            auto dtype = DataType::UNDEFINED;

            if (vi.type().has_tensor_type())
                dtype = dataTypeFromOnnx(vi.type().tensor_type().elem_type());

            graph->addTensor(std::make_shared<Tensor>(vi.name(), dtype, shape));
        }
        

        for (const auto& init : gp.initializer())
        {
            auto tensor = tensorFromProto(init);
            graph->addTensor(tensor);
        }
        
        for (const auto& vi : gp.value_info())
        {
            if (graph->findTensor(vi.name())) continue;
            auto shape = shapeFromTypeProto(vi.type());
            auto dtype = DataType::UNDEFINED;
            if (vi.type().has_tensor_type())
                dtype = dataTypeFromOnnx(vi.type().tensor_type().elem_type());
            graph->addTensor(std::make_shared<Tensor>(vi.name(), dtype, shape));
        }
        

        for (const auto& vi : gp.output())
        {
            graph->addOutput(vi.name());
            if (!graph->findTensor(vi.name()))
            {
                auto shape = shapeFromTypeProto(vi.type());
                auto dtype = DataType::UNDEFINED;

                if (vi.type().has_tensor_type())
                    dtype = dataTypeFromOnnx(vi.type().tensor_type().elem_type());

                graph->addTensor(std::make_shared<Tensor>(vi.name(), dtype, shape));
            }
        }
        


        for (int i = 0; i < gp.node_size(); ++i)
        {
            const auto& np = gp.node(i);
            std::string node_name = np.name();

            if (node_name.empty())
                node_name = np.op_type() + "_" + std::to_string(i);
            
            std::vector<std::string> inputs(np.input().begin(), np.input().end());
            std::vector<std::string> outputs(np.output().begin(), np.output().end());
            
            Node::AttributeMap attrs;
            for (const auto& ap : np.attribute())
            {
                auto attr = convertAttribute(ap);
                attrs.emplace(attr.getName(), std::move(attr));
            }
            
            auto op = opTypeFromString(np.op_type());
            auto node = std::make_shared<Node>(
                node_name, op, np.op_type(),
                std::move(inputs), std::move(outputs),
                std::move(attrs)
            );
            graph->addNode(std::move(node));
        }
        
        return graph;
    }


    static Attribute convertAttribute(const onnx::AttributeProto& ap)
    {
        const auto& name = ap.name();

        switch (ap.type())
        {
                
            case onnx::AttributeProto::FLOAT:
                return {name, AttributeType::FLOAT, ap.f()};

            case onnx::AttributeProto::INT:
                return {name, AttributeType::INT, ap.i()};

            case onnx::AttributeProto::STRING:
                return {name, AttributeType::STRING, ap.s()};

            case onnx::AttributeProto::TENSOR:
            {
                auto tensor = tensorFromProto(ap.t());
                return {name, AttributeType::TENSOR, std::move(tensor)};
            }

            case onnx::AttributeProto::GRAPH:
            {
                auto subgraph = graphFromProto(ap.g());
                return {name, AttributeType::GRAPH, std::move(subgraph)};
            }

            case onnx::AttributeProto::SPARSE_TENSOR:
            {
                const auto& st = ap.sparse_tensor();
                std::vector<uint8_t> data;

                // TODO: values
                return {name, AttributeType::SPARSE_TENSOR, std::move(data)};
            }

            case onnx::AttributeProto::FLOATS:
            {
                std::vector<float> vals(ap.floats().begin(), ap.floats().end());
                return {name, AttributeType::FLOATS, std::move(vals)};
            }

            case onnx::AttributeProto::INTS:
            {
                std::vector<int64_t> vals(ap.ints().begin(), ap.ints().end());
                return {name, AttributeType::INTS, std::move(vals)};
            }

            case onnx::AttributeProto::STRINGS:
            {
                std::vector<std::string> vals(ap.strings().begin(), ap.strings().end());
                return {name, AttributeType::STRINGS, std::move(vals)};
            }

            case onnx::AttributeProto::TENSORS:
            {
                std::vector<std::shared_ptr<Tensor>> tensors;
                tensors.reserve(ap.tensors_size());
                for (const auto& tp : ap.tensors())
                    tensors.push_back(tensorFromProto(tp));
                return {name, AttributeType::TENSORS, std::move(tensors)};
            }

            case onnx::AttributeProto::GRAPHS:
            {
                std::vector<std::shared_ptr<Graph>> graphs;
                graphs.reserve(ap.graphs_size());
                for (const auto& gp : ap.graphs())
                    graphs.push_back(graphFromProto(gp));
                return {name, AttributeType::GRAPHS, std::move(graphs)};
            }

            case onnx::AttributeProto::SPARSE_TENSORS:
            {
                std::vector<std::vector<uint8_t>> sparse_tensors;
                sparse_tensors.reserve(ap.sparse_tensors_size());
                for (const auto& st : ap.sparse_tensors())
                {
                    //TODO - values
                    sparse_tensors.emplace_back();
                }
                return {name, AttributeType::SPARSE_TENSORS, std::move(sparse_tensors)};
            }

            default:

                return {name, AttributeType::STRING, std::string("<unsupported_attribute>")};
        }
    }


    std::shared_ptr<Graph> OnnxLoader::load(const std::filesystem::path& path) const
    {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs)
            throw std::runtime_error("Cannot open file: " + path.string());

        GOOGLE_PROTOBUF_VERIFY_VERSION;

        onnx::ModelProto model;
        if (!model.ParseFromIstream(&ifs))
            throw std::runtime_error("Failed to parse ONNX model: " + path.string());

        const auto& gp = model.graph();
        auto graph = std::make_shared<Graph>(
            gp.name().empty() ? "onnx_graph" : gp.name());

        std::unordered_set<std::string> initializer_names;
        for (const auto& init : gp.initializer())
        {
            initializer_names.insert(init.name());
            auto shape  = shapeFromTensorProto(init);
            auto dtype  = dataTypeFromOnnx(init.data_type());
            auto tensor = std::make_shared<Tensor>(init.name(), dtype, shape);
            tensor->setRawData(rawDataFromTensorProto(init));
            graph->addTensor(std::move(tensor));
        }

        for (const auto& vi : gp.input())
        {
            if (initializer_names.contains(vi.name())) continue;
            graph->addInput(vi.name());
            auto shape = shapeFromTypeProto(vi.type());
            auto dtype = DataType::UNDEFINED;
            if (vi.type().has_tensor_type())
                dtype = dataTypeFromOnnx(vi.type().tensor_type().elem_type());
            if (!graph->findTensor(vi.name()))
                graph->addTensor(std::make_shared<Tensor>(vi.name(), dtype, shape));
        }

        auto addValueInfo = [&](const onnx::ValueInfoProto& vi)
        {
            if (graph->findTensor(vi.name())) return;
            auto shape = shapeFromTypeProto(vi.type());
            auto dtype = DataType::UNDEFINED;
            if (vi.type().has_tensor_type())
                dtype = dataTypeFromOnnx(vi.type().tensor_type().elem_type());
            graph->addTensor(std::make_shared<Tensor>(vi.name(), dtype, shape));
        };

        for (const auto& vi : gp.value_info()) addValueInfo(vi);
        for (const auto& vi : gp.output())     addValueInfo(vi);

        for (const auto& vi : gp.output())
            graph->addOutput(vi.name());

        for (int i = 0; i < gp.node_size(); ++i)
        {
            const auto& np = gp.node(i);

            std::string node_name = np.name();
            if (node_name.empty())
                node_name = np.op_type() + "_" + std::to_string(i);

            std::vector<std::string> inputs(np.input().begin(),  np.input().end());
            std::vector<std::string> outputs(np.output().begin(), np.output().end());

            Node::AttributeMap attrs;
            for (const auto& ap : np.attribute())
            {
                auto attr = convertAttribute(ap);
                attrs.emplace(attr.getName(), std::move(attr));
            }

            auto op   = opTypeFromString(np.op_type());
            auto node = std::make_shared<Node>(
                node_name, op, np.op_type(),
                std::move(inputs), std::move(outputs),
                std::move(attrs));
            graph->addNode(std::move(node));
        }

        return graph;
    }

    OnnxLoader::ModelInfo OnnxLoader::readModelInfo(const std::filesystem::path& path)
    {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs)
            throw std::runtime_error("Cannot open file: " + path.string());

        onnx::ModelProto model;
        if (!model.ParseFromIstream(&ifs))
            throw std::runtime_error("Failed to parse ONNX model");

        ModelInfo info;
        info.ir_version       = model.ir_version();
        info.producer_name    = model.producer_name();
        info.producer_version = model.producer_version();
        info.domain           = model.domain();
        info.model_version    = model.model_version();
        info.doc_string       = model.doc_string();
        info.graph_name       = model.graph().name();
        return info;
    }

} // namespace tc
