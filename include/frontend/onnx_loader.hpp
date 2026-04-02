#ifndef ONNX_LOADER_HPP
#define ONNX_LOADER_HPP

#include "graph/graph.hpp"
#include <filesystem>
#include <memory>
#include <string>

namespace tc
{

    class OnnxLoader
    {
    public:
        OnnxLoader() = default;

        [[nodiscard]] std::shared_ptr<Graph>
        load(const std::filesystem::path& path) const;

        struct ModelInfo
        {
            int64_t     ir_version{};
            std::string producer_name;
            std::string producer_version;
            std::string domain;
            int64_t     model_version{};
            std::string doc_string;
            std::string graph_name;
        };

        [[nodiscard]] static ModelInfo
        readModelInfo(const std::filesystem::path& path);
    };

} // namespace tc

#endif // ONNX_LOADER_HPP
