#ifndef DOT_EXPORTER_HPP
#define DOT_EXPORTER_HPP



#include "graph/graph.hpp"
#include <filesystem>
#include <string>

namespace tc
{

    class DotExporter
    {
    public:
        struct Options
        {
            bool show_tensor_shapes{true};
            bool show_attributes{true};
            bool color_by_optype{true};
        };

        DotExporter() = default;

        explicit DotExporter(Options opts);

        [[nodiscard]] std::string toDot(const Graph& graph) const;

        void exportToFile(const Graph& graph, const std::filesystem::path& path) const;

        void exportToPng(const Graph& graph, const std::filesystem::path& png_path) const;

    private:
        Options opts_;

        [[nodiscard]] static std::string nodeColor(OpType op);
        [[nodiscard]] static std::string escapeLabel(const std::string& s);
    };

} // namespace tc


#endif // DOT_EXPORTER_HPP
