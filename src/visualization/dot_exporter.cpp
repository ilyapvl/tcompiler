#include "visualization/dot_exporter.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstdlib>
#include <algorithm>

namespace tc
{

    DotExporter::DotExporter(Options opts) : opts_(opts) {}

    std::string DotExporter::nodeColor(OpType op)
    {
        switch (op)
        {
            case OpType::Conv:   return "#AED6F1";
            case OpType::Relu:   return "#A9DFBF";
            case OpType::Add:    return "#F9E79F";
            case OpType::Mul:    return "#F5CBA7";
            case OpType::MatMul: return "#D7BDE2";
            case OpType::Gemm:   return "#C39BD3";
            default:             return "#E8E8E8";
        }
    }

    std::string DotExporter::escapeLabel(const std::string& s)
    {
        std::string out;
        out.reserve(s.size());
        for (char c : s)
        {
            if (c == '"')       out += "\\\"";
            else if (c == '<')  out += "\\<";
            else if (c == '>')  out += "\\>";
            else if (c == '{')  out += "\\{";
            else if (c == '}')  out += "\\}";
            else if (c == '|')  out += "\\|";
            else                out += c;
        }
        return out;
    }

    std::string DotExporter::toDot(const Graph& graph) const
    {
        std::ostringstream dot;

        dot << "digraph \"" << escapeLabel(graph.getName()) << "\" {\n";
        dot << "  graph [rankdir=TB, bgcolor=\"#FAFAFA\", fontname=\"Helvetica\"];\n";
        dot << "  node  [shape=record, style=filled, fontname=\"Helvetica\", fontsize=11];\n";
        dot << "  edge  [fontname=\"Helvetica\", fontsize=9];\n\n";

        const auto& tensors = graph.getTensors();

        dot << "  // Graph inputs\n";
        for (const auto& inp : graph.getInputs())
        {
            std::string label = escapeLabel(inp);

            if (opts_.show_tensor_shapes)
            {
                auto it = tensors.find(inp);
                if (it != tensors.end())
                    label += "\\n" + escapeLabel(it->second->getShape().toString());
            }





            dot << "  \"input_" << escapeLabel(inp) << "\" ["
                << "label=\"{INPUT|" << label << "}\", "
                << "fillcolor=\"#85C1E9\", shape=record];\n";
        }

        dot << "\n  // Graph outputs\n";

        for (const auto& out : graph.getOutputs())
        {
            std::string label = escapeLabel(out);

            if (opts_.show_tensor_shapes)
            {
                auto it = tensors.find(out);
                if (it != tensors.end())
                    label += "\\n" + escapeLabel(it->second->getShape().toString());
            }

            dot << "  \"output_" << escapeLabel(out) << "\" ["
                << "label=\"{OUTPUT|" << label << "}\", "
                << "fillcolor=\"#82E0AA\", shape=record];\n";
        }


        dot << "\n  // Operation nodes\n";


        for (const auto& node : graph.getNodes())
        {
            std::string color = opts_.color_by_optype
                ? nodeColor(node->getOpType())
                : "#E8E8E8";

            std::string header = escapeLabel(node->getOpStr());
            if (!node->getName().empty())
                header += "\\n(" + escapeLabel(node->getName()) + ")";

            std::string attr_str;
            if (opts_.show_attributes && !node->getAttributes().empty())
            {
                for (const auto& [k, a] : node->getAttributes())
                    attr_str += escapeLabel(a.toString()) + "\\l";
            }

            std::string const_str;
            for (const auto& inp : node->getInputs())
            {
                auto it = tensors.find(inp);
                if (it != tensors.end() && it->second->hasData())
                {
                    bool is_graph_input = false;
                    for (const auto& gi : graph.getInputs())
                        if (gi == inp) { is_graph_input = true; break; }
                    if (is_graph_input) continue;

                    const_str += escapeLabel(inp) + " = ";
                    const auto& tensor = it->second;
                    const_str += dataTypeToString(tensor->getDtype()) + tensor->getShape().toString();
                    const_str += "\\l";
                }
            }

            std::string label = "{" + header;
            if (!attr_str.empty()) label += "|" + attr_str;
            if (!const_str.empty()) label += "|const inputs:\\l" + const_str;
            label += "}";

            dot << "  \"" << escapeLabel(node->getName()) << "\" ["
                << "label=\"" << label << "\", "
                << "fillcolor=\"" << color << "\"];\n";
        }

        dot << "\n  // Edges\n";

        std::unordered_map<std::string, std::vector<std::string>> consumers;

        for (const auto& node : graph.getNodes())
            for (const auto& inp : node->getInputs())
                consumers[inp].push_back(node->getName());

        std::unordered_map<std::string, std::string> producers;

        for (const auto& node : graph.getNodes())
            for (const auto& out : node->getOutputs())
                producers[out] = node->getName();

        for (const auto& inp : graph.getInputs())
        {
            auto it = consumers.find(inp);
            if (it == consumers.end()) continue;
            for (const auto& dst : it->second)
            {
                std::string edge_label;
                if (opts_.show_tensor_shapes)
                {
                    auto ti = tensors.find(inp);


                    if (ti != tensors.end())
                        edge_label = " [label=\"" + escapeLabel(ti->second->getShape().toString()) + "\"]";
                }



                dot << "  \"input_" << escapeLabel(inp) << "\" -> \""
                    << escapeLabel(dst) << "\"" << edge_label << ";\n";
            }
        }

        for (const auto& node : graph.getNodes())
        {
            for (const auto& out : node->getOutputs())
            {
                bool is_graph_output = false;
                for (const auto& go : graph.getOutputs())
                    if (go == out) { is_graph_output = true; break; }

                if (is_graph_output)
                {
                    std::string edge_label;
                    if (opts_.show_tensor_shapes)
                    {
                        auto ti = tensors.find(out);

                        if (ti != tensors.end())
                            edge_label = " [label=\"" + escapeLabel(ti->second->getShape().toString()) + "\"]";
                    }



                    dot << "  \"" << escapeLabel(node->getName()) << "\" -> "
                        << "\"output_" << escapeLabel(out) << "\"" << edge_label << ";\n";
                    continue;
                }

                auto it = consumers.find(out);
                if (it == consumers.end()) continue;
                for (const auto& dst : it->second)
                {
                    std::string edge_label;
                    if (opts_.show_tensor_shapes)
                    {
                        auto ti = tensors.find(out);

                        if (ti != tensors.end())
                            edge_label = " [label=\"" + escapeLabel(ti->second->getShape().toString()) + "\"]";
                    }



                    dot << "  \"" << escapeLabel(node->getName()) << "\" -> \""
                        << escapeLabel(dst) << "\"" << edge_label << ";\n";
                }
            }
        }

        dot << "}\n";
        return dot.str();
    }

    void DotExporter::exportToFile(const Graph& graph, const std::filesystem::path& path) const
    {
        std::ofstream ofs(path);
        if (!ofs)
            throw std::runtime_error("Cannot open output file: " + path.string());
        ofs << toDot(graph);
    }

    void DotExporter::exportToPng(const Graph& graph, const std::filesystem::path& png_path) const
    {
        auto dot_path = png_path.parent_path() / (png_path.stem().string() + ".dot");
        exportToFile(graph, dot_path);

        std::string cmd = "dot -Tpng \"" + dot_path.string()
                        + "\" -o \"" + png_path.string() + "\"";
        int ret = std::system(cmd.c_str());
        if (ret != 0)
            throw std::runtime_error(
                "dot command failed (is GraphViz installed?)\nDOT file written to: " + dot_path.string());
    }

}
