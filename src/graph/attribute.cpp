#include "graph/attribute.hpp"
#include "graph/tensor.hpp"
#include "graph/graph.hpp"
#include <sstream>
#include <stdexcept>

namespace tc
{

    float Attribute::asFloat() const
    {
        if (const auto* v = std::get_if<float>(&value_)) return *v;
        throw std::bad_variant_access{};
    }

    int64_t Attribute::asInt() const
    {
        if (const auto* v = std::get_if<int64_t>(&value_)) return *v;
        throw std::bad_variant_access{};
    }

    const std::string& Attribute::asString() const
    {
        if (const auto* v = std::get_if<std::string>(&value_)) return *v;
        throw std::bad_variant_access{};
    }

    const std::shared_ptr<Tensor>& Attribute::asTensor() const
    {
        if (const auto* v = std::get_if<std::shared_ptr<Tensor>>(&value_)) return *v;
        throw std::bad_variant_access{};
    }

    const std::shared_ptr<Graph>& Attribute::asGraph() const
    {
        if (const auto* v = std::get_if<std::shared_ptr<Graph>>(&value_)) return *v;
        throw std::bad_variant_access{};
    }

    const std::vector<uint8_t>& Attribute::asSparseTensor() const
    {
        if (const auto* v = std::get_if<std::vector<uint8_t>>(&value_)) return *v;
        throw std::bad_variant_access{};
    }

    const std::vector<float>& Attribute::asFloats() const
    {
        if (const auto* v = std::get_if<std::vector<float>>(&value_)) return *v;
        throw std::bad_variant_access{};
    }

    const std::vector<int64_t>& Attribute::asInts() const
    {
        if (const auto* v = std::get_if<std::vector<int64_t>>(&value_)) return *v;
        throw std::bad_variant_access{};
    }

    const std::vector<std::string>& Attribute::asStrings() const
    {
        if (const auto* v = std::get_if<std::vector<std::string>>(&value_)) return *v;
        throw std::bad_variant_access{};
    }

    const std::vector<std::shared_ptr<Tensor>>& Attribute::asTensors() const
    {
        if (const auto* v = std::get_if<std::vector<std::shared_ptr<Tensor>>>(&value_)) return *v;
        throw std::bad_variant_access{};
    }

    const std::vector<std::shared_ptr<Graph>>& Attribute::asGraphs() const
    {
        if (const auto* v = std::get_if<std::vector<std::shared_ptr<Graph>>>(&value_)) return *v;
        throw std::bad_variant_access{};
    }

    const std::vector<std::vector<uint8_t>>& Attribute::asSparseTensors() const
    {
        if (const auto* v = std::get_if<std::vector<std::vector<uint8_t>>>(&value_)) return *v;
        throw std::bad_variant_access{};
    }

    std::string Attribute::toString() const
    {
        std::ostringstream oss;
        oss << name_ << "=";

        std::visit(
            [&oss](const auto& v)
            {
                using T = std::decay_t<decltype(v)>;


                if constexpr (std::is_same_v<T, float>)
                {
                    oss << v;
                }

                else if constexpr (std::is_same_v<T, int64_t>)
                {
                    oss << v;
                }

                else if constexpr (std::is_same_v<T, std::string>)
                {
                    oss << '"' << v << '"';
                }

                else if constexpr (std::is_same_v<T, std::shared_ptr<Tensor>>)
                {
                    if (v) oss << "Tensor(" << v->toString() << ")";
                    else oss << "nullptr";
                }

                else if constexpr (std::is_same_v<T, std::shared_ptr<Graph>>)
                {
                    if (v) oss << "Graph(" << v->getName() << ", nodes=" << v->getNodes().size() << ")";
                    else oss << "nullptr";
                }

                else if constexpr (std::is_same_v<T, std::vector<uint8_t>>)
                {
                    oss << "SparseTensor(data_size=" << v.size() << ")";
                }

                else if constexpr (std::is_same_v<T, std::vector<float>>)
                {
                    oss << '[';
                    for (size_t i = 0; i < v.size(); ++i)
                    {
                        if (i) oss << ',';
                        oss << v[i];
                    }
                    oss << ']';
                }

                else if constexpr (std::is_same_v<T, std::vector<int64_t>>)
                {
                    oss << '[';
                    for (size_t i = 0; i < v.size(); ++i)
                    {
                        if (i) oss << ',';
                        oss << v[i];
                    }
                    oss << ']';
                }

                else if constexpr (std::is_same_v<T, std::vector<std::string>>)
                {
                    oss << '[';
                    for (size_t i = 0; i < v.size(); ++i)
                    {
                        if (i) oss << ',';
                        oss << '"' << v[i] << '"';
                    }
                    oss << ']';
                }

                else if constexpr (std::is_same_v<T, std::vector<std::shared_ptr<Tensor>>>)
                {
                    oss << '[' << v.size() << " tensors]";
                }

                else if constexpr (std::is_same_v<T, std::vector<std::shared_ptr<Graph>>>)
                {
                    oss << '[' << v.size() << " graphs]";
                }

                else if constexpr (std::is_same_v<T, std::vector<std::vector<uint8_t>>>)
                {
                    oss << '[' << v.size() << " sparse tensors]";
                }
            },


            value_);



            
        return oss.str();
    }

} // namesoace tc
