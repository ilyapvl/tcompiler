#ifndef GRAPH_TYPES_H
#define GRAPH_TYPES_H

#include <cstdint>
#include <string>
#include <vector>
#include <variant>
#include <map>

namespace graph
{

/**
 * @brief Типы данных тензора
 */
enum class DataType
{
    UNDEFINED,
    FLOAT,
    INT64,
    INT32,
};

/**
 * @brief Информация о тензоре
 */
struct TensorInfo
{
    std::string             name;                               ///< Имя
    std::vector<int64_t>    dims;                               ///< Размерности
    DataType                data_type = DataType::UNDEFINED;    ///< Тип данных
    bool                    is_constant = false;                ///< true если тензор const
    std::vector<uint8_t>    raw_data;                           ///< Данные
};

/**
 * @brief Представление атрибута
 */
using Attribute = std::variant<
    int64_t,
    float,
    std::string,
    std::vector<int64_t>,
    std::vector<float>,
    std::vector<std::string>,
    TensorInfo
>;


inline bool operator==(const TensorInfo& lhs, const TensorInfo& rhs)
{
    return lhs.name == rhs.name &&
           lhs.dims == rhs.dims &&
           lhs.data_type == rhs.data_type &&
           lhs.is_constant == rhs.is_constant;
}

inline bool operator==(const Attribute& lhs, const Attribute& rhs)
{
    if (lhs.index() != rhs.index()) return false;
    return std::visit([](const auto& a, const auto& b) -> bool { return a == b; }, lhs, rhs);
}

} // namespace graph

#endif // GRAPH_TYPES_H
