#ifndef GRAPH_NODE_H
#define GRAPH_NODE_H

#include "graph/types.h"
#include <string>
#include <vector>
#include <map>

namespace graph
{

/**
 * @brief Узел вычислительного графа
 */
class Node
{
public:

    Node() = default;

    /**
     * @brief           Конструктор с именем и типом операции
     * @param name      Уникальное имя узла
     * @param op_type   Тип операции
     */
    Node(std::string name, std::string op_type)
        : name_(std::move(name)), op_type_(std::move(op_type)) {}

    /// @return Имя узла
    const std::string& name() const { return name_; }
    /// @param n Новое имя узла
    void set_name(const std::string& n) { name_ = n; }

    /// @return Тип операции
    const std::string& op_type() const { return op_type_; }
    /// @param ot Новый тип операции
    void set_op_type(const std::string& ot) { op_type_ = ot; }

    /// @return Ссылка на карту атрибутов
    std::map<std::string, Attribute>& attributes() { return attributes_; }
    /// @return Константная ссылка на карту атрибутов
    const std::map<std::string, Attribute>& attributes() const { return attributes_; }

    /**
     * @brief           Установить атрибут.
     * @param key       Имя атрибута.
     * @param value     Значение атрибута.
     */
    void set_attribute(const std::string& key, const Attribute& value) { attributes_[key] = value; }

    /// @return Список имён входных тензоров
    std::vector<std::string>& inputs() { return inputs_; }
    const std::vector<std::string>& inputs() const { return inputs_; }

    /// @brief Добавить входной тензор
    void add_input(const std::string& tensor_name) { inputs_.push_back(tensor_name); }

    /// @return Список имён выходных тензоров
    std::vector<std::string>& outputs() { return outputs_; }
    const std::vector<std::string>& outputs() const { return outputs_; }

    /// @brief Добавить выходной тензор
    void add_output(const std::string& tensor_name) { outputs_.push_back(tensor_name); }

private:
    std::string                         name_;          ///< Имя узла
    std::string                         op_type_;       ///< Тип операции
    std::map<std::string, Attribute>    attributes_;    ///< Атрибуты операции
    std::vector<std::string>            inputs_;        ///< Имена входных тензоров
    std::vector<std::string>            outputs_;       ///< Имена выходных тензоров
};

} // namespace graph

#endif // GRAPH_NODE_H
