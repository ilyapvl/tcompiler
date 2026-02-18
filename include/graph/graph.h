#ifndef GRAPH_GRAPH_H
#define GRAPH_GRAPH_H

#include <graph/types.h>
#include <graph/node.h>
#include <string>
#include <vector>
#include <map>
#include <ostream>

namespace graph
{

/**
 * @brief Вычислительный граф нейронной сети
 */
class Graph
{
public:


    /**
     * @brief           Добавить новый узел
     * @param name      Имя узла (должно быть уникальным)
     * @param op_type   Тип операции
     * @return          Указатель на созданный узел
     * @throw           validation_error если узел с таким именем уже существует
     */
    Node* add_node(std::string name, std::string op_type);

    /// @return Указатель на узел по имени, или nullptr, если не найден.
    Node* get_node(const std::string& name);

    /// @return Ссылка на карту узлов.
    const std::map<std::string, Node>& nodes() const { return nodes_; }


    /**
     * @brief           Добавить новый тензор
     * @param name      Имя тензора (должно быть уникальным)
     * @return          Указатель на созданный тензор
     * @throw           validation_error если тензор с таким именем уже существует
     */
    TensorInfo* add_tensor(const std::string& name);

    /// @return Указатель на тензор по имени, или nullptr, если не найден
    TensorInfo* get_tensor(const std::string& name);

    /// @return Ссылка на карту тензоров
    const std::map<std::string, TensorInfo>& tensors() const { return tensors_; }


    /// @brief Добавить имя входного тензора
    void add_input(const std::string& tensor_name);

    /// @brief Добавить имя выходного тензора
    void add_output(const std::string& tensor_name);


    /// @return Список имён входных тензоров
    const std::vector<std::string>& inputs() const { return inputs_; }

    /// @return Список имён выходных тензоров
    const std::vector<std::string>& outputs() const { return outputs_; }


    /**
     * @brief           Построить связи на основе текущих узлов
     * @throw           validation_error если обнаружен тензор с неоднозначной привязкой  
     */
    void build_connections();


    /**
     * @brief           Загрузить граф из файла ONNX
     * @param filename  Путь к .onnx файлу
     * @return          true в случае успеха
     * @throw           io_error если файл не может быть открыт
     * @throw           parse_error если ошибка парсинга
     * @throw           unsupported_error при встрече неподдерживаемого атрибута
     * @throw           validation_error при нарушении целостности графа
     */
    bool load_from_onnx(const std::string& filename);


    /**
     * @brief           Вывести описание графа в формате .dot
     * @param os        Поток вывода
     */
    void dump_dot(std::ostream& os) const;




private:
    std::map<std::string, Node>                         nodes_;         ///< Операции 
    std::map<std::string, TensorInfo>                   tensors_;       ///< Тензоры
    std::vector<std::string>                            inputs_;        ///< Входные тензоры
    std::vector<std::string>                            outputs_;       ///< Выходные тензоры
    std::map<std::string, std::string>                  producer_;      ///< Производители
    std::map<std::string, std::vector<std::string>>     consumers_;     ///< Потребители
};

} // namespace graph

#endif // GRAPH_GRAPH_H
