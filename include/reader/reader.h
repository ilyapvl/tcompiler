#ifndef GRAPH_PROTO_READER_H
#define GRAPH_PROTO_READER_H

#include <cstdint>
#include <vector>
#include <string>
#include <utility>
#include "graph/exceptions.h"
#include "graph/types.h"


namespace graph
{
namespace proto
{

/**
 * @brief Читатель protobuf-сообщений, работающий с сырыми данными.
 *
 * Предоставляет методы для чтения protobuf-полей
 */
class ProtoReader
{
public:
    /**
     * @brief                   Конструктор из указателя на данные и размера
     * @param data              Указатель на начало буфера
     * @param size              Размер буфера в байтах
     */
    ProtoReader(const uint8_t* data, size_t size);

    /**
     * @brief Конструктор из вектора байт
     * @param data Вектор данных
     */
    explicit ProtoReader(const std::vector<uint8_t>& data);

    /**
     * @brief Прочитать ключ поля
     * @return Пара {номер поля, wire type}
     */
    std::pair<uint32_t, int> read_key();

    /**
     * @brief               Прочитать varint
     * @return              Значение varint
     * @throw               parse_error Если varint слишком длинный.
     */
    uint64_t read_varint();

    /**
     * @brief               Прочитать length-delimited поле
     * @return              Пара {указатель на начало, размер}.
     * @throw               parse_error Если неверная длина поля
     */
    std::pair<const uint8_t*, size_t> read_length_delimited();

    /**
     * @brief Прочитать 32-битное целое (wire type 5)
     * @return 32-битное целое
     */
    uint32_t read_fixed32();

    /**
     * @brief Прочитать 64-битное целое (wire type 1)
     * @return 64-битное целое
     */
    uint64_t read_fixed64();

    /**
     * @brief                   Пропустить поле заданного wire type
     * @param wire_type         Тип поля от 0 до 5
     * @throw                   parse_error Если wire_type не поддерживается или неверная длина поля
     */
    void skip_field(int wire_type);

    /**
     * @brief Прочитать строку из length-delimited поля
     * @return Строка, считанная как length-delimited
     */
    std::string read_string();

    /**
     * @brief Пропустить группу wire type 3/4
     */
    void skip_group();

    /**
     * @brief Получить позицию конца буфера
     * @return Указатель на конец данных
     */
    size_t end_pos() const { return end_; }

    /**
     * @brief EOF?
     * @return true, если текущая позиция за концом буфера
     */
    bool eof() const { return pos_ >= end_; }

    /**
     * @brief Текущая позиция чтения
     * @return Текущая позиция чтения
     */
    size_t position() const { return pos_; }

private:
    const uint8_t* data_;   ///< Указатель на начало
    size_t pos_;            ///< Текущая позиция
    size_t end_;            ///< Позиция конца

    /**
     * @brief                   Проверить, что в буфере осталось достаточно байт
     * @param needed            Необходимое количество байт
     * @throw                   parse_error Если доступно меньше, чем needed
     */
    void check_bound(size_t needed) const;
};

/**
 * @brief Обработчик простых полей
 *
 * @param field_number          Номер поля
 * @param wire_type             Тип поля из [0, 1, 5]
 * @param value                 Прочитанное значение
 */
using SimpleFieldHandler = std::function<void(uint32_t field_number, int wire_type, uint64_t value)>;

/**
 * @brief Обработчик для length-delimited
 *
 * @param field_number          Номер поля
 * @param data                  Указатель на начало блока данных
 * @param size                  Размер блока
 */
using LengthDelimitedHandler = std::function<void(uint32_t field_number, const uint8_t* data, size_t size)>;

/**
 * @brief                       Разобрать protobuf-сообщение
 * @param reader                Читатель, позиционированный на начало
 * @param on_simple             Обработчик простых полей
 * @param on_length_delimited   Обработчик length-delimited 
 * @throw                       parse_error при ошибках формата
 */
void parse_message(ProtoReader& reader,
                   const SimpleFieldHandler& on_simple,
                   const LengthDelimitedHandler& on_length_delimited);

/**
 * @brief                       Парсинг TensorProto
 * @param data                  Указатель на данные
 * @param size                  Размер данных
 * @return                      TensorInfo
 * @throw                       parse_error при ошибках формата
 */
TensorInfo parse_TensorInfo(const uint8_t* data, size_t size);

/**
 * @brief                       Парсинг ValueInfoProto
 * @param data                  Указатель на данные
 * @param size                  Размер данных
 * @return                      TensorInfo
 * @throw                       parse_error при ошибках формата
 */
TensorInfo parse_ValueInfoProto(const uint8_t* data, size_t size);

/**
 * @brief                       Парсинг AttributeProto
 * @param data                  Указатель на данные сообщения
 * @param size                  Размер данных
 * @return                      Пара {имя атрибута, значение}
 * @throw                       parse_error при ошибках формата
 * @throw                       unsupported_error для неподдерживаемого типа атрибута
 */
std::pair<std::string, Attribute> parse_AttributeProto(const uint8_t* data, size_t size);

/**
 * @brief Структура для хранения информации об узле
 */
struct NodeProtoInfo
{
    std::string name;                                           ///< Имя узла
    std::string op_type;                                        ///< Тип операции
    std::vector<std::string> inputs;                            ///< Входнын тензороы
    std::vector<std::string> outputs;                           ///< Выходные тензоры
    std::vector<std::pair<std::string, Attribute>> attributes;  ///< Атрибуты
};

/**
 * @brief                       Парсинг NodeProto
 * @param data                  Указатель на данные
 * @param size                  Размер данных
 * @return                      NodeProtoInfo
 * @throw                       parse_error при ошибках формата
 */
NodeProtoInfo parse_NodeProto(const uint8_t* data, size_t size);

/**
 * @brief Структура для хранения информации о графе
 */
struct GraphProtoInfo
{
    std::vector<NodeProtoInfo> nodes;       ///< Узлы графа
    std::vector<TensorInfo> initializers;   ///< Константные тензоры
    std::vector<TensorInfo> inputs;         ///< Входные тензоры
    std::vector<TensorInfo> outputs;        ///< Выходные тензоры
    std::vector<TensorInfo> value_infos;    ///< Промежуточные тензоры
};

/**
 * @brief                       Парсинг GraphProto
 * @param data                  Указатель на данные
 * @param size                  Размер данных
 * @return                      GraphProtoInfo
 * @throw                       parse_error при ошибках формата
 */
GraphProtoInfo parse_GraphProto(const uint8_t* data, size_t size);

/**
 * @brief                       Парсинг ModelProto
 * @param data                  Указатель на данные
 * @param size                  Размер данных
 * @return                      GraphProtoInfo
 * @throw                       parse_error при ошибках формата
 */
GraphProtoInfo parse_ModelProto(const uint8_t* data, size_t size);

} //namespace proto
} //namespace graph

#endif
