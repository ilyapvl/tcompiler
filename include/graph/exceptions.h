#ifndef GRAPH_EXCEPTIONS_H
#define GRAPH_EXCEPTIONS_H

#include <stdexcept>
#include <string>

namespace graph
{

/**
 * @brief Базовый класс исключений
 */
class graph_exception : public std::runtime_error
{
public:
    explicit graph_exception(const std::string& msg) : std::runtime_error(msg) {}
};

/**
 * @brief Ошибки ввода/вывода
 */
class io_error : public graph_exception
{
public:
    explicit io_error(const std::string& msg) : graph_exception("I/O error: " + msg) {}
};

/**
 * @brief Ошибки парсинга ONNX
 */
class parse_error : public graph_exception
{
public:
    explicit parse_error(const std::string& msg) : graph_exception("Parse error: " + msg) {}
};

/**
 * @brief Ошибки целостности графа
 */
class validation_error : public graph_exception
{
public:
    explicit validation_error(const std::string& msg) : graph_exception("Validation error: " + msg) {}
};

/**
 * @brief Неподдерживаемая операция/атрибут
 */
class unsupported_error : public graph_exception
{
public:
    explicit unsupported_error(const std::string& msg) : graph_exception("Unsupported: " + msg) {}
};

} // namespace graph

#endif // GRAPH_EXCEPTIONS_H
