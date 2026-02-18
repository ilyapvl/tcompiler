# TCompiler

Проект реализует фронтенд тензорного компилятора: загрузка модели из формата ONNX, построение вычислительного графа и визуализация

## Возможности

- Чтение ONNX-моделей (формат `.onnx`)
- Построение графа
- Поддержка операций: Add, Mul, Conv, Relu, MatMul, Gemm (а также любых других с ограничениями при визуализации)
- Визуализация графа в формате GraphViz (.dot)

## Требования

- Компилятор с поддержкой C++20
- CMake 3.15 или выше.
- GraphViz (опционально, для конвертации .dot в .svg)

## Сборка

```
git clone https://github.com/ilyapvl/tcompiler.git
cd tcompiler
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

## Запуск

```
./main
```

Сгенерированный ```.dot``` файл находится в ```tcompiler/output/graph.dot```

## Документация

Doxygen документация проекта находится в ```tcompiler/html/index.html```
