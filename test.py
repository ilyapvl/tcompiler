import onnx
model = onnx.load('models/add_model.onnx')
print(model.graph)
