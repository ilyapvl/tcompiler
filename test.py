import torch
import torch.nn as nn
import numpy as np

class ModelWithInputBias(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, bias1, bias2):
        x = x.view(1, 192, 16)
        x = x + bias1
        x = torch.relu(x)
        x = x + bias2
        return x

model = ModelWithInputBias()
model.eval()


dummy_input = torch.randn(1, 3, 32, 32)
dummy_bias1 = torch.randn(1, 192, 16) * 0.1
dummy_bias2 = torch.randn(1, 192, 16) * 0.1

torch.onnx.export(
    model,
    (dummy_input, dummy_bias1, dummy_bias2),
    "test.onnx",
    input_names=["X", "bias1", "bias2"],
    output_names=["Y"],
    opset_version=14
)

print("ONNX model with explicit bias inputs saved to 'reshape_add_relu_model.onnx'")


bias1_np = dummy_bias1.numpy().flatten()
bias2_np = dummy_bias2.numpy().flatten()
bias1_np.astype(np.float32).tofile("bias1.bin")
bias2_np.astype(np.float32).tofile("bias2.bin")
print("Bias values saved to bias1.bin and bias2.bin")
