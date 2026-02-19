import onnx
from onnx import helper, TensorProto, checker
import numpy as np
import os

def generate():

    X1 = helper.make_tensor_value_info('X1', TensorProto.FLOAT, [1, 3, 32, 32])
    Out1 = helper.make_tensor_value_info('Out1', TensorProto.FLOAT, [1, 16, 32, 32])

    Y1 = helper.make_tensor_value_info('Y1', TensorProto.FLOAT, [1, 16, 32, 32])
    Z1 = helper.make_tensor_value_info('Z1', TensorProto.FLOAT, [1, 16, 32, 32])
    A1 = helper.make_tensor_value_info('A1', TensorProto.FLOAT, [1, 16, 32, 32])

    C1_data = np.full((1, 16, 32, 32), 2.0, dtype=np.float32).flatten().tolist()
    C1 = helper.make_tensor('C1', TensorProto.FLOAT, [1, 16, 32, 32], C1_data)

    C2_data = np.full((1, 16, 32, 32), 0.5, dtype=np.float32).flatten().tolist()
    C2 = helper.make_tensor('C2', TensorProto.FLOAT, [1, 16, 32, 32], C2_data)

    W_conv_data = np.ones((16, 3, 3, 3), dtype=np.float32).flatten().tolist()
    W_conv = helper.make_tensor('W_conv', TensorProto.FLOAT, [16, 3, 3, 3], W_conv_data)
    B_conv_data = np.zeros(16, dtype=np.float32).tolist()
    B_conv = helper.make_tensor('B_conv', TensorProto.FLOAT, [16], B_conv_data)

    conv = helper.make_node('Conv', inputs=['X1', 'W_conv', 'B_conv'], outputs=['Y1'],
                            kernel_shape=[3,3], pads=[1,1,1,1], strides=[1,1], name='conv')
    relu = helper.make_node('Relu', inputs=['Y1'], outputs=['Z1'], name='relu')
    add = helper.make_node('Add', inputs=['Z1', 'C1'], outputs=['A1'], name='add')
    mul = helper.make_node('Mul', inputs=['A1', 'C2'], outputs=['Out1'], name='mul')

    X2 = helper.make_tensor_value_info('X2', TensorProto.FLOAT, [1, 256])
    Out2 = helper.make_tensor_value_info('Out2', TensorProto.FLOAT, [1, 64])

    Y2 = helper.make_tensor_value_info('Y2', TensorProto.FLOAT, [1, 128])

    W_matmul_data = np.full((256, 128), 0.1, dtype=np.float32).flatten().tolist()
    W_matmul = helper.make_tensor('W_matmul', TensorProto.FLOAT, [256, 128], W_matmul_data)

    W_gemm_data = np.full((128, 64), 0.2, dtype=np.float32).flatten().tolist()
    W_gemm = helper.make_tensor('W_gemm', TensorProto.FLOAT, [128, 64], W_gemm_data)
    B_gemm_data = np.full(64, 0.05, dtype=np.float32).tolist()
    B_gemm = helper.make_tensor('B_gemm', TensorProto.FLOAT, [64], B_gemm_data)

    matmul = helper.make_node('MatMul', inputs=['X2', 'W_matmul'], outputs=['Y2'], name='matmul')
    gemm = helper.make_node('Gemm', inputs=['Y2', 'W_gemm', 'B_gemm'], outputs=['Out2'],
                            alpha=1.0, beta=1.0, transB=0, name='gemm')

  
    graph = helper.make_graph(
        [conv, relu, add, mul, matmul, gemm],
        'six_ops_fixed',
        [X1, X2],
        [Out1, Out2],
        initializer=[C1, C2, W_conv, B_conv, W_matmul, W_gemm, B_gemm],
        value_info=[Y1, Z1, A1, Y2]
    )

    model = helper.make_model(graph, producer_name='test', opset_imports=[helper.make_opsetid("", 16)])
    checker.check_model(model)
    onnx.save(model, 'models/test.onnx')

if __name__ == '__main__':
    generate()
