#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>

template<typename T, int N>
struct StridedMemRef
{
    T *basePtr;
    T *data;
    int64_t offset;
    int64_t sizes[N];
    int64_t strides[N];
};


extern "C"
{
    void _mlir_ciface_main_graph(
        StridedMemRef<float, 3> *out,
        StridedMemRef<float, 4> *X,
        StridedMemRef<float, 3> *bias1,
        StridedMemRef<float, 3> *bias2
    );
}

void fillTensor4D(float *data, int64_t N, int64_t C, int64_t H, int64_t W, float val)
{
    for (int64_t i = 0; i < N * C * H * W; ++i) data[i] = val;
}

bool compareTensors(const float *a, const float *b, int64_t totalElements, float tol = 1e-5f)
{
    for (int64_t i = 0; i < totalElements; ++i)
    {
        if (std::fabs(a[i] - b[i]) > tol)
        {
            printf("Mismatch at index %lld: expected %f, got %f\n", i, b[i], a[i]);
            return false;
        }
    }
    return true;
}

float* loadBinaryF32(const char* filename, int64_t expectedCount)
{
    FILE* f = fopen(filename, "rb");
    if (!f) { printf("Error opening %s\n", filename); exit(1); }

    float* data = (float*)malloc(expectedCount * sizeof(float));
    size_t read = fread(data, sizeof(float), expectedCount, f);
    if (read != expectedCount) { printf("Error reading %s\n", filename); exit(1); }

    fclose(f);
    return data;
}

int main()
{
    const int64_t N = 1, C = 3, H = 32, W = 32;
    const int64_t inputTotal = N * C * H * W;
    const int64_t D1 = 192, D2 = 16;
    const int64_t outputTotal = N * D1 * D2;

    // input X
    float *input_data = (float*)malloc(inputTotal * sizeof(float));
    fillTensor4D(input_data, N, C, H, W, 1.0f);
    StridedMemRef<float, 4> X = {
        input_data, input_data, 0,
        {N, C, H, W},
        {C*H*W, H*W, W, 1}
    };

    // biases from binary
    float *bias1_data = loadBinaryF32("../bias1.bin", outputTotal);
    float *bias2_data = loadBinaryF32("../bias2.bin", outputTotal);

    StridedMemRef<float, 3> B1 = {
        bias1_data, bias1_data, 0,
        {N, D1, D2},
        {D1*D2, D2, 1}
    };
    StridedMemRef<float, 3> B2 = {
        bias2_data, bias2_data, 0,
        {N, D1, D2},
        {D1*D2, D2, 1}
    };

    StridedMemRef<float, 3> Y;
    _mlir_ciface_main_graph(&Y, &X, &B1, &B2);
    float *model_output = Y.data;

    // computation for checking
    float *reference_out = (float*)malloc(outputTotal * sizeof(float));
    memcpy(reference_out, input_data, inputTotal * sizeof(float));
    for (int64_t i = 0; i < outputTotal; ++i)
    {
        float v = reference_out[i] + bias1_data[i];
        v = (v > 0.0f) ? v : 0.0f;
        v = v + bias2_data[i];
        reference_out[i] = v;
    }


    // checking
    bool ok = compareTensors(model_output, reference_out, outputTotal);
    printf("%s\n", ok ? "SUCCESS" : "FAILURE");

    free(input_data);
    free(bias1_data);
    free(bias2_data);
    free(Y.basePtr);
    free(reference_out);
    return ok ? 0 : 1;
}
