#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <memory>
#include <cassert>
#include <cmath>
#include <dlfcn.h>
#include <getopt.h>


template<typename T, int N>
struct StridedMemRef
{
    T* basePtr;
    T* data;
    int64_t offset;
    int64_t sizes[N];
    int64_t strides[N];
};


struct MemRefBase
{
    virtual ~MemRefBase() = default;

    virtual void*   getData() = 0;
    virtual int64_t getTotalElements() const = 0;
    virtual void    printShape(std::ostream& os) const = 0;
    virtual void*   getDescPtr() = 0;
    virtual void    freeData() = 0;
    virtual void    loadFromFile(const std::string& filename) = 0;
    virtual void    saveToFile(const std::string& filename) const = 0;
};


template<typename T, int N>
class MemRefHolder : public MemRefBase
{
public:
    explicit MemRefHolder(const std::vector<int64_t>& dims)
    {
        if (dims.size() != N) throw std::runtime_error("Dimension count mismatch");

        for (int i = 0; i < N; ++i) desc_.sizes[i] = dims[i];
        int64_t stride = 1;
        
        for (int i = N - 1; i >= 0; --i)
        {
            desc_.strides[i] = stride;
            stride *= dims[i];
        }

        desc_.offset = 0;
        desc_.basePtr = nullptr;
        desc_.data = nullptr;
    }

    void* getData() override { return desc_.data; }

    int64_t getTotalElements() const override
    {
        int64_t total = 1;
        for (int i = 0; i < N; ++i) total *= desc_.sizes[i];
        return total;
    }

    void printShape(std::ostream& os) const override
    {
        os << "[";
        for (int i = 0; i < N; ++i) {
            os << desc_.sizes[i];
            if (i != N - 1) os << ", ";
        }
        os << "]";
    }

    void* getDescPtr() override { return &desc_; }

    void freeData() override
    {
        if (desc_.basePtr)
        {
            std::free(desc_.basePtr);
            desc_.basePtr = nullptr;
            desc_.data = nullptr;
        }
    }

    void loadFromFile(const std::string& filename) override
    {
        int64_t total = getTotalElements();
        ownedData_.resize(total);
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs) throw std::runtime_error("Cannot open input file: " + filename);

        ifs.read(reinterpret_cast<char*>(ownedData_.data()), total * sizeof(T));
        if (!ifs) throw std::runtime_error("Failed to read input file: " + filename);

        desc_.basePtr = ownedData_.data();
        desc_.data = ownedData_.data();
    }

    void saveToFile(const std::string& filename) const override
    {
        std::ofstream ofs(filename, std::ios::binary);
        if (!ofs) throw std::runtime_error("Cannot open output file: " + filename);

        ofs.write(reinterpret_cast<const char*>(desc_.data), getTotalElements() * sizeof(T));
        if (!ofs) throw std::runtime_error("Failed to write output file: " + filename);
    }

private:
    StridedMemRef<T, N> desc_;
    std::vector<T> ownedData_;
};



std::vector<int64_t> parseShape(const std::string& s)
{
    std::vector<int64_t> dims;
    std::stringstream ss(s);
    std::string item;

    while (std::getline(ss, item, ','))
    {
        dims.push_back(std::stoll(item));
    }

    return dims;
}

std::vector<std::string> split(const std::string& s, char delim)
{
    std::vector<std::string> tokens;
    std::stringstream ss(s);
    std::string token;

    while (std::getline(ss, token, delim))
    {
        tokens.push_back(token);
    }

    return tokens;
}

std::vector<float> readReference(const std::string& filename, int64_t expectedSize)
{
    std::vector<float> data(expectedSize);
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) throw std::runtime_error("Cannot open reference file: " + filename);

    ifs.read(reinterpret_cast<char*>(data.data()), expectedSize * sizeof(float));
    if (!ifs) throw std::runtime_error("Failed to read reference file: " + filename);

    return data;
}

bool compareTensors(const float* a, const float* b, int64_t size, double tol)
{
    for (int64_t i = 0; i < size; ++i)
    {
        if (std::fabs(a[i] - b[i]) > tol)
        {
            std::cerr << "Mismatch at index " << i << ": expected " << b[i] << ", got " << a[i] << "\n";
            return false;
        }
    }

    return true;
}




using FuncPtr = void (*)();

void invokeModel(FuncPtr func,
                std::vector<std::unique_ptr<MemRefBase>>& outputs,
                std::vector<std::unique_ptr<MemRefBase>>& inputs)
{
    const int numOutputs = outputs.size();
    const int numInputs = inputs.size();
    const int totalArgs = numOutputs + numInputs;
    constexpr int MAX_ARGS = 5;
    if (totalArgs > MAX_ARGS) throw std::runtime_error("Too many arguments");

    void* args[MAX_ARGS];
    int idx = 0;
    for (auto& out : outputs) args[idx++] = out->getDescPtr();
    for (auto& in : inputs)   args[idx++] = in->getDescPtr();

    switch (totalArgs)
    {
        case 1:  reinterpret_cast<void (*)(void*)>(func)(args[0]); break;
        case 2:  reinterpret_cast<void (*)(void*,void*)>(func)(args[0], args[1]); break;
        case 3:  reinterpret_cast<void (*)(void*,void*,void*)>(func)(args[0], args[1], args[2]); break;
        case 4:  reinterpret_cast<void (*)(void*,void*,void*,void*)>(func)(args[0], args[1], args[2], args[3]); break;
        case 5:  reinterpret_cast<void (*)(void*,void*,void*,void*,void*)>(func)(args[0], args[1], args[2], args[3], args[4]); break;
        
        default: throw std::runtime_error("Unsupported argument count");
    }
}


std::unique_ptr<MemRefBase> createHolder(const std::vector<int64_t>& dims)
{
    switch (dims.size())
    {
        case 1: return std::make_unique<MemRefHolder<float, 1>>(dims);
        case 2: return std::make_unique<MemRefHolder<float, 2>>(dims);
        case 3: return std::make_unique<MemRefHolder<float, 3>>(dims);
        case 4: return std::make_unique<MemRefHolder<float, 4>>(dims);

        default: throw std::runtime_error("Unsupported rank (max 4)");
    }
}


int main(int argc, char* argv[])
{
    std::string funcName = "_mlir_ciface_main_graph";
    std::vector<std::string> inputFiles, inputShapeStrs, outputFiles, outputShapeStrs, refFiles;
    double tolerance = 1e-4;
    bool verbose = false;

    static struct option long_options[] = {
        {"func",          required_argument, 0, 'f'},
        {"input",         required_argument, 0, 'i'},
        {"input-shape",   required_argument, 0, 's'},
        {"output",        required_argument, 0, 'o'},
        {"output-shape",  required_argument, 0, 'p'},
        {"ref",           required_argument, 0, 'r'},
        {"verbose",       no_argument,       0, 'v'},
        {0, 0, 0, 0}
    };

    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "f:i:s:o:p:r:v", long_options, &option_index)) != -1)
    {
        switch (opt)
        {
            case 'f': funcName          = optarg;               break;
            case 'i': inputFiles        = split(optarg, ',');   break;
            case 's': inputShapeStrs    = split(optarg, ';');   break;
            case 'o': outputFiles       = split(optarg, ',');   break;
            case 'p': outputShapeStrs   = split(optarg, ';');   break;
            case 'r': refFiles          = split(optarg, ',');   break;
            case 'v': verbose           = true;                 break;

            default: return 1;
        }
    }

    if (inputFiles.empty() || inputShapeStrs.size() != inputFiles.size())
    {
        std::cerr << "Input files and shapes must be provided and match in number\n";
        return 1;
    }

    if (!outputShapeStrs.empty() && outputShapeStrs.size() != outputFiles.size() && !outputFiles.empty())
    {
        std::cerr << "Output shapes count must match output files count\n";
        return 1;
    }

    if (!refFiles.empty() && refFiles.size() != outputShapeStrs.size())
    {
        std::cerr << "Number of reference files must match number of outputs\n";
        return 1;
    }

    try
    {
        void* exec = dlopen(nullptr, RTLD_LAZY);
        if (!exec) throw std::runtime_error(std::string("dlopen failed: ") + dlerror());

        dlerror();
        FuncPtr func = reinterpret_cast<FuncPtr>(dlsym(exec, funcName.c_str()));
        const char* err = dlerror();
        if (err) throw std::runtime_error(std::string("dlsym failed: ") + err);

        std::vector<std::unique_ptr<MemRefBase>> inputs;
        for (size_t i = 0; i < inputFiles.size(); ++i)
        {
            auto dims = parseShape(inputShapeStrs[i]);
            auto holder = createHolder(dims);

            holder->loadFromFile(inputFiles[i]);
            inputs.push_back(std::move(holder));

            if (verbose)
            {
                std::cout << "Input " << i << ": " << inputFiles[i] << " shape ";
                inputs.back()->printShape(std::cout);
                std::cout << "\n";
            }
        }


        std::vector<std::unique_ptr<MemRefBase>> outputs;
        for (size_t i = 0; i < outputShapeStrs.size(); ++i)
        {
            auto dims = parseShape(outputShapeStrs[i]);
            auto holder = createHolder(dims);
            outputs.push_back(std::move(holder));

            if (verbose)
            {
                std::cout << "Output " << i << ": shape ";
                outputs.back()->printShape(std::cout);
                std::cout << "\n";
            }
        }



        invokeModel(func, outputs, inputs);



        bool allOk = true;
        for (size_t i = 0; i < outputs.size(); ++i)
        {
            float* data = static_cast<float*>(outputs[i]->getData());
            int64_t total = outputs[i]->getTotalElements();

            if (i < outputFiles.size() && !outputFiles[i].empty())
            {
                outputs[i]->saveToFile(outputFiles[i]);
            }

            if (i < refFiles.size() && !refFiles[i].empty())
            {
                auto ref = readReference(refFiles[i], total);
                bool ok = compareTensors(data, ref.data(), total, tolerance);

                if (!ok)
                {
                    std::cout << "Output " << i << " MISMATCH\n";
                    allOk = false;
                }
                
                else if (verbose)
                {
                    std::cout << "Output " << i << " MATCH\n";
                }
            }
        }

        for (auto& out : outputs)
        {
            out->freeData();
        }

        dlclose(exec);
        std::cout << (allOk ? "OK\n" : "FAIL\n");
        return allOk ? 0 : 1;
    }
    
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
