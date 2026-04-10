#include "backend/codegen.hpp"
#include "middle_end/mlir_builders.hpp"

// ── MLIR ───────────────────────────────────────────────────────────────────
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"

// ── MLIR IR ───────────────────────────────────────────────────────────────────
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/AffineMap.h"

// ── MLIR passes ───────────────────────────────────────────────────────────────────
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"

// ── conversions ───────────────────────────────────────────────────────────────────
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"

 




// ── buffering ───────────────────────────────────────────────────────────────────
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"

// ── MLIR to LLVM ───────────────────────────────────────────────────────────────────
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

// ── LLVM ───────────────────────────────────────────────────────────────────
#include "llvm/IR/LegacyPassManager.h"

#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/CodeGen/Passes.h"





#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cstring>

namespace tc
{

    static void registerAllDialects(mlir::MLIRContext& ctx)
    {
        ctx.loadDialect<
            mlir::arith::ArithDialect,
            mlir::func::FuncDialect,
            mlir::linalg::LinalgDialect,
            mlir::memref::MemRefDialect,
            mlir::scf::SCFDialect,
            mlir::tensor::TensorDialect,
            mlir::affine::AffineDialect,
            mlir::math::MathDialect,
            mlir::bufferization::BufferizationDialect,
            mlir::LLVM::LLVMDialect
        >();
    }

    // tc::DataType to mlir::Type
    static mlir::Type mlirElemType(DataType dt, mlir::MLIRContext* ctx)
    {
        switch (dt)
        {
            case DataType::FLOAT:   return mlir::Float32Type::get(ctx);
            case DataType::DOUBLE:  return mlir::Float64Type::get(ctx);
            case DataType::FLOAT16: return mlir::Float16Type::get(ctx);
            case DataType::INT8:    return mlir::IntegerType::get(ctx,  8);
            case DataType::INT16:   return mlir::IntegerType::get(ctx, 16);
            case DataType::INT32:   return mlir::IntegerType::get(ctx, 32);
            case DataType::INT64:   return mlir::IntegerType::get(ctx, 64);
            case DataType::UINT8:   return mlir::IntegerType::get(ctx,  8, mlir::IntegerType::Unsigned);
            case DataType::BOOL:    return mlir::IntegerType::get(ctx,  1);
            default:                return mlir::Float32Type::get(ctx);
        }
    }

    CodeGen::CodeGen(mlir::MLIRContext& mlir_ctx, llvm::LLVMContext& llvm_ctx) : mlir_ctx_(mlir_ctx), llvm_ctx_(llvm_ctx)
    {
        registerAllDialects(mlir_ctx_);
        mlir::registerBuiltinDialectTranslation(mlir_ctx_);
        mlir::registerLLVMDialectTranslation(mlir_ctx_);

        llvm::InitializeAllTargetInfos();
        llvm::InitializeAllTargets();
        llvm::InitializeAllTargetMCs();
        llvm::InitializeAllAsmParsers();
        llvm::InitializeAllAsmPrinters();

        LLVMInitializeAArch64Target();
        LLVMInitializeAArch64TargetInfo();
        LLVMInitializeAArch64TargetMC();
        LLVMInitializeAArch64AsmPrinter();
    }

    //ranked tensor from data
    mlir::RankedTensorType CodeGen::makeTensorType(DataType dt, const TensorShape& shape) const
    {
        auto elem = mlirElemType(dt, &mlir_ctx_);
        llvm::SmallVector<int64_t> dims;
        dims.reserve(shape.dims.size());

        for (auto d : shape.dims)
            dims.push_back(d < 0 ? mlir::ShapedType::kDynamic : d);

        return mlir::RankedTensorType::get(dims, elem);
    }

    mlir::RankedTensorType CodeGen::tensorTypeOf(const Tensor& t) const
    {
        return makeTensorType(t.getDtype(), t.getShape());
    }

    mlir::Value CodeGen::makeWeightConstant(mlir::OpBuilder& builder,
                                            mlir::Location   loc,
                                            const Tensor&    w) const
    {
        auto rtt = tensorTypeOf(w);

        if (w.getDtype() == DataType::FLOAT && w.hasValidData())
        {
            auto span = w.getDataAs<float>();
            llvm::SmallVector<float> vals(span.begin(), span.end());
            auto attr = mlir::DenseElementsAttr::get(rtt, llvm::ArrayRef<float>(vals));
            return mlir::arith::ConstantOp::create(builder, loc, rtt, attr);
        }

        if (w.getDtype() == DataType::INT64 && w.hasValidData())
        {
            auto span = w.getDataAs<int64_t>();
            llvm::SmallVector<int64_t> vals(span.begin(), span.end());
            auto attr = mlir::DenseIntElementsAttr::get(rtt, llvm::ArrayRef<int64_t>(vals));
            return mlir::arith::ConstantOp::create(builder, loc, rtt, attr);
        }

        //TODO - other data types

        // else zero tensor
        // std::cout << "No data for ConstantOp" << std::endl;
        auto zero = builder.getZeroAttr(rtt.getElementType());
        auto attr = mlir::DenseElementsAttr::get(rtt, zero);
        return mlir::arith::ConstantOp::create(builder, loc, rtt, attr);
    }

    void CodeGen::runOptPipeline(mlir::ModuleOp mod)
    {
        //TODO - optimize
        return;
    }


    mlir::OwningOpRef<mlir::ModuleOp> CodeGen::runLoweringPipeline(mlir::ModuleOp mod)
    {
        //FIXME - bufferization using external mlir-opt call
        std::string tempInput   = std::filesystem::temp_directory_path() / "temp_input.mlir";
        std::string tempOutput  = std::filesystem::temp_directory_path() / "temp_output.mlir";

        {
            std::error_code ec;
            llvm::raw_fd_ostream os(tempInput, ec);

            if (ec)
            {
                llvm::errs() << "Failed to create temp file: " << ec.message() << "\n";
                return nullptr;
            }

            mod->print(os);
        }

        std::string cmd = std::string(MLIR_OPT_PATH) +
            " --one-shot-bufferize=\"bufferize-function-boundaries=true allow-return-allocs-from-loops=true\"" +
            " " + tempInput + " -o " + tempOutput;

        int ret = std::system(cmd.c_str());
        if (ret != 0)
        {
            llvm::errs() << "mlir-opt failed with code " << ret << "\n";
            return nullptr;
        }

        mlir::MLIRContext *ctx = mod->getContext();

        auto newMod = mlir::parseSourceFile<mlir::ModuleOp>(tempOutput, ctx);
        if (!newMod)
        {
            llvm::errs() << "Failed to parse bufferized module\n";
            return nullptr;
        }

        std::filesystem::remove(tempInput);
        std::filesystem::remove(tempOutput);

        return newMod;
    }

    
    std::unique_ptr<llvm::Module> CodeGen::translateToLLVMIR(mlir::ModuleOp mod, llvm::raw_ostream &os)
    {
        auto llvmModule = mlir::translateModuleToLLVMIR(mod, llvm_ctx_);
        if (!llvmModule)
        {
            throw std::runtime_error("Failed to translate MLIR module to LLVM IR");
        }

        return llvmModule;
    }

    void CodeGen::emitObject(llvm::Module *llvmModule, const std::string &filename, const CodeGenOptions& opts)
    {
        if (!llvmModule) throw std::runtime_error("Null module");

        llvm::Triple targetTriple(opts.target_triple);
        std::string error;
        const llvm::Target *target = llvm::TargetRegistry::lookupTarget(targetTriple, error);

        if (!target) throw std::runtime_error("Target lookup failed: " + error);

        llvm::TargetOptions opt;
        opt.FloatABIType = llvm::FloatABI::Soft;

        std::unique_ptr<llvm::TargetMachine> TM(target->createTargetMachine(
            targetTriple, opts.cpu, opts.features, opt,
            llvm::Reloc::PIC_, llvm::CodeModel::Small, llvm::CodeGenOptLevel::None
        ));

        llvmModule->setDataLayout(TM->createDataLayout());
        llvmModule->setTargetTriple(targetTriple);

        std::error_code ec;
        llvm::raw_fd_ostream dest(filename, ec, llvm::sys::fs::OF_None);
        if (ec) throw std::runtime_error("Cannot open file: " + ec.message());


        {
            llvm::legacy::PassManager pm;
            if (TM->addPassesToEmitFile(pm, dest, nullptr, llvm::CodeGenFileType::ObjectFile))
                throw std::runtime_error("Cannot emit assembly");

            pm.run(*llvmModule);
        }


        dest.flush();
    }







    
    void CodeGen::processNode(mlir::OpBuilder& builder,
                              const Node&      node,
                              ValueMap&        vmap,
                              const Graph&     graph) const
    {
        auto loc = builder.getUnknownLoc();
        auto nodeType = node.getOpType();

        // get data by tensor name
        auto resolve = [&](const std::string& name) -> mlir::Value
        {
            if (name.empty())
                throw std::runtime_error("Empty tensor name");

            auto it = vmap.find(name);
            if (it != vmap.end()) return it->second;

            auto opt = graph.findTensor(name);
            if (opt && (*opt)->hasData())
            {
                auto val = makeWeightConstant(builder, loc, **opt);
                vmap[name] = val;
                return val;
            }
            throw std::runtime_error(
                "Cannot resolve tensor '" + name +
                "' in node '" + node.getName() + "'");
        };


        






        // ── Add / Mul ───────────────────────────────────────────────────────────────────
        if (nodeType == OpType::Add || nodeType == OpType::Mul)
        {
            auto lhs = resolve(node.getInputs()[0]);
            auto rhs = resolve(node.getInputs()[1]);


            //always use generic build //TODO - straight generation when same shapes


            auto result = buildElementwiseGeneric(nodeType, builder, loc, lhs, rhs, &mlir_ctx_);
            vmap[node.getOutputs()[0]] = result;
        
            return;
        }








        // ── MatMul ────────────────────────────────────────────────────────────────
        if (nodeType == OpType::MatMul)
        {
            auto A = resolve(node.getInputs()[0]);
            auto B = resolve(node.getInputs()[1]);

            //TODO - if 2D or 3D use linalg.batch_matmul
            auto result = buildMatmulGeneric(builder, loc, A, B, false, false, &mlir_ctx_);
            vmap[node.getOutputs()[0]] = result;
            return;
        }




        // ── Gemm ──────────────────────────────────────────────────────────────────
        if (nodeType == OpType::Gemm)
        {
            auto A = resolve(node.getInputs()[0]);
            auto B = resolve(node.getInputs()[1]);

            mlir::Value C = nullptr;

            if (node.getInputs().size() >= 3 && !node.getInputs()[2].empty())
                C = resolve(node.getInputs()[2]);

            float alpha = 1.0f, beta = 1.0f;
            bool transA = false, transB = false;

            if (node.hasAttribute("alpha"))     alpha   = node.getAttribute("alpha").asFloat();
            if (node.hasAttribute("beta"))      beta    = node.getAttribute("beta").asFloat();
            if (node.hasAttribute("transA"))    transA  = node.getAttribute("transA").asInt() != 0;
            if (node.hasAttribute("transB"))    transB  = node.getAttribute("transB").asInt() != 0;

            // A[] * B[]
            mlir::Value result = buildMatmulGeneric(builder, loc, A, B, transA, transB, &mlir_ctx_);
            auto resultType = llvm::cast<mlir::RankedTensorType>(result.getType());

            // dynamic dimensions
            llvm::SmallVector<mlir::Value> dynSizes;
            for (unsigned i = 0; i < resultType.getRank(); ++i)
            {
                if (resultType.isDynamicDim(i))
                {
                    dynSizes.push_back(mlir::tensor::DimOp::create(builder, loc, result, i));
                }
            }


            auto alphaTensor = createConstantTensor(builder, loc, resultType, dynSizes, alpha);
            result = buildElementwiseGeneric(OpType::Mul, builder, loc, result, alphaTensor, &mlir_ctx_);
            

            // + С * beta
            if (C)
            {
                mlir::Value Cval = C;

                // scaling C
                auto betaTensor = createConstantTensor(builder, loc, resultType, dynSizes, beta);
                Cval = buildElementwiseGeneric(OpType::Mul, builder, loc, Cval, betaTensor, &mlir_ctx_);
                

                // result + C
                result = buildElementwiseGeneric(OpType::Add, builder, loc, result, Cval, &mlir_ctx_);
            }

            vmap[node.getOutputs()[0]] = result;
            return;
        }
        

        // ── Relu ──────────────────────────────────────────────────────────────────
        if (nodeType == OpType::Relu)
        {
            auto input = resolve(node.getInputs()[0]);
            auto result = buildReLUGeneric(builder, loc, input, &mlir_ctx_);
            vmap[node.getOutputs()[0]] = result;

            return;
        }
        


        // ── Shape ─────────────────────────────────────────────────────────────────
        if (nodeType == OpType::Shape)
        {
            auto input = resolve(node.getInputs()[0]);

            int64_t start = 0;
            int64_t end = std::numeric_limits<int64_t>::max();

            if (node.hasAttribute("start"))
                start = node.getAttribute("start").asInt();

            if (node.hasAttribute("end"))
                end = node.getAttribute("end").asInt();

            auto result = buildShapeOp(builder, loc, input, start, end);
            vmap[node.getOutputs()[0]] = result;

            return;
        }


        // ── Reshape ───────────────────────────────────────────────────────────────
        if (nodeType == OpType::Reshape)
        {
            auto data = resolve(node.getInputs()[0]);
            auto shape = resolve(node.getInputs()[1]);

            bool allowZero = false;

            if (node.hasAttribute("allowzero"))
                allowZero = node.getAttribute("allowzero").asInt() != 0;

            auto result = buildReshapeOp(builder, loc, data, shape, allowZero);
            vmap[node.getOutputs()[0]] = result;

            return;
        }


        // ── Concat ────────────────────────────────────────────────────────────────
        if (nodeType == OpType::Concat)
        {
            int64_t axis = node.getAttribute("axis").asInt();

            llvm::SmallVector<mlir::Value> inputs;
            inputs.reserve(node.getInputs().size());

            for (const auto& inputName : node.getInputs())
                inputs.push_back(resolve(inputName));

            auto result = buildConcatOp(builder, loc, inputs, axis);
            vmap[node.getOutputs()[0]] = result;

            return;
        }


        // ── Conv2d ────────────────────────────────────────────────────────────────
        if (nodeType == OpType::Conv)
        {
            auto input = resolve(node.getInputs()[0]);
            auto weights = resolve(node.getInputs()[1]);

            std::optional<mlir::Value> bias = std::nullopt;
            if (node.getInputs().size() >= 3 && !node.getInputs()[2].empty())
                bias = resolve(node.getInputs()[2]);

            std::vector<int64_t> kernelShape = {};
            if (node.hasAttribute("kernel_shape"))
                kernelShape = node.getAttribute("kernel_shape").asInts();

            std::vector<int64_t> strides = {1, 1};
            if (node.hasAttribute("strides"))
                strides = node.getAttribute("strides").asInts();

            std::vector<int64_t> pads = {0, 0, 0, 0};
            if (node.hasAttribute("pads"))
                pads = node.getAttribute("pads").asInts();

            std::vector<int64_t> dilations = {1, 1};
            if (node.hasAttribute("dilations"))
                dilations = node.getAttribute("dilations").asInts();

            int64_t group = 1;
            if (node.hasAttribute("group"))
                group = node.getAttribute("group").asInt();

            llvm::StringRef autoPad = "NOTSET";
            if (node.hasAttribute("auto_pad"))
                autoPad = node.getAttribute("auto_pad").asString();






            auto result = buildConv2dOp(builder, 
                                        loc,
                                        input,
                                        weights,
                                        bias,
                                        kernelShape,
                                        strides,
                                        pads,
                                        dilations,
                                        group,
                                        autoPad,
                                        &mlir_ctx_);



            vmap[node.getOutputs()[0]] = result;
            return;
        }





        std::cerr << "unsupported op '"
                << node.getOpStr() << "' (node: " << node.getName()
                << "), skipping\n";
    }



    

    void CodeGen::lowerToLLVM(mlir::ModuleOp mod)
    {
        mlir::PassManager pm(mod->getContext());

        mod.walk([](mlir::func::FuncOp funcOp)
        {
            funcOp->setAttr("llvm.emit_c_interface", mlir::UnitAttr::get(funcOp.getContext()));
        });

        pm.addPass(mlir::createConvertLinalgToLoopsPass());
        pm.addPass(mlir::createLowerAffinePass());

        pm.addPass(mlir::createSCFToControlFlowPass());


        pm.addPass(mlir::createArithToLLVMConversionPass());
        pm.addPass(mlir::createConvertFuncToLLVMPass());



        pm.addPass(mlir::memref::createExpandStridedMetadataPass());

        pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());

        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());


        pm.addPass(mlir::createConvertIndexToLLVMPass());


        pm.addPass(mlir::createConvertControlFlowToLLVMPass());
        pm.addPass(mlir::createConvertToLLVMPass());  
        pm.addPass(mlir::createReconcileUnrealizedCastsPass());

        if (mlir::failed(pm.run(mod)))
        {
            mod->dump();
            throw std::runtime_error("Lowering to LLVM dialect failed");
        }

        std::cout << "Successfully lowered to LLVM dialect\n";
    }


    int CodeGen::generate(const Graph& graph, const CodeGenOptions& opts,
                            const std::string& mlir_out, const std::string& asm_out)
    {
        auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&mlir_ctx_), graph.getName());

        mlir::OpBuilder builder(&mlir_ctx_);
        builder.setInsertionPointToEnd(module.getBody());

        llvm::SmallVector<mlir::Type> arg_types;
        for (const auto& inp_name : graph.getInputs())
        {
            auto opt = graph.findTensor(inp_name);
            if (!opt)
                throw std::runtime_error("Input tensor not found: " + inp_name);
            arg_types.push_back(tensorTypeOf(**opt));
        }

        llvm::SmallVector<mlir::Type> ret_types;
        for (const auto& out_name : graph.getOutputs())
        {
            auto opt = graph.findTensor(out_name);
            if (!opt)
                throw std::runtime_error("Output tensor not found: " + out_name);
            ret_types.push_back(tensorTypeOf(**opt));
        }

        auto func_type = builder.getFunctionType(arg_types, ret_types);
        auto func = mlir::func::FuncOp::create(builder.getUnknownLoc(), graph.getName(), func_type);

        module.push_back(func);
        func.addEntryBlock();

        builder.setInsertionPointToStart(&func.getBody().front());




        ValueMap vmap;
        for (size_t i = 0; i < graph.getInputs().size(); ++i)
            vmap[graph.getInputs()[i]] = func.getArgument(static_cast<unsigned>(i));

        auto sorted = graph.topologicalSort();
        for (const auto& node : sorted)
            processNode(builder, *node, vmap, graph);


        llvm::SmallVector<mlir::Value> ret_vals;
        for (const auto& out_name : graph.getOutputs())
        {
            auto it = vmap.find(out_name);
            if (it == vmap.end())
                throw std::runtime_error(
                    "Output tensor '" + out_name + "' not computed");
            ret_vals.push_back(it->second);
        }


        mlir::func::ReturnOp::create(builder, builder.getUnknownLoc(), ret_vals);

        




        if (opts.print_mlir)
        {
            llvm::outs() << "\nMLIR representation:\n";
            module.print(llvm::outs());
            llvm::outs() << "\n";
        }

        if (!mlir_out.empty())
        {
            std::error_code ec;
            llvm::raw_fd_ostream os(mlir_out, ec);
            if (ec)
            {
                throw std::runtime_error("Cannot open MLIR output file: " + ec.message());
            }
            module->print(os);
            os.flush();
        }

        if (mlir::failed(mlir::verify(module)))
            throw std::runtime_error("MLIR module verification failed");


        auto bufferized = runLoweringPipeline(module);

        if (!opts.mlir_out.empty())
        {
            std::error_code ec;
            llvm::raw_fd_ostream ofs(opts.mlir_out.string(), ec);
            if (!ec) module.print(ofs);
        }

        lowerToLLVM(*bufferized);

        auto llvmModule = translateToLLVMIR(*bufferized, llvm::outs());
        
        if (opts.optimize) runOptPipeline(module);

        std::string asm_out_final = (asm_out == "") ? "out.s" : asm_out;
        emitObject(llvmModule.get(), asm_out_final, opts);
        std::cout << "Asm code for " << opts.target_triple << " generated successfully" << std::endl;
        return 0;
    }




    void printMLIRHelp()
    {
        std::cout <<
                R"(
                MLIR / LLVM codegen options:
                --print-mlir            Print MLIR before optimization
                
                --mlir-out=<path>       Write MLIR to file
    )";
    }





















    CodeGenOptions parseMLIROptions(int argc, char* argv[])
    {
        CodeGenOptions opts;

        auto startsWith = [](const std::string& s, const std::string& prefix)
        {
            return s.rfind(prefix, 0) == 0;
        };

        auto getValue = [](const std::string& s, const std::string& prefix)
        {
            return s.substr(prefix.size());
        };






        for (int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];

            if (arg == "--print-mlir")         { opts.print_mlir     = true; continue; }
            if (arg == "--print-mlir-opt")     { opts.print_mlir_opt = true; continue; }
            if (arg == "--no-optimize")        { opts.optimize       = false; continue; }

            if (startsWith(arg, "--target-triple="))
            { opts.target_triple = getValue(arg, "--target-triple="); continue; }

            if (startsWith(arg, "--cpu="))
            { opts.cpu = getValue(arg, "--cpu="); continue; }

            if (startsWith(arg, "--features="))
            { opts.features = getValue(arg, "--features="); continue; }

            if (startsWith(arg, "--mlir-out="))
            { opts.mlir_out = getValue(arg, "--mlir-out="); continue; }

        }

        return opts;
    }

} // namespace tc
