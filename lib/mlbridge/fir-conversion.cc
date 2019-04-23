// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "fir-conversion.h"
#include "fir-dialect.h"
#include "mlir/LLVMIR/LLVMDialect.h"
#include "mlir/LLVMIR/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Config/abi-breaking.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

using namespace Fortran;
using namespace Fortran::mlbridge;
using DialectOpConversion = mlir::DialectOpConversion;
using MLIRContext = mlir::MLIRContext;
using Pass = mlir::Pass;

namespace {

class ApplyExprConversion : public DialectOpConversion {
public:
  explicit ApplyExprConversion(MLIRContext *ctxt)
    : DialectOpConversion(ApplyExpr::getOperationName(), 1, ctxt) {}

  llvm::SmallVector<mlir::Value *, 4> rewrite(mlir::Operation *op,
      llvm::ArrayRef<mlir::Value *> operands,
      mlir::FuncBuilder &rewriter) const override {
    return {};
  }
};

class LocateExprConversion : public DialectOpConversion {
public:
  explicit LocateExprConversion(MLIRContext *ctxt)
    : DialectOpConversion(LocateExpr::getOperationName(), 1, ctxt) {}

  llvm::SmallVector<mlir::Value *, 4> rewrite(mlir::Operation *op,
      llvm::ArrayRef<mlir::Value *> operands,
      mlir::FuncBuilder &rewriter) const override {
    return {};
  }
};

class UnreachableOpConversion : public DialectOpConversion {
public:
  explicit UnreachableOpConversion(MLIRContext *ctxt)
    : DialectOpConversion(UnreachableOp::getOperationName(), 1, ctxt) {}

  llvm::SmallVector<mlir::Value *, 4> rewrite(mlir::Operation *op,
      llvm::ArrayRef<mlir::Value *> operands,
      mlir::FuncBuilder &rewriter) const override {
    rewriter.create<mlir::LLVM::UnreachableOp>(op->getLoc(),
        llvm::ArrayRef<mlir::Value *>{}, llvm::ArrayRef<mlir::Block *>{},
        llvm::ArrayRef<llvm::ArrayRef<mlir::Value *>>{}, op->getAttrs());
    return {};
  }
};

/// Convert the module from the FIR dialect to LLVM dialect
class FIRConversion : public mlir::DialectConversion {
protected:
  llvm::DenseSet<mlir::DialectOpConversion *> initConverters(
      MLIRContext *ctxt) override {
    return mlir::ConversionListBuilder<ApplyExprConversion,
        LocateExprConversion, UnreachableOpConversion>::build(&allocator, ctxt);
  }

private:
  llvm::BumpPtrAllocator allocator;
};

/// Convert to LLVM dialect
class LLVMDialectLoweringPass
  : public mlir::ModulePass<LLVMDialectLoweringPass> {
public:
  void runOnModule() override {
    auto *mod{&getModule()};
    auto ctxt{mod->getContext()};
    if (mlir::failed(FIRConversion{}.convert(mod)) ||
        mlir::failed(mlir::createStdToLLVMConverter()->convert(mod))) {
      ctxt->emitError(
          mlir::UnknownLoc::get(ctxt), "error in converting to LLVM dialect\n");
      signalPassFailure();
    }
  }
};

/// Lower to LLVM-IR and dump the module
class LLVMIRLoweringPass : public mlir::ModulePass<LLVMIRLoweringPass> {
public:
  void runOnModule() override {
    if (auto llvmModule{mlir::translateModuleToLLVMIR(getModule())}) {
      std::error_code ec;
      auto stream{llvm::raw_fd_ostream("a.ll", ec, llvm::sys::fs::F_None)};
      stream << *llvmModule << '\n';
    } else {
      auto ctxt{getModule().getContext()};
      ctxt->emitError(mlir::UnknownLoc::get(ctxt), "could not emit LLVM-IR\n");
      signalPassFailure();
    }
  }
};

}  // namespace

Pass *Fortran::mlbridge::createLLVMDialectLoweringPass() {
  return new LLVMDialectLoweringPass();
}

Pass *Fortran::mlbridge::createLLVMIRLoweringPass() {
  return new LLVMIRLoweringPass();
}
