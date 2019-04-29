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
#include "mlir/IR/StandardTypes.h"
#include "mlir/LLVMIR/LLVMDialect.h"
#include "mlir/LLVMIR/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Config/abi-breaking.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

using namespace Fortran;
using namespace Fortran::mlbridge;
using DialectOpConversion = mlir::DialectOpConversion;
using MLIRContext = mlir::MLIRContext;
using Pass = mlir::Pass;

namespace {

using SmallVecResult = llvm::SmallVector<mlir::Value *, 4>;
using OperandTy = llvm::ArrayRef<mlir::Value *>;
using AttributeTy = llvm::ArrayRef<mlir::NamedAttribute>;

// throw-away code: create a dummy call as a proxy for a temporary
mlir::Value *dummyTemporary(mlir::Operation *op, mlir::FuncBuilder &rewriter) {
  mlir::Function *dummy =
      op->getFunction()->getModule()->getNamedFunction("temp");
  auto &llvmContext = static_cast<mlir::LLVM::LLVMDialect *>(
      op->getContext()->getRegisteredDialect("llvm"))
                          ->getLLVMModule()
                          .getContext();
  auto returnTy = mlir::LLVM::LLVMType::get(op->getContext(),
      mlir::LLVM::LLVMType::get(
          op->getContext(), llvm::IRBuilder<>(llvmContext).getIntNTy(64))
          .cast<mlir::LLVM::LLVMType>()
          .getUnderlyingType()
          ->getPointerTo());
  if (!dummy) {
    auto dummyType = rewriter.getFunctionType({}, returnTy);
    dummy = new mlir::Function{rewriter.getUnknownLoc(), "temp", dummyType};
    op->getFunction()->getModule()->getFunctions().push_back(dummy);
  }
  return rewriter
      .create<mlir::LLVM::CallOp>(
          op->getLoc(), returnTy, rewriter.getFunctionAttr(dummy), OperandTy{})
      .getResult(0);
}

class ApplyExprConversion : public DialectOpConversion {
public:
  explicit ApplyExprConversion(MLIRContext *ctxt)
    : DialectOpConversion(ApplyExpr::getOperationName(), 1, ctxt) {}

  SmallVecResult rewrite(mlir::Operation *op, OperandTy operands,
      mlir::FuncBuilder &rewriter) const override {
    SmallVecResult result;
#if 0
    mlir::Type elementTypePtr = rewriter.getIntegerType(64);  // FIXME
    mlir::Value *dataPtr{rewriter.create<mlir::LLVM::GEPOp>(
        op->getLoc(), elementTypePtr, OperandTy{}, AttributeTy{})};
#else
    auto &llvmContext = static_cast<mlir::LLVM::LLVMDialect *>(
        op->getContext()->getRegisteredDialect("llvm"))
                            ->getLLVMModule()
                            .getContext();
    auto returnTy = mlir::LLVM::LLVMType::get(
        op->getContext(), llvm::IRBuilder<>(llvmContext).getIntNTy(64));
    mlir::Value *dataPtr = dummyTemporary(op, rewriter);
#endif
    result.push_back(rewriter.create<mlir::LLVM::LoadOp>(
        op->getLoc(), returnTy, OperandTy{dataPtr}, AttributeTy{}));
    return result;
  }
};

class LocateExprConversion : public DialectOpConversion {
public:
  explicit LocateExprConversion(MLIRContext *ctxt)
    : DialectOpConversion(LocateExpr::getOperationName(), 1, ctxt) {}

  SmallVecResult rewrite(mlir::Operation *op, OperandTy operands,
      mlir::FuncBuilder &rewriter) const override {
    SmallVecResult result;
#if 0
    mlir::Type elementTypePtr = rewriter.getIntegerType(64);  // FIXME
    result.push_back(rewriter.create<mlir::LLVM::GEPOp>(
        op->getLoc(), elementTypePtr, OperandTy{}, AttributeTy{}));
#else
    result.push_back(dummyTemporary(op, rewriter));
#endif
    return result;
  }
};

class AllocaExprConversion : public DialectOpConversion {
public:
  explicit AllocaExprConversion(MLIRContext *ctxt)
    : DialectOpConversion(AllocaExpr::getOperationName(), 1, ctxt) {}

  SmallVecResult rewrite(mlir::Operation *op, OperandTy operands,
      mlir::FuncBuilder &rewriter) const override {
    return {};
  }
};

class LoadExprConversion : public DialectOpConversion {
public:
  explicit LoadExprConversion(MLIRContext *ctxt)
    : DialectOpConversion(LoadExpr::getOperationName(), 1, ctxt) {}

  SmallVecResult rewrite(mlir::Operation *op, OperandTy operands,
      mlir::FuncBuilder &rewriter) const override {
    return {};
  }
};

class StoreExprConversion : public DialectOpConversion {
public:
  explicit StoreExprConversion(MLIRContext *ctxt)
    : DialectOpConversion(StoreExpr::getOperationName(), 1, ctxt) {}

  SmallVecResult rewrite(mlir::Operation *op, OperandTy operands,
      mlir::FuncBuilder &rewriter) const override {
    mlir::Value *dataPtr = operands[1];  // FIXME
    rewriter.create<mlir::LLVM::StoreOp>(op->getLoc(), operands[0], dataPtr);
    return {};
  }
};

class UnreachableOpConversion : public DialectOpConversion {
public:
  explicit UnreachableOpConversion(MLIRContext *ctxt)
    : DialectOpConversion(UnreachableOp::getOperationName(), 1, ctxt) {}

  SmallVecResult rewrite(mlir::Operation *op, OperandTy operands,
      mlir::FuncBuilder &rewriter) const override {
    rewriter.create<mlir::LLVM::UnreachableOp>(op->getLoc(), OperandTy{},
        llvm::ArrayRef<mlir::Block *>{}, llvm::ArrayRef<OperandTy>{},
        op->getAttrs());
    return {};
  }
};

/// Convert the module from the FIR dialect to LLVM dialect
class FIRConversion : public mlir::DialectConversion {
protected:
  llvm::DenseSet<mlir::DialectOpConversion *> initConverters(
      MLIRContext *ctxt) override {
    return mlir::ConversionListBuilder<ApplyExprConversion,
        LocateExprConversion, LoadExprConversion, StoreExprConversion,
        AllocaExprConversion, UnreachableOpConversion>::build(&allocator, ctxt);
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
