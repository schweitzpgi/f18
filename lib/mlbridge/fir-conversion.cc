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
#include "canonicalize.h"
#include "expression.h"
#include "fir-dialect.h"
#include "fir-type.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Config/abi-breaking.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/AffineOps/AffineOps.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/LLVMIR/LLVMDialect.h"
#include "mlir/LLVMIR/LLVMLowering.h"
#include "mlir/LLVMIR/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/StandardOps/Ops.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/DialectConversion.h"

namespace Br = Fortran::mlbridge;
namespace M = mlir;

using namespace Fortran;
using namespace Fortran::mlbridge;

namespace {

using SmallVecResult = llvm::SmallVector<M::Value *, 4>;
using OperandTy = llvm::ArrayRef<M::Value *>;
using AttributeTy = llvm::ArrayRef<M::NamedAttribute>;

/// FIR type converter
/// This converts FIR types to LLVM types (for now)
class FIRTypeConverter : public M::LLVMTypeConverter {
public:
  using LLVMTypeConverter::LLVMTypeConverter;

  /// Convert FIR types to LLVM IR dialect types
  M::Type convertType(M::Type t) override {
    if (auto ref = t.dyn_cast<FIRReferenceType>()) {
      auto eleTy = convertType(ref.getEleTy());
      return M::MemRefType::get(llvm::ArrayRef<std::int64_t>{}, eleTy);
    }
    if (auto real = t.dyn_cast<FIRRealType>()) {
      auto *ctx{real.getContext()};
      auto &llvmCtx{getLLVMContext()};
      llvm::Type *llTy{nullptr};
      switch (real.getFKind()) {
      case 10: llTy = llvm::Type::getX86_FP80Ty(llvmCtx); break;
      case 16:
        // FIXME: llTy = llvm::Type::getPPC_FP128Ty(llvmCtx);
        llTy = llvm::Type::getFP128Ty(llvmCtx);
        break;
      default: assert(false && "unsupported REAL kind"); break;
      }
      return M::LLVM::LLVMType::get(ctx, llTy);
    }
    if (auto log = t.dyn_cast<FIRLogicalType>()) {
      llvm::Type *llTy{
          llvm::Type::getIntNTy(getLLVMContext(), log.getSizeInBits())};
      return M::LLVM::LLVMType::get(log.getContext(), llTy);
    }
    if (auto chr = t.dyn_cast<FIRCharacterType>()) {
      llvm::Type *llTy{
          llvm::Type::getIntNTy(getLLVMContext(), chr.getSizeInBits())};
      return M::LLVM::LLVMType::get(chr.getContext(), llTy);
    }
    if (auto tup = t.dyn_cast<FIRTupleType>()) {
      return t;  // fixme
    }
    if (auto seq = t.dyn_cast<FIRSequenceType>()) {
      return t;  // fixme
    }
    return t;
  }

  M::Type convertToLLVMType(M::Type t) {
    return LLVMTypeConverter::convertType(t);
  }
};

class FIROpConversion : public M::ConversionPattern {
protected:
  FIROpConversion(llvm::StringRef rootName, M::PatternBenefit benefit,
      M::MLIRContext *ctx, FIRTypeConverter &lowering)
    : ConversionPattern(rootName, benefit, ctx), lowering(lowering) {}

  FIRTypeConverter &lowering;
};

class AllocaExprConversion : public FIROpConversion {
public:
  explicit AllocaExprConversion(
      M::MLIRContext *ctxt, FIRTypeConverter &lowering)
    : FIROpConversion(AllocaExpr::getOperationName(), 1, ctxt, lowering) {}

  M::PatternMatchResult matchAndRewrite(M::Operation *op, OperandTy operands,
      M::PatternRewriter &rewriter) const override {
    auto alloc{M::cast<AllocaExpr>(op)};
    llvm::SmallVector<M::Value *, 1> vec;
    vec.emplace_back(rewriter.create<M::LLVM::AllocaOp>(alloc.getLoc(),
        lowering.convertType(alloc.getType()), llvm::ArrayRef<M::Value *>{},
        llvm::ArrayRef<M::NamedAttribute>{}));
    rewriter.replaceOp(op, vec);
    return matchSuccess();
  }
};

class AllocMemOpConversion : public FIROpConversion {
public:
  explicit AllocMemOpConversion(
      M::MLIRContext *ctxt, FIRTypeConverter &lowering)
    : FIROpConversion(AllocMemOp::getOperationName(), 1, ctxt, lowering) {}

  M::PatternMatchResult matchAndRewrite(M::Operation *op, OperandTy operands,
      M::PatternRewriter &rewriter) const override {
    // FIXME
    assert(false);
    return matchSuccess();
  }
};

class ApplyExprConversion : public FIROpConversion {
public:
  explicit ApplyExprConversion(M::MLIRContext *ctxt, FIRTypeConverter &lowering)
    : FIROpConversion(ApplyExpr::getOperationName(), 1, ctxt, lowering) {}

  M::PatternMatchResult matchAndRewrite(M::Operation *op, OperandTy operands,
      M::PatternRewriter &rewriter) const override {
    auto apply = M::cast<ApplyExpr>(op);
    rewriter.replaceOp(op, lowerSomeExpr(&rewriter, operands, apply));
    return matchSuccess();
  }
};

class FreeMemOpConversion : public FIROpConversion {
public:
  explicit FreeMemOpConversion(M::MLIRContext *ctxt, FIRTypeConverter &lowering)
    : FIROpConversion(FreeMemOp::getOperationName(), 1, ctxt, lowering) {}

  M::PatternMatchResult matchAndRewrite(M::Operation *op, OperandTy operands,
      M::PatternRewriter &rewriter) const override {
    // FIXME
    assert(false);
    return matchSuccess();
  }
};

class GlobalExprConversion : public FIROpConversion {
public:
  explicit GlobalExprConversion(
      M::MLIRContext *ctxt, FIRTypeConverter &lowering)
    : FIROpConversion(GlobalExpr::getOperationName(), 1, ctxt, lowering) {}

  M::PatternMatchResult matchAndRewrite(M::Operation *op, OperandTy operands,
      M::PatternRewriter &rewriter) const override {
    // FIXME
    assert(false);
    return matchSuccess();
  }
};

class LoadExprConversion : public FIROpConversion {
public:
  explicit LoadExprConversion(M::MLIRContext *ctxt, FIRTypeConverter &lowering)
    : FIROpConversion(LoadExpr::getOperationName(), 1, ctxt, lowering) {}

  M::PatternMatchResult matchAndRewrite(M::Operation *op, OperandTy operands,
      M::PatternRewriter &rewriter) const override {
    auto load = M::cast<LoadExpr>(op);
    llvm::SmallVector<M::Value *, 1> vec;
    vec.emplace_back(rewriter.create<M::LoadOp>(load.getLoc(),
        lowering.convertType(load.getType()), operands, AttributeTy{}));
    rewriter.replaceOp(op, vec);
    return matchSuccess();
  }
};

class LocateExprConversion : public FIROpConversion {
public:
  explicit LocateExprConversion(
      M::MLIRContext *ctxt, FIRTypeConverter &lowering)
    : FIROpConversion(LocateExpr::getOperationName(), 1, ctxt, lowering) {}

  M::PatternMatchResult matchAndRewrite(M::Operation *op, OperandTy operands,
      M::PatternRewriter &rewriter) const override {
    auto loc = M::cast<LocateExpr>(op);
    rewriter.replaceOp(op, lowerSomeExpr(&rewriter, operands, loc));
    return matchSuccess();
  }
};

class SelectOpConversion : public FIROpConversion {
public:
  explicit SelectOpConversion(M::MLIRContext *ctxt, FIRTypeConverter &lowering)
    : FIROpConversion(SelectOp::getOperationName(), 1, ctxt, lowering) {}

  M::PatternMatchResult matchAndRewrite(M::Operation *op, OperandTy operands,
      llvm::ArrayRef<M::Block *> destinations,
      llvm::ArrayRef<OperandTy> destOperands,
      M::PatternRewriter &rewriter) const override {
    // FIXME
    assert(false);
    return matchSuccess();
  }
};

class SelectCaseOpConversion : public FIROpConversion {
public:
  explicit SelectCaseOpConversion(
      M::MLIRContext *ctxt, FIRTypeConverter &lowering)
    : FIROpConversion(SelectCaseOp::getOperationName(), 1, ctxt, lowering) {}

  M::PatternMatchResult matchAndRewrite(M::Operation *op, OperandTy operands,
      llvm::ArrayRef<M::Block *> destinations,
      llvm::ArrayRef<OperandTy> destOperands,
      M::PatternRewriter &rewriter) const override;
};

class SelectRankOpConversion : public FIROpConversion {
public:
  explicit SelectRankOpConversion(
      M::MLIRContext *ctxt, FIRTypeConverter &lowering)
    : FIROpConversion(SelectRankOp::getOperationName(), 1, ctxt, lowering) {}

  M::PatternMatchResult matchAndRewrite(M::Operation *op, OperandTy operands,
      llvm::ArrayRef<M::Block *> destinations,
      llvm::ArrayRef<OperandTy> destOperands,
      M::PatternRewriter &rewriter) const override;
};

class SelectTypeOpConversion : public FIROpConversion {
public:
  explicit SelectTypeOpConversion(
      M::MLIRContext *ctxt, FIRTypeConverter &lowering)
    : FIROpConversion(SelectTypeOp::getOperationName(), 1, ctxt, lowering) {}

  M::PatternMatchResult matchAndRewrite(M::Operation *op, OperandTy operands,
      llvm::ArrayRef<M::Block *> destinations,
      llvm::ArrayRef<OperandTy> destOperands,
      M::PatternRewriter &rewriter) const override;
};

class StoreExprConversion : public FIROpConversion {
public:
  explicit StoreExprConversion(M::MLIRContext *ctxt, FIRTypeConverter &lowering)
    : FIROpConversion(StoreExpr::getOperationName(), 1, ctxt, lowering) {}

  M::PatternMatchResult matchAndRewrite(M::Operation *op, OperandTy operands,
      M::PatternRewriter &rewriter) const override {
    M::Value *data = operands[1];
    M::Value *addr = operands[0];
    rewriter.create<M::StoreOp>(op->getLoc(), addr, data);
    op->erase();
    return matchSuccess();
  }
};

class UndefOpConversion : public FIROpConversion {
public:
  explicit UndefOpConversion(M::MLIRContext *ctxt, FIRTypeConverter &lowering)
    : FIROpConversion(UndefOp::getOperationName(), 1, ctxt, lowering) {}

  // rewrite FIR Undef to an LLVM IR undef
  M::PatternMatchResult matchAndRewrite(M::Operation *op, OperandTy operands,
      M::PatternRewriter &rewriter) const override {
    auto undef = M::cast<UndefOp>(op);
    llvm::SmallVector<M::Value *, 1> vec;
    vec.emplace_back(rewriter.create<M::LLVM::UndefOp>(
        undef.getLoc(), lowering.convertToLLVMType(undef.getType())));
    rewriter.replaceOp(op, vec);
    return matchSuccess();
  }
};

class UnreachableOpConversion : public FIROpConversion {
public:
  explicit UnreachableOpConversion(
      M::MLIRContext *ctxt, FIRTypeConverter &lowering)
    : FIROpConversion(UnreachableOp::getOperationName(), 1, ctxt, lowering) {}

  M::PatternMatchResult matchAndRewrite(M::Operation *op, OperandTy operands,
      M::PatternRewriter &rewriter) const override {
    llvm::SmallVector<M::Block *, 1> destinations;
    llvm::SmallVector<OperandTy, 1> destOperands;
    rewriter.create<M::LLVM::UnreachableOp>(
        op->getLoc(), operands, destinations, destOperands, op->getAttrs());
    return matchSuccess();
  }
};

// Lower a SELECT operation into a cascade of conditional branches. The last
// case must be the `true` condition.
inline void rewriteSelectConstruct(M::Operation *op, OperandTy operands,
    llvm::ArrayRef<M::Block *> dests, llvm::ArrayRef<OperandTy> destOperands,
    M::OpBuilder &rewriter) {
  llvm::SmallVector<M::Value *, 1> noargs;
  llvm::SmallVector<M::Block *, 8> blocks;
  auto loc{op->getLoc()};
  blocks.push_back(rewriter.getInsertionBlock());
  for (std::size_t i = 1; i < dests.size(); ++i)
    blocks.push_back(rewriter.createBlock(dests[0]));
  rewriter.setInsertionPointToEnd(blocks[0]);
  if (dests.size() == 1) {
    rewriter.create<M::BranchOp>(loc, dests[0], destOperands[0]);
    return;
  }
  rewriter.create<M::CondBranchOp>(
      loc, operands[1], dests[0], destOperands[0], blocks[1], noargs);
  for (std::size_t i = 1; i < dests.size() - 1; ++i) {
    rewriter.setInsertionPointToEnd(blocks[i]);
    rewriter.create<M::CondBranchOp>(
        loc, operands[i + 1], dests[i], destOperands[i], blocks[i + 1], noargs);
  }
  std::size_t last{dests.size() - 1};
  rewriter.setInsertionPointToEnd(blocks[last]);
  rewriter.create<M::BranchOp>(loc, dests[last], destOperands[last]);
}

/// Convert FIR dialect to standard dialect
class FIRToStdLoweringPass : public M::ModulePass<FIRToStdLoweringPass> {
  M::OpBuilder *builder;

  void lowerSelect(M::Operation *op) {
    if (M::dyn_cast<SelectCaseOp>(op) || M::dyn_cast<SelectRankOp>(op) ||
        M::dyn_cast<SelectTypeOp>(op)) {
      // build the lists of operands and successors
      llvm::SmallVector<M::Value *, 4> operands{
          op->operand_begin(), op->operand_end()};
      llvm::SmallVector<M::Block *, 2> destinations;
      destinations.reserve(op->getNumSuccessors());
      llvm::SmallVector<llvm::ArrayRef<M::Value *>, 2> destOperands;
      unsigned firstSuccOpd = op->getSuccessorOperandIndex(0);
      for (unsigned i = 0, seen = 0, e = op->getNumSuccessors(); i < e; ++i) {
        destinations.push_back(op->getSuccessor(i));
        unsigned n = op->getNumSuccessorOperands(i);
        destOperands.push_back(
            llvm::makeArrayRef(operands.data() + firstSuccOpd + seen, n));
        seen += n;
      }
      // do the rewrite
      rewriteSelectConstruct(op,
          llvm::makeArrayRef(operands.data(), operands.data() + firstSuccOpd),
          destinations, destOperands, *builder);
      op->erase();
    }
  }

public:
  void runOnModule() override {
    for (auto &fn : getModule()) {
      M::OpBuilder rewriter{&fn.getBody()};
      builder = &rewriter;
      fn.walk([&](M::Operation *op) { lowerSelect(op); });
    }
    auto &context{getContext()};
    FIRTypeConverter typeConverter{&context};
    M::OwningRewritePatternList patterns;
    M::RewriteListBuilder<AllocaExprConversion, AllocMemOpConversion,
        ApplyExprConversion, FreeMemOpConversion, GlobalExprConversion,
        LoadExprConversion, LocateExprConversion, StoreExprConversion,
        UndefOpConversion, UnreachableOpConversion>::build(patterns, &context,
        typeConverter);
    M::ConversionTarget target{context};
    target.addLegalDialect<M::AffineOpsDialect, M::LLVM::LLVMDialect,
        M::StandardOpsDialect>();
    if (M::failed(M::applyConversionPatterns(
            getModule(), target, typeConverter, std::move(patterns)))) {
      context.emitError(M::UnknownLoc::get(&context),
          "error in converting to standard dialect\n");
      signalPassFailure();
    }
  }
};

/// Lower from LLVM IR dialect to proper LLVM-IR and dump the module
class LLVMIRLoweringPass : public M::ModulePass<LLVMIRLoweringPass> {
public:
  void runOnModule() override {
    if (auto llvmModule{M::translateModuleToLLVMIR(getModule())}) {
      std::error_code ec;
      auto stream{llvm::raw_fd_ostream("a.ll", ec, llvm::sys::fs::F_None)};
      stream << *llvmModule << '\n';
    } else {
      auto ctxt{getModule().getContext()};
      ctxt->emitError(M::UnknownLoc::get(ctxt), "could not emit LLVM-IR\n");
      signalPassFailure();
    }
  }
};

}  // namespace

M::Pass *Br::createFIRToStdPass() { return new FIRToStdLoweringPass(); }

M::Pass *Br::createStdToLLVMPass() { return M::createConvertToLLVMIRPass(); }

M::Pass *Br::createLLVMDialectToLLVMPass() { return new LLVMIRLoweringPass(); }
