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

#include "canonicalize.h"
#include "expr-lowering.h"
#include "fir-dialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/AffineOps/AffineOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/StandardOps/Ops.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

namespace M = mlir;
namespace Br = Fortran::mlbridge;

using namespace Fortran;
using namespace Fortran::mlbridge;

namespace {

using OperandTy = llvm::ArrayRef<M::Value *>;

class ApplyExprLower : public M::ConversionPattern {
public:
  explicit ApplyExprLower(M::MLIRContext *ctxt)
    : ConversionPattern(ApplyExpr::getOperationName(), 1, ctxt) {}

  M::PatternMatchResult matchAndRewrite(M::Operation *op, OperandTy operands,
      M::PatternRewriter &rewriter) const override {
    auto apply = M::cast<ApplyExpr>(op);
    rewriter.replaceOp(op, lowerSomeExpr(&rewriter, operands, {apply}));
    return matchSuccess();
  }
};

class LocateExprLower : public M::ConversionPattern {
public:
  explicit LocateExprLower(M::MLIRContext *ctxt)
    : ConversionPattern(LocateExpr::getOperationName(), 1, ctxt) {}

  M::PatternMatchResult matchAndRewrite(M::Operation *op, OperandTy operands,
      M::PatternRewriter &rewriter) const override {
    auto loc = M::cast<LocateExpr>(op);
    rewriter.replaceOp(op, lowerSomeExpr(&rewriter, operands, {loc}));
    return matchSuccess();
  }
};

/// The NOP type converter
/// This type converter does nothing
class NOPTypeConverter : public M::TypeConverter {
public:
  using TypeConverter::TypeConverter;

protected:
  M::Type convertType(M::Type t) override { return t; }
};

/// Convert high-level FIR dialect to FIR dialect
class FIRLoweringPass : public M::ModulePass<FIRLoweringPass> {
public:
  void runOnModule() override {
    auto &context{getContext()};
    NOPTypeConverter typeConverter;
    M::OwningRewritePatternList patterns;
    M::RewriteListBuilder<ApplyExprLower, LocateExprLower>::build(
        patterns, &context);
    M::ConversionTarget target{context};
    target.addLegalDialect<M::AffineOpsDialect, M::LLVM::LLVMDialect,
        M::StandardOpsDialect>();
    target.addLegalOp<AllocaExpr, AllocMemOp, FreeMemOp, GlobalExpr, LoadExpr,
        SelectOp, SelectCaseOp, SelectRankOp, SelectTypeOp, StoreExpr, UndefOp,
        UnreachableOp>();
    if (M::failed(M::applyConversionPatterns(
            getModule(), target, typeConverter, std::move(patterns)))) {
      context.emitError(M::UnknownLoc::get(&context),
          "error in canonicalizing FIR dialect\n");
      signalPassFailure();
    }
  }
};

}  // namespace

M::Pass *Br::createFIRLoweringPass() { return new FIRLoweringPass(); }
