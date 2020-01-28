//===-- StdConverter.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/optimizer/Transforms/StdConverter.h"
#include "fir/Dialect/FIRAttr.h"
#include "fir/Dialect/FIRDialect.h"
#include "fir/Dialect/FIROpsSupport.h"
#include "fir/Dialect/FIRType.h"
#include "fir/KindMapping.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"

// This module performs the conversion of some FIR operations.
// Convert some FIR types to standard dialect?

static llvm::cl::opt<bool>
    ClDisableFirToStd("disable-fir2std",
                      llvm::cl::desc("disable FIR to standard pass"),
                      llvm::cl::init(false), llvm::cl::Hidden);

namespace fir {
namespace {

using SmallVecResult = llvm::SmallVector<mlir::Value, 4>;
using OperandTy = llvm::ArrayRef<mlir::Value>;
using AttributeTy = llvm::ArrayRef<mlir::NamedAttribute>;

/// FIR to standard type converter
/// This converts a subset of FIR types to standard types
class FIRToStdTypeConverter : public mlir::TypeConverter {
public:
  using TypeConverter::TypeConverter;

  explicit FIRToStdTypeConverter(KindMapping &kindMap) : kindMap{kindMap} {}

  // convert front-end REAL kind value to a std dialect type, if possible
  static mlir::Type kindToRealType(KindMapping &kindMap, KindTy kind) {
    auto *ctx = kindMap.getContext();
    switch (kindMap.getRealTypeID(kind)) {
    case llvm::Type::TypeID::HalfTyID:
      return mlir::FloatType::getF16(ctx);
#if 0
    // FIXME: there is no BF16 type in LLVM yet
    case llvm::Type::TypeID:: FIXME TyID:
      return mlir::FloatType::getBF16(ctx);
#endif
    case llvm::Type::TypeID::FloatTyID:
      return mlir::FloatType::getF32(ctx);
    case llvm::Type::TypeID::DoubleTyID:
      return mlir::FloatType::getF64(ctx);
    case llvm::Type::TypeID::X86_FP80TyID: // MLIR does not support yet
    case llvm::Type::TypeID::FP128TyID:    // MLIR does not support yet
    default:
      return RealType::get(ctx, kind);
    }
  }

  /// Convert some FIR types to MLIR standard dialect types
  mlir::Type convertType(mlir::Type t) override {
    // To lower types, we have to convert everything that uses these types...
    if (auto cplx = t.dyn_cast<CplxType>())
      return mlir::ComplexType::get(kindToRealType(kindMap, cplx.getFKind()));
    if (auto integer = t.dyn_cast<IntType>())
      return mlir::IntegerType::get(integer.getFKind() * 8,
                                    integer.getContext());
    if (auto real = t.dyn_cast<RealType>())
      return kindToRealType(kindMap, real.getFKind());
    return t;
  }

private:
  KindMapping &kindMap;
};

/// FIR conversion pattern template
template <typename FromOp>
class FIROpConversion : public mlir::ConversionPattern {
public:
  explicit FIROpConversion(
      mlir::MLIRContext *ctx /*, FIRToStdTypeConverter &lowering*/)
      : ConversionPattern(FromOp::getOperationName(), 1,
                          ctx) /*, lowering(lowering)*/
  {}

protected:
  // mlir::Type convertType(mlir::Type ty) const { return
  // lowering.convertType(ty); }

  // FIRToStdTypeConverter &lowering;
};

/// SelectTypeOp converted to an if-then-else chain
///
/// This lowers the test conditions to calls into the runtime
struct SelectTypeOpConversion : public FIROpConversion<SelectTypeOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  llvm::ArrayRef<mlir::Block *> destinations,
                  llvm::ArrayRef<OperandTy> destOperands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto selectType = mlir::cast<SelectTypeOp>(op);
    auto conds = selectType.getNumConditions();
    auto attrName = SelectTypeOp::AttrName;
    auto caseAttr = selectType.getAttrOfType<mlir::ArrayAttr>(attrName);
    auto cases = caseAttr.getValue();
    // Selector must be of type !fir.box<T>
    auto &selector = operands[0];
    auto loc = selectType.getLoc();
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    for (unsigned t = 0; t != conds; ++t) {
      auto &attr = cases[t];
      if (auto a = attr.dyn_cast_or_null<ExactTypeAttr>()) {
        genTypeLadderStep(loc, true, selector, a.getType(), destinations[t],
                          destOperands[t], mod, rewriter);
        continue;
      }
      if (auto a = attr.dyn_cast_or_null<SubclassAttr>()) {
        genTypeLadderStep(loc, false, selector, a.getType(), destinations[t],
                          destOperands[t], mod, rewriter);
        continue;
      }
      assert(attr.dyn_cast_or_null<mlir::UnitAttr>());
      assert((t + 1 == conds) && "unit must be last");
      rewriter.replaceOpWithNewOp<mlir::BranchOp>(
          selectType, destinations[t], mlir::ValueRange{destOperands[t]});
    }
    return matchSuccess();
  }

  static void genTypeLadderStep(mlir::Location loc, bool exactTest,
                                mlir::Value selector, mlir::Type ty,
                                mlir::Block *dest, OperandTy destOps,
                                mlir::ModuleOp module,
                                mlir::ConversionPatternRewriter &rewriter) {
    mlir::Type tydesc = TypeDescType::get(ty);
    auto tyattr = mlir::TypeAttr::get(ty);
    mlir::Value t = rewriter.create<GenTypeDescOp>(loc, tydesc, tyattr);
    mlir::Type selty = BoxType::get(rewriter.getNoneType());
    mlir::Value csel = rewriter.create<ConvertOp>(loc, selty, selector);
    mlir::Type tty = ReferenceType::get(rewriter.getNoneType());
    mlir::Value ct = rewriter.create<ConvertOp>(loc, tty, t);
    std::vector<mlir::Value> actuals = {csel, ct};
    auto fty = rewriter.getI1Type();
    std::vector<mlir::Type> argTy = {selty, tty};
    llvm::StringRef funName =
        exactTest ? "FIXME_exact_type_match" : "FIXME_isa_type_test";
    createFuncOp(rewriter.getUnknownLoc(), module, funName,
                 rewriter.getFunctionType(argTy, fty));
    // FIXME: need to call actual runtime routines for (1) testing if the
    // runtime type of the selector is an exact match to a derived type or (2)
    // testing if the runtime type of the selector is a derived type or one of
    // that derived type's subtypes.
    auto cmp = rewriter.create<mlir::CallOp>(
        loc, fty, rewriter.getSymbolRefAttr(funName), actuals);
    auto *thisBlock = rewriter.getInsertionBlock();
    auto *newBlock = rewriter.createBlock(dest);
    rewriter.setInsertionPointToEnd(thisBlock);
    rewriter.create<mlir::CondBranchOp>(loc, cmp.getResult(0), dest, destOps,
                                        newBlock, OperandTy{});
    rewriter.setInsertionPointToEnd(newBlock);
  }
};

/// Convert affine dialect, fir.select_type to standard dialect
class FIRToStdLoweringPass : public mlir::FunctionPass<FIRToStdLoweringPass> {
public:
  explicit FIRToStdLoweringPass(KindMapping &kindMap) : kindMap{kindMap} {}

  void runOnFunction() override {
    if (ClDisableFirToStd)
      return;

    auto *context{&getContext()};
    // FIRToStdTypeConverter typeConverter{kindMap};
    mlir::OwningRewritePatternList patterns;
    // patterns.insert<SelectTypeOpConversion>(context, typeConverter);
    patterns.insert<SelectTypeOpConversion>(context);
    mlir::populateAffineToStdConversionPatterns(patterns, context);
    // mlir::populateFuncOpTypeConversionPattern(patterns, context,
    // typeConverter);
    mlir::ConversionTarget target{*context};
    target.addLegalDialect<mlir::StandardOpsDialect, FIROpsDialect>();
    // target.addDynamicallyLegalOp<mlir::FuncOp>([&](mlir::FuncOp op) {
    //  return typeConverter.isSignatureLegal(op.getType());
    //});
    target.addIllegalOp<SelectTypeOp>();
    if (mlir::failed(mlir::applyPartialConversion(
            // getModule(), target, std::move(patterns), &typeConverter))) {
            getModule(), target, std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(context),
                      "error in converting to standard dialect\n");
      signalPassFailure();
    }
  }

  mlir::ModuleOp getModule() {
    return getFunction().getParentOfType<mlir::ModuleOp>();
  }

private:
  KindMapping &kindMap;
};
} // namespace

std::unique_ptr<mlir::Pass> createFIRToStdPass(KindMapping &kindMap) {
  return std::make_unique<FIRToStdLoweringPass>(kindMap);
}
} // namespace fir
