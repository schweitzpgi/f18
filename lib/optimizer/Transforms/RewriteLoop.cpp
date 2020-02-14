//===-- RewriteLoop.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/optimizer/Dialect/FIRDialect.h"
#include "flang/optimizer/Dialect/FIROps.h"
#include "flang/optimizer/Transforms/Passes.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/CommandLine.h"
#include <memory>

/// disable FIR to affine dialect conversion
static llvm::cl::opt<bool>
    ClDisableAffinePromo("disable-affine-promotion",
                         llvm::cl::desc("disable FIR to Affine pass"),
                         llvm::cl::init(false));

/// disable FIR to loop dialect conversion
static llvm::cl::opt<bool>
    ClDisableLoopConversion("disable-loop-conversion",
                            llvm::cl::desc("disable FIR to Loop pass"),
                            llvm::cl::init(false));

namespace fir {
namespace {

template <typename FROM>
class OpRewrite : public mlir::RewritePattern {
public:
  explicit OpRewrite(mlir::MLIRContext *ctx)
      : RewritePattern(FROM::getOperationName(), 1, ctx) {}
};

/// Convert `fir.loop` to `affine.for`
class AffineLoopConv : public OpRewrite<LoopOp> {
public:
  using OpRewrite::OpRewrite;
};

/// Convert `fir.where` to `affine.if`
class AffineWhereConv : public OpRewrite<WhereOp> {
public:
  using OpRewrite::OpRewrite;
};

/// Promote fir.loop and fir.where to affine.for and affine.if, in the cases
/// where such a promotion is possible.
class AffineDialectPromotion
    : public mlir::FunctionPass<AffineDialectPromotion> {
public:
  void runOnFunction() override {
    if (ClDisableAffinePromo)
      return;

    auto *context{&getContext()};
    mlir::OwningRewritePatternList patterns;
    patterns.insert<AffineLoopConv, AffineWhereConv>(context);
    mlir::ConversionTarget target{*context};
    target.addLegalDialect<mlir::AffineOpsDialect, FIROpsDialect,
                           mlir::loop::LoopOpsDialect,
                           mlir::StandardOpsDialect>();
    // target.addDynamicallyLegalOp<LoopOp, WhereOp>();

    // apply the patterns
    if (mlir::failed(mlir::applyPartialConversion(getFunction(), target,
                                                  std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(context),
                      "error in converting to affine dialect\n");
      signalPassFailure();
    }
  }
};

// Conversion to the MLIR loop dialect
//
// FIR loops that cannot be converted to the affine dialect will remain as
// `fir.loop` operations.  These can be converted to `loop.for` operations. MLIR
// includes a pass to lower `loop.for` operations to a CFG.

/// Convert `fir.loop` to `loop.for`
class LoopLoopConv : public mlir::OpRewritePattern<LoopOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::PatternMatchResult
  matchAndRewrite(LoopOp loop, mlir::PatternRewriter &rewriter) const override {
    auto low = loop.lowerBound();
    auto high = loop.upperBound();
    auto optStep = loop.optStep();
    auto loc = loop.getLoc();
    mlir::Value step;
    if (optStep.begin() != optStep.end()) {
      step = *optStep.begin();
    } else {
      auto conStep = loop.constantStep();
      step = rewriter.create<mlir::ConstantIndexOp>(
          loc, conStep.hasValue() ? conStep.getValue().getSExtValue() : 1);
    }
    auto f = rewriter.create<mlir::loop::ForOp>(loc, low, high, step);
    f.region().getBlocks().clear();
    rewriter.inlineRegionBefore(loop.region(), f.region(), f.region().end());
    if (loop.hasLastValue()) {
      // Compute the final value of the loop iterator.
      // FIXME: If there are no iterations?
      auto ty = low.getType();
      auto d = rewriter.create<mlir::SubIOp>(loc, ty, high, low);
      auto q = rewriter.create<mlir::SignedDivIOp>(loc, ty, d, step);
      auto dist = rewriter.create<mlir::MulIOp>(loc, ty, q, step);
      rewriter.replaceOpWithNewOp<mlir::AddIOp>(loop, ty, low, dist);
    } else {
      rewriter.eraseOp(loop);
    }
    return matchSuccess();
  }
};

/// Convert `fir.where` to `loop.if`
class LoopWhereConv : public mlir::OpRewritePattern<WhereOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::PatternMatchResult
  matchAndRewrite(WhereOp where,
                  mlir::PatternRewriter &rewriter) const override {
    auto loc = where.getLoc();
    bool hasOtherRegion = !where.otherRegion().empty();
    auto cond = where.condition();
    auto ifOp = rewriter.create<mlir::loop::IfOp>(loc, cond, hasOtherRegion);
    rewriter.inlineRegionBefore(where.whereRegion(), &ifOp.thenRegion().back());
    ifOp.thenRegion().back().erase();
    if (hasOtherRegion) {
      rewriter.inlineRegionBefore(where.otherRegion(),
                                  &ifOp.elseRegion().back());
      ifOp.elseRegion().back().erase();
    }
    rewriter.eraseOp(where);
    return matchSuccess();
  }
};

/// Replace FirEndOp with TerminatorOp
class LoopFirEndConv : public mlir::OpRewritePattern<FirEndOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::PatternMatchResult
  matchAndRewrite(FirEndOp op, mlir::PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::loop::TerminatorOp>(op);
    return matchSuccess();
  }
};

/// Convert `fir.loop` and `fir.where` to `loop.for` and `loop.if`.
class LoopDialectConversion : public mlir::FunctionPass<LoopDialectConversion> {
public:
  void runOnFunction() override {
    if (ClDisableLoopConversion)
      return;

    auto *context{&getContext()};
    mlir::OwningRewritePatternList patterns;
    patterns.insert<LoopLoopConv, LoopWhereConv, LoopFirEndConv>(context);
    mlir::ConversionTarget target{*context};
    target.addLegalDialect<mlir::AffineOpsDialect, FIROpsDialect,
                           mlir::loop::LoopOpsDialect,
                           mlir::StandardOpsDialect>();
    target.addIllegalOp<FirEndOp, LoopOp, WhereOp>();

    // apply the patterns
    if (mlir::failed(mlir::applyPartialConversion(getFunction(), target,
                                                  std::move(patterns)))) {
      mlir::emitError(mlir::UnknownLoc::get(context),
                      "error in converting to MLIR loop dialect\n");
      signalPassFailure();
    }
  }
};
} // namespace

/// Convert FIR loop constructs to the Affine dialect
std::unique_ptr<mlir::Pass> createPromoteToAffinePass() {
  return std::make_unique<AffineDialectPromotion>();
}

/// Convert `fir.loop` and `fir.where` to `loop.for` and `loop.if`.  This
/// conversion enables the `createLowerToCFGPass` to transform these to CFG
/// form.
std::unique_ptr<mlir::Pass> createLowerToLoopPass() {
  return std::make_unique<LoopDialectConversion>();
}
} // namespace fir
