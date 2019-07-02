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
#include "builder.h"
#include "expression.h"
#include "fe-helper.h"
#include "fir-dialect.h"
#include "fir-type.h"
#include "runtime.h"
#include "../evaluate/decimal.h"
#include "../evaluate/fold.h"
#include "../evaluate/real.h"
#include "../semantics/expression.h"
#include "../semantics/symbol.h"
#include "../semantics/type.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/AffineOps/AffineOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/StandardOps/Ops.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

namespace Br = Fortran::mlbridge;
namespace Co = Fortran::common;
namespace Ev = Fortran::evaluate;
namespace L = llvm;
namespace M = mlir;
namespace Pa = Fortran::parser;
namespace Se = Fortran::semantics;

using namespace Fortran;
using namespace Fortran::mlbridge;

namespace {

#define TODO() \
  assert(false); \
  return {}

inline M::Location dummyLocation(M::OpBuilder *builder) {
  return M::UnknownLoc::get(builder->getContext());
}

/// Lowering of Fortran::evaluate::Expr<T> expressions
class ExprLowering {
  M::OpBuilder *builder;
  L::SmallVector<M::Value *, 8> operands;
  M::Operation *op;
  const SomeExpr *expr;
  L::DenseMap<void *, unsigned> operMap;
  bool coordinate{false};

  // FIXME: how do we map an evaluate::Expr<T> to a source location?
  M::Location dummyLoc() { return dummyLocation(builder); }

  template<typename A> void initializeMap(A &e) {
    auto &dict{*e.getDict()};
    auto sz{dict.size()};
    assert(sz == operands.size() && "operands and dict differ in size");
    for (decltype(sz) i{0}; i != sz; ++i) {
      operMap[dict[i]] = i;
    }
  }

  /// Convert parser's INTEGER relational operators to MLIR.  TODO: using
  /// unordered, but we may want to cons ordered in certain situation.
  static M::CmpIPredicate translateRelational(Co::RelationalOperator rop) {
    switch (rop) {
    case Co::RelationalOperator::LT: return M::CmpIPredicate::SLT;
    case Co::RelationalOperator::LE: return M::CmpIPredicate::SLE;
    case Co::RelationalOperator::EQ: return M::CmpIPredicate::EQ;
    case Co::RelationalOperator::NE: return M::CmpIPredicate::NE;
    case Co::RelationalOperator::GT: return M::CmpIPredicate::SGT;
    case Co::RelationalOperator::GE: return M::CmpIPredicate::SGE;
    }
    assert(false && "unhandled INTEGER relational operator");
    return {};
  }

  /// Convert parser's REAL relational operators to MLIR.  TODO: using
  /// unordered, but we may want to cons ordered in certain situation.
  static M::CmpFPredicate translateFloatRelational(Co::RelationalOperator rop) {
    switch (rop) {
    case Co::RelationalOperator::LT: return M::CmpFPredicate::ULT;
    case Co::RelationalOperator::LE: return M::CmpFPredicate::ULE;
    case Co::RelationalOperator::EQ: return M::CmpFPredicate::UEQ;
    case Co::RelationalOperator::NE: return M::CmpFPredicate::UNE;
    case Co::RelationalOperator::GT: return M::CmpFPredicate::UGT;
    case Co::RelationalOperator::GE: return M::CmpFPredicate::UGE;
    }
    assert(false && "unhandled REAL relational operator");
    return {};
  }

  /// Generate an integral constant of `value`
  template<int KIND>
  RewriteVals genIntegerConstant(M::MLIRContext *context, std::int64_t value) {
    M::Type type{M::IntegerType::get(KIND * 8, context)};
    auto attr{builder->getIntegerAttr(type, value)};
    auto res{builder->create<M::ConstantOp>(dummyLoc(), type, attr)};
    return res.getResult();
  }

  /// Generate a logical/boolean constant of `value`
  template<int KIND>
  RewriteVals genLogicalConstant(M::MLIRContext *context, bool value) {
    auto attr{builder->getBoolAttr(value)};
    M::Type logTy{FIRLogicalType::get(context, KIND)};
    auto res{builder->create<M::ConstantOp>(dummyLoc(), logTy, attr)};
    return res.getResult();
  }

  template<int KIND>
  RewriteVals genRealConstant(
      M::MLIRContext *context, const L::APFloat &value) {
    M::Type fltTy{convertReal(KIND, context)};
    auto attr{builder->getFloatAttr(fltTy, value)};
    auto res{builder->create<M::ConstantOp>(dummyLoc(), fltTy, attr)};
    return res.getResult();
  }

  M::Type getSomeKindInteger() {
    return M::IntegerType::get(SomeKindIntegerBits, builder->getContext());
  }

  template<typename OpTy, typename A>
  RewriteVals createBinaryOp(const A &ex, RewriteVals lhs, RewriteVals rhs) {
    assert(lhs && rhs && "argument did not lower");
    auto x = builder->create<OpTy>(dummyLoc(), lhs, rhs);
    return x.getResult();
  }
  template<typename OpTy, typename A>
  RewriteVals createBinaryOp(const A &ex, RewriteVals rhs) {
    return createBinaryOp<OpTy>(ex, genval(ex.left()), rhs);
  }
  template<typename OpTy, typename A> RewriteVals createBinaryOp(const A &ex) {
    return createBinaryOp<OpTy>(ex, genval(ex.left()), genval(ex.right()));
  }
  M::Function *getFunction(RuntimeEntryCode callee, M::FunctionType funTy) {
    auto name{getRuntimeEntryName(callee)};
    auto *module{builder->getBlock()->getFunction()->getModule()};
    if (auto *func{module->getNamedFunction(name)}) {
      return func;
    }
    return createFunction(module, name, funTy);
  }

  // FIXME binary operation :: ('a, 'a) -> 'a
  template<Co::TypeCategory TC, int KIND> M::FunctionType createFunctionType() {
    if constexpr (TC == IntegerCat) {
      M::Type output{M::IntegerType::get(KIND, builder->getContext())};
      L::SmallVector<M::Type, 2> inputs;
      inputs.push_back(output);
      inputs.push_back(output);
      return M::FunctionType::get(inputs, output, builder->getContext());
    } else if constexpr (TC == RealCat) {
      M::Type output{convertReal(KIND, builder->getContext())};
      L::SmallVector<M::Type, 2> inputs;
      inputs.push_back(output);
      inputs.push_back(output);
      return M::FunctionType::get(inputs, output, builder->getContext());
    } else {
      assert(false);
      return {};
    }
  }

  /// Create a call to a Fortran runtime entry point
  template<Co::TypeCategory TC, int KIND, typename A>
  RewriteVals createBinaryFIRTCall(const A &ex, RuntimeEntryCode callee) {
    L::SmallVector<M::Value *, 2> operands;
    operands.push_back(genval(ex.left()));
    operands.push_back(genval(ex.right()));
    M::FunctionType funTy = createFunctionType<TC, KIND>();
    auto *func{getFunction(callee, funTy)};
    auto x{builder->create<M::CallOp>(dummyLoc(), func, operands)};
    return x.getResult(0);  // FIXME
  }

  template<typename OpTy, typename A>
  RewriteVals createCompareOp(
      const A &ex, M::CmpIPredicate pred, RewriteVals lhs, RewriteVals rhs) {
    assert(lhs && rhs && "argument did not lower");
    auto x = builder->create<OpTy>(dummyLoc(), pred, lhs, rhs);
    return x.getResult();
  }
  template<typename OpTy, typename A>
  RewriteVals createCompareOp(const A &ex, M::CmpIPredicate pred) {
    return createCompareOp<OpTy>(
        ex, pred, genval(ex.left()), genval(ex.right()));
  }
  template<typename OpTy, typename A>
  RewriteVals createFltCmpOp(
      const A &ex, M::CmpFPredicate pred, RewriteVals lhs, RewriteVals rhs) {
    assert(lhs && rhs && "argument did not lower");
    auto x = builder->create<OpTy>(dummyLoc(), pred, lhs, rhs);
    return x.getResult();
  }
  template<typename OpTy, typename A>
  RewriteVals createFltCmpOp(const A &ex, M::CmpFPredicate pred) {
    return createFltCmpOp<OpTy>(
        ex, pred, genval(ex.left()), genval(ex.right()));
  }

  // always return the value rather than (possibly) a coordinate
  template<typename A> RewriteVals genval(const A &e) {
    coordinate = false;
    RewriteVals v{gen(e)};
    if (coordinate) {
      v = builder->create<LoadExpr>(dummyLoc(), v);
    }
    coordinate = false;
    return v;
  }

  RewriteVals gen(const Ev::BOZLiteralConstant &) { TODO(); }
  RewriteVals gen(const Ev::ProcedureRef &) { TODO(); }
  RewriteVals gen(const Ev::ProcedureDesignator &) { TODO(); }
  RewriteVals gen(const Ev::NullPointer &) { TODO(); }
  RewriteVals gen(const Ev::StructureConstructor &) { TODO(); }
  RewriteVals gen(const Ev::ImpliedDoIndex &) { TODO(); }
  RewriteVals gen(const Ev::DescriptorInquiry &) { TODO(); }
  template<int KIND> RewriteVals gen(const Ev::TypeParamInquiry<KIND> &) {
    TODO();
  }
  template<int KIND> RewriteVals gen(const Ev::ComplexComponent<KIND> &) {
    TODO();
  }
  template<Co::TypeCategory TC, int KIND>
  RewriteVals gen(const Ev::Negate<Ev::Type<TC, KIND>> &) {
    TODO();
  }
  template<Co::TypeCategory TC, int KIND>
  RewriteVals gen(const Ev::Add<Ev::Type<TC, KIND>> &op) {
    if constexpr (TC == IntegerCat) {
      return createBinaryOp<M::AddIOp>(op);
    } else if constexpr (TC == RealCat) {
      return createBinaryOp<M::AddFOp>(op);
    } else {
      TODO();
    }
  }
  template<Co::TypeCategory TC, int KIND>
  RewriteVals gen(const Ev::Subtract<Ev::Type<TC, KIND>> &op) {
    if constexpr (TC == IntegerCat) {
      return createBinaryOp<M::SubIOp>(op);
    } else if constexpr (TC == RealCat) {
      return createBinaryOp<M::SubFOp>(op);
    } else {
      TODO();
    }
  }
  template<Co::TypeCategory TC, int KIND>
  RewriteVals gen(const Ev::Multiply<Ev::Type<TC, KIND>> &op) {
    if constexpr (TC == IntegerCat) {
      return createBinaryOp<M::MulIOp>(op);
    } else if constexpr (TC == RealCat) {
      return createBinaryOp<M::MulFOp>(op);
    } else {
      TODO();
    }
  }
  template<Co::TypeCategory TC, int KIND>
  RewriteVals gen(const Ev::Divide<Ev::Type<TC, KIND>> &op) {
    if constexpr (TC == IntegerCat) {
      return createBinaryOp<M::DivISOp>(op);
    } else if constexpr (TC == RealCat) {
      return createBinaryOp<M::DivFOp>(op);
    } else if constexpr (TC == ComplexCat) {
      return createBinaryFIRTCall<TC, KIND>(op, FIRT_CDIV);
    } else {
      TODO();
    }
  }
  template<Co::TypeCategory TC, int KIND>
  RewriteVals gen(const Ev::Power<Ev::Type<TC, KIND>> &op) {
    if constexpr (TC == IntegerCat) {
      return createBinaryFIRTCall<TC, KIND>(op, FIRT_POW);
    } else {
      TODO();
    }
  }
  template<Co::TypeCategory TC, int KIND>
  RewriteVals gen(const Ev::RealToIntPower<Ev::Type<TC, KIND>> &) {
    TODO();
  }
  template<int KIND> RewriteVals gen(const Ev::ComplexConstructor<KIND> &) {
    TODO();
  }
  template<int KIND> RewriteVals gen(const Ev::Concat<KIND> &op) {
    return createBinaryFIRTCall<CharacterCat, KIND>(op, FIRT_CONCAT);
  }

  /// MIN and MAX operations
  template<Co::TypeCategory TC, int KIND>
  RewriteVals gen(const Ev::Extremum<Ev::Type<TC, KIND>> &op) {
    if constexpr (TC == IntegerCat) {
      return createBinaryFIRTCall<TC, KIND>(
          op, op.ordering == Ev::Ordering::Greater ? FIRT_MAX : FIRT_MIN);
    } else {
      TODO();
    }
  }

  template<int KIND> RewriteVals gen(const Ev::SetLength<KIND> &) { TODO(); }

  template<Co::TypeCategory TC, int KIND>
  RewriteVals gen(const Ev::Relational<Ev::Type<TC, KIND>> &op) {
    if constexpr (TC == IntegerCat) {
      return createCompareOp<M::CmpIOp>(op, translateRelational(op.opr));
    } else if constexpr (TC == RealCat) {
      return createFltCmpOp<M::CmpFOp>(op, translateFloatRelational(op.opr));
    } else {
      TODO();
    }
  }
  RewriteVals gen(const Ev::Relational<Ev::SomeType> &op) {
    return std::visit([&](const auto &x) { return genval(x); }, op.u);
  }

  template<Co::TypeCategory TC1, int KIND, Co::TypeCategory TC2>
  RewriteVals gen(const Ev::Convert<Ev::Type<TC1, KIND>, TC2> &convert) {
    auto ty{genTypeFromCategoryAndKind(builder->getContext(), TC1, KIND)};
    return builder->create<ConvertOp>(dummyLoc(), genval(convert.left()), ty);
  }
  template<typename A> RewriteVals gen(const Ev::Parentheses<A> &) { TODO(); }
  template<int KIND> RewriteVals gen(const Ev::Not<KIND> &op) {
    auto *context{builder->getContext()};
    return createBinaryOp<M::XOrOp>(op, genLogicalConstant<KIND>(context, 1));
  }

  template<int KIND> RewriteVals gen(const Ev::LogicalOperation<KIND> &op) {
    switch (op.logicalOperator) {
    case Ev::LogicalOperator::And: return createBinaryOp<M::AndOp>(op);
    case Ev::LogicalOperator::Or: return createBinaryOp<M::OrOp>(op);
    case Ev::LogicalOperator::Eqv:
      return createCompareOp<M::CmpIOp>(op, M::CmpIPredicate::EQ);
    case Ev::LogicalOperator::Neqv:
      return createCompareOp<M::CmpIOp>(op, M::CmpIPredicate::NE);
    }
    assert(false && "unhandled logical operation");
    return {};
  }

  template<Co::TypeCategory TC, int KIND>
  RewriteVals gen(const Ev::Constant<Ev::Type<TC, KIND>> &con) {
    if constexpr (TC == IntegerCat) {
      auto opt{con.GetScalarValue()};
      if (opt.has_value())
        return genIntegerConstant<KIND>(builder->getContext(), opt->ToInt64());
      assert(false && "integer constant has no value");
      return {};
    } else if constexpr (TC == LogicalCat) {
      auto opt{con.GetScalarValue()};
      if (opt.has_value())
        return genLogicalConstant<KIND>(builder->getContext(), opt->IsTrue());
      assert(false && "logical constant has no value");
      return {};
    } else if constexpr (TC == RealCat) {
      auto opt{con.GetScalarValue()};
      if (opt.has_value()) {
        std::string str{opt.value().DumpHexadecimal()};
        if constexpr (KIND == 2) {
          L::APFloat floatVal{L::APFloatBase::IEEEhalf(), str};
          return genRealConstant<KIND>(builder->getContext(), floatVal);
        } else if constexpr (KIND == 4) {
          L::APFloat floatVal{L::APFloatBase::IEEEsingle(), str};
          return genRealConstant<KIND>(builder->getContext(), floatVal);
        } else if constexpr (KIND == 10) {
          L::APFloat floatVal{L::APFloatBase::x87DoubleExtended(), str};
          return genRealConstant<KIND>(builder->getContext(), floatVal);
        } else if constexpr (KIND == 16) {
          L::APFloat floatVal{L::APFloatBase::IEEEquad(), str};
          return genRealConstant<KIND>(builder->getContext(), floatVal);
        } else {
          // convert everything else to double
          L::APFloat floatVal{L::APFloatBase::IEEEdouble(), str};
          return genRealConstant<KIND>(builder->getContext(), floatVal);
        }
      }
      assert(false && "real constant has no value");
      return {};
    } else {
      assert(false && "unhandled constant");
      return {};
    }
  }

  template<Co::TypeCategory TC>
  RewriteVals gen(const Ev::Constant<Ev::SomeKind<TC>> &con) {
    if constexpr (TC == IntegerCat) {
      auto opt = (*con).ToInt64();
      M::Type type{getSomeKindInteger()};
      auto attr{builder->getIntegerAttr(type, opt)};
      auto res{builder->create<M::ConstantOp>(dummyLoc(), type, attr)};
      return res.getResult();
    } else {
      assert(false && "unhandled constant of unknown kind");
      return {};
    }
  }

  // Lookup a data-ref by its key value.  The `key` must be in the map
  // `operands`.
  RewriteVals genKey(void *key) {
    auto iter{operMap.find(key)};
    assert(iter != operMap.end() && "key not in dictionary");
    return operands[iter->second];
  }

  template<typename A> RewriteVals gen(const Ev::ArrayConstructor<A> &) {
    TODO();
  }
  RewriteVals gen(const Ev::ComplexPart &) { TODO(); }
  RewriteVals gen(const Ev::Substring &) { TODO(); }
  RewriteVals gen(const Ev::Triplet &trip) { TODO(); }

  RewriteVals gen(const Ev::Subscript &subs) {
    return std::visit(Co::visitors{
                          [&](const Ev::IndirectSubscriptIntegerExpr &x) {
                            return genval(x.value());
                          },
                          [&](const Ev::Triplet &x) { return genval(x); },
                      },
        subs.u);
  }

  RewriteVals gen(const Ev::DataRef &dref) {
    return std::visit(Co::visitors{
                          [&](const Se::Symbol *x) {
                            return genKey(const_cast<Se::Symbol *>(x));
                          },
                          [&](const auto &x) { return gen(x); },
                      },
        dref.u);
  }

  // Helper function to turn the left-recursive Component structure into a list.
  // Returns the object used as the base coordinate for the component chain.
  static const Ev::DataRef *reverseComponents(
      const Ev::Component &cmpt, std::list<const Ev::Component *> &list) {
    list.push_front(&cmpt);
    return std::visit(
        Co::visitors{
            [&](const Ev::Component &x) { return reverseComponents(x, list); },
            [&](auto &) { return &cmpt.base(); },
        },
        cmpt.base().u);
  }

  // Return the coordinate of the component reference
  RewriteVals gen(const Ev::Component &cmpt) {
    std::list<const Ev::Component *> list;
    auto *base{reverseComponents(cmpt, list)};
    L::SmallVector<M::Value *, 2> coorArgs;
    coorArgs.push_back(gen(*base));
    const Se::Symbol *sym{nullptr};
    for (auto *field : list) {
      sym = &field->GetLastSymbol();
      auto name{sym->name().ToString()};
      coorArgs.push_back(builder->create<FieldValueOp>(dummyLoc(), name));
    }
    assert(sym && "no component(s)?");
    M::Type ty{translateSymbolToFIRType(builder->getContext(), sym)};
    coordinate = true;
    return builder->create<CoordinateOp>(dummyLoc(), coorArgs, ty);
  }

  // Determine the result type after removing `dims` dimensions from the array
  // type `arrTy`
  M::Type genSubType(M::Type arrTy, unsigned dims) {
    if (auto memRef{arrTy.dyn_cast<M::MemRefType>()}) {
      if (dims < memRef.getRank()) {
        auto shape{memRef.getShape()};
        llvm::SmallVector<int64_t, 4> newShape;
        // TODO: should we really remove rows here?
        for (unsigned i = dims, e = memRef.getRank(); i < e; ++i) {
          newShape.push_back(shape[i]);
        }
        return M::MemRefType::get(newShape, memRef.getElementType());
      }
      return memRef.getElementType();
    }
    auto unwrapTy{arrTy.cast<FIRReferenceType>().getEleTy()};
    auto seqTy{unwrapTy.cast<FIRSequenceType>()};
    return std::visit(
        Co::visitors{
            [&](const FIRSequenceType::Unknown &) { return seqTy.getEleTy(); },
            [&](const FIRSequenceType::Bounds &bnds) -> M::Type {
              if (dims < bnds.size()) {
                FIRSequenceType::Bounds newBnds;
                // follow Fortran semantics and remove columns
                for (unsigned i = 0; i < dims; ++i) {
                  newBnds.push_back(bnds[i]);
                }
                return FIRSequenceType::get({newBnds}, seqTy.getEleTy());
              }
              return seqTy.getEleTy();
            },
        },
        seqTy.getShape());
  }

  // Return the coordinate of the array reference
  RewriteVals gen(const Ev::ArrayRef &aref) {
    RewriteVals base;
    if (aref.base().IsSymbol()) {
      base = genKey(const_cast<Se::Symbol *>(&aref.base().GetFirstSymbol()));
    } else {
      base = gen(aref.base().GetComponent());
    }
    llvm::SmallVector<M::Value *, 8> args;
    args.push_back(base);
    for (auto &subsc : aref.subscript()) {
      coordinate = false;
      auto *sub{genval(subsc)};
      if (coordinate) {
        // promote array subscript to a value
        sub = builder->create<LoadExpr>(dummyLoc(), sub);
      }
      args.push_back(sub);
    }
    auto subTy{genSubType(base->getType(), args.size() - 1)};
    coordinate = true;
    return builder->create<CoordinateOp>(dummyLoc(), args, subTy);
  }

  // Return a coordinate of the coarray reference. This is necessary as a
  // Component may have a CoarrayRef as its base coordinate.
  RewriteVals gen(const Ev::CoarrayRef &coref) {
    // FIXME: need to visit the cosubscripts...
    // return gen(coref.base());
    coordinate = true;
    TODO();
  }
  template<typename A> RewriteVals gen(const Ev::Designator<A> &des) {
    return std::visit(Co::visitors{
                          [&](const Se::Symbol *x) {
                            return genKey(const_cast<Se::Symbol *>(x));
                          },
                          [&](const auto &x) { return gen(x); },
                      },
        des.u);
  }

  // lookup call in the map
  template<typename A> RewriteVals gen(const Ev::FunctionRef<A> &funRef) {
    return genKey(const_cast<Ev::FunctionRef<A> *>(&funRef));
  }

  template<typename A> RewriteVals gen(const Ev::Expr<A> &exp) {
    return std::visit([&](const auto &e) { return gen(e); }, exp.u);
  }

public:
  template<typename A>
  explicit ExprLowering(M::OpBuilder *bldr, OperandTy operands, A &vop)
    : builder{bldr}, operands{operands.begin(), operands.end()},
      op{vop.getOperation()}, expr{vop.getRawExpr()} {
    initializeMap(vop);
  }

  /// Lower the expression `expr` into MLIR standard dialect
  RewriteVals gen() { return gen(*expr); }
  RewriteVals genval() { return genval(*expr); }
};

/// Transform an `apply_expr` operation into its fundamental operations
class ApplyExprLower : public M::ConversionPattern {
public:
  explicit ApplyExprLower(M::MLIRContext *ctxt)
    : ConversionPattern(ApplyExpr::getOperationName(), 1, ctxt) {}

  M::PatternMatchResult matchAndRewrite(M::Operation *op, OperandTy operands,
      M::PatternRewriter &rewriter) const override {
    auto apply = M::cast<ApplyExpr>(op);
    rewriter.replaceOp(op, lowerSomeExpr(&rewriter, operands, apply));
    return matchSuccess();
  }
};

/// Transform a `locate_expr` operation into its fundamental operations
class LocateExprLower : public M::ConversionPattern {
public:
  explicit LocateExprLower(M::MLIRContext *ctxt)
    : ConversionPattern(LocateExpr::getOperationName(), 1, ctxt) {}

  M::PatternMatchResult matchAndRewrite(M::Operation *op, OperandTy operands,
      M::PatternRewriter &rewriter) const override {
    auto loc = M::cast<LocateExpr>(op);
    rewriter.replaceOp(op, lowerSomeExpr(&rewriter, operands, loc));
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
    // everything except ApplyExpr and LocateExpr
    target.addLegalOp<AllocaExpr, AllocMemOp, ConvertOp, CoordinateOp,
        ExtractValueOp, FreeMemOp, GlobalExpr, InsertValueOp, LoadExpr,
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

RewriteVals Br::lowerSomeExpr(
    M::OpBuilder *bldr, OperandTy opnds, ApplyExpr &op) {
  ExprLowering lower{bldr, opnds, op};
  return lower.genval();
}

RewriteVals Br::lowerSomeExpr(
    M::OpBuilder *bldr, OperandTy opnds, LocateExpr &op) {
  ExprLowering lower{bldr, opnds, op};
  return lower.gen();
}
