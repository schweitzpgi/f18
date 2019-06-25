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

#include "fe-helper.h"
#include "fir-type.h"
#include "../semantics/expression.h"
#include "../semantics/tools.h"
#include "../semantics/type.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/StandardTypes.h"

namespace Br = Fortran::mlbridge;
namespace Co = Fortran::common;
namespace Ev = Fortran::evaluate;
namespace M = mlir;
namespace Pa = Fortran::parser;
namespace Se = Fortran::semantics;

using namespace Fortran;
using namespace Fortran::mlbridge;

namespace {

template<typename A> bool isConstant(const Ev::Expr<A> &e) {
  return Ev::IsConstantExpr(SomeExpr{e});
}

template<typename A> int64_t toConstant(const Ev::Expr<A> &e) {
  auto opt = Ev::ToInt64(e);
  assert(opt.has_value() && "expression didn't resolve to a constant");
  return opt.value();
}

#undef TODO
#define TODO() assert(false)

int defaultRealKind(Se::SemanticsContext &sc) {
  return Ev::ExpressionAnalyzer{sc}.GetDefaultKind(RealCat);
}

int defaultIntegerKind(Se::SemanticsContext &sc) {
  return Ev::ExpressionAnalyzer{sc}.GetDefaultKind(IntegerCat);
}

int defaultCharKind(Se::SemanticsContext &sc) {
  return Ev::ExpressionAnalyzer{sc}.GetDefaultKind(CharacterCat);
}

int defaultLogicalKind(Se::SemanticsContext &sc) {
  return Ev::ExpressionAnalyzer{sc}.GetDefaultKind(LogicalCat);
}

int defaultIntegerBits(Se::SemanticsContext &sc) {
  return defaultIntegerKind(sc) * 8;
}

/// Recover the type of an evaluate::Expr<T> and convert it to an
/// mlir::Type. The type returned can be a MLIR standard or FIR type.
class TypeBuilder {
  M::MLIRContext *context;
  Se::SemanticsContext &semanticsContext;

  int getDefaultRealKind() const { return defaultRealKind(semanticsContext); }

  int getDefaultIntegerBits() const {
    return defaultIntegerBits(semanticsContext);
  }

public:
  explicit TypeBuilder(M::MLIRContext *context, Se::SemanticsContext &sc)
    : context{context}, semanticsContext{sc} {}

  /// Create a FIR/MLIR real type
  template<int KIND> static M::Type genReal(M::MLIRContext *context) {
    if constexpr (KIND == 2) {
      return M::FloatType::getF16(context);
    } else if constexpr (KIND == 3) {
      return M::FloatType::getBF16(context);
    } else if constexpr (KIND == 4) {
      return M::FloatType::getF32(context);
    } else if constexpr (KIND == 8) {
      return M::FloatType::getF64(context);
    }
    return FIRRealType::get(context, KIND);
  }

  template<int KIND> M::Type genReal() { return genReal<KIND>(context); }

  /// Create a FIR/MLIR real type from information at runtime
  static M::Type genReal(int kind, M::MLIRContext *context) {
    switch (kind) {
    case 2: return genReal<2>(context);
    case 3: return genReal<3>(context);
    case 4: return genReal<4>(context);
    case 8: return genReal<8>(context);
    default: return FIRRealType::get(context, kind);
    }
  }

  M::Type genReal(int kind) { return genReal(kind, context); }

  static M::Type genType(M::MLIRContext *ctxt, Co::TypeCategory tc, int kind) {
    switch (tc) {
    case IntegerCat: return M::IntegerType::get(kind * 8, ctxt);
    case RealCat: return genReal(kind, ctxt);
    case ComplexCat: return M::ComplexType::get(genReal(kind, ctxt));
    case CharacterCat: return FIRCharacterType::get(ctxt, kind);
    case LogicalCat: return FIRLogicalType::get(ctxt, kind);
    default: break;
    }
    assert(false && "unhandled type category");
    return {};
  }

  static M::Type genType(
      M::MLIRContext *ctxt, Se::SemanticsContext &sc, Co::TypeCategory tc) {
    switch (tc) {
    case IntegerCat: return M::IntegerType::get(defaultIntegerBits(sc), ctxt);
    case RealCat: return genType(ctxt, tc, defaultRealKind(sc));
    case ComplexCat: return genType(ctxt, tc, defaultRealKind(sc));
    case CharacterCat: return genType(ctxt, tc, defaultCharKind(sc));
    case LogicalCat: return genType(ctxt, tc, defaultLogicalKind(sc));
    case DerivedCat: return genType(ctxt, tc, 0);
    default: break;
    }
    assert(false && "unknown type category");
    return {};
  }

  M::Type gen(const Ev::ImpliedDoIndex &) {
    return genType(context, semanticsContext, IntegerCat);
  }

  template<template<typename> typename A, Co::TypeCategory TC>
  M::Type gen(const A<Ev::SomeKind<TC>> &) {
    return genType(context, semanticsContext, TC);
  }

  template<int KIND> M::Type gen(const Ev::TypeParamInquiry<KIND> &) {
    return genType(context, IntegerCat, KIND);
  }

  template<typename A> M::Type gen(const Ev::Relational<A> &) {
    return FIRLogicalType::get(context, 1);
  }

  template<template<typename> typename A, Co::TypeCategory TC, int KIND>
  M::Type gen(const A<Ev::Type<TC, KIND>> &) {
    return genType(context, TC, KIND);
  }

  // breaks the conflict between A<Type<TC,KIND>> and Expr<B> deduction
  template<Co::TypeCategory TC, int KIND>
  M::Type gen(const Ev::Expr<Ev::Type<TC, KIND>> &) {
    return genType(context, TC, KIND);
  }

  template<typename A> M::Type genVariant(const A &variant) {
    return std::visit([&](const auto &x) { return gen(x); }, variant.u);
  }

  // breaks the conflict between A<SomeKind<TC>> and Expr<B> deduction
  template<Co::TypeCategory TC>
  M::Type gen(const Ev::Expr<Ev::SomeKind<TC>> &expr) {
    return genVariant(expr);
  }

  template<typename A> M::Type gen(const Ev::Expr<A> &expr) {
    return genVariant(expr);
  }

  M::Type gen(const Ev::DataRef &dref) { return genVariant(dref); }

  M::Type mkVoid() { return M::TupleType::get(context); }

  FIRSequenceType::Shape genSeqShape(const Se::Symbol *symbol) {
    assert(symbol->IsObjectArray());
    FIRSequenceType::Bounds bounds;
    auto &details = symbol->get<Se::ObjectEntityDetails>();
    const auto size = details.shape().size();
    for (auto &ss : details.shape()) {
      auto lb = ss.lbound();
      auto ub = ss.ubound();
      if (lb.isAssumed() && ub.isAssumed() && size == 1) {
        return {FIRSequenceType::Unknown{}};
      }
      if (lb.isExplicit() && ub.isExplicit()) {
        auto &lbv = lb.GetExplicit();
        auto &ubv = ub.GetExplicit();
        if (lbv.has_value() && ubv.has_value() && isConstant(lbv.value()) &&
            isConstant(ubv.value())) {
          bounds.emplace_back(FIRSequenceType::BoundInfo{
              toConstant(lbv.value()), toConstant(ubv.value()), 1});
        } else {
          bounds.emplace_back(FIRSequenceType::Unknown{});
        }
      } else {
        bounds.emplace_back(FIRSequenceType::Unknown{});
      }
    }
    return {bounds};
  }

  /// Type consing from a symbol. A symbol's type must be created from the type
  /// discovered by the front-end at runtime.
  M::Type gen(const Se::Symbol *symbol) {
    if (auto *proc = symbol->detailsIf<Se::SubprogramDetails>()) {
      M::Type returnTy{mkVoid()};
      if (proc->isFunction()) {
        returnTy = gen(&proc->result());
      }
      // FIXME: handle alt-return
      llvm::SmallVector<M::Type, 4> inputTys;
      for (auto *arg : proc->dummyArgs()) {
        // FIXME: not all args are pass by ref
        inputTys.emplace_back(FIRReferenceType::get(gen(arg)));
      }
      return M::FunctionType::get(inputTys, returnTy, context);
    }
    M::Type returnTy{};
    if (auto *type{symbol->GetType()}) {
      if (auto *tySpec{type->AsIntrinsic()}) {
        int kind = toConstant(tySpec->kind());
        switch (tySpec->category()) {
        case IntegerCat:
          returnTy = M::IntegerType::get(kind * 8, context);
          break;
        case RealCat: {
          returnTy = genReal(kind);
        } break;
        case ComplexCat: returnTy = M::ComplexType::get(genReal(kind)); break;
        case CharacterCat:
          returnTy = FIRCharacterType::get(context, kind);
          break;
        case LogicalCat: returnTy = FIRLogicalType::get(context, kind); break;
        case DerivedCat: {
          TODO();
        } break;
        }
      }
    }
    if (symbol->IsObjectArray()) {
      // FIXME: add bounds info
      returnTy = FIRSequenceType::get(genSeqShape(symbol), returnTy);
    } else if (Se::IsPointer(*symbol)) {
      // FIXME: what about allocatable?
      returnTy = FIRReferenceType::get(returnTy);
    }
    return returnTy;
  }

  FIRSequenceType::Shape trivialShape(int size) {
    FIRSequenceType::Bounds bounds;
    bounds.push_back(FIRSequenceType::BoundInfo{1, size, 1});
    return {bounds};
  }

  // some sequence of `n` bytes
  M::Type gen(const Ev::StaticDataObject::Pointer &ptr) {
    M::Type byteTy{M::IntegerType::get(8, context)};
    return FIRSequenceType::get(trivialShape(ptr->itemBytes()), byteTy);
  }

  M::Type gen(const Ev::Substring &ss) {
    return genVariant(ss.GetBaseObject());
  }

  M::Type genTypelessPtr() { return FIRReferenceType::get(mkVoid()); }
  M::Type gen(const Ev::NullPointer &) { return genTypelessPtr(); }
  M::Type gen(const Ev::ProcedureRef &) { return genTypelessPtr(); }
  M::Type gen(const Ev::ProcedureDesignator &) { return genTypelessPtr(); }
  M::Type gen(const Ev::BOZLiteralConstant &) { return genTypelessPtr(); }

  M::Type gen(const Ev::ArrayRef &) { TODO(); }
  M::Type gen(const Ev::CoarrayRef &) { TODO(); }
  M::Type gen(const Ev::Component &) { TODO(); }
  M::Type gen(const Ev::ComplexPart &) { TODO(); }
  M::Type gen(const Ev::DescriptorInquiry &) { TODO(); }
  M::Type gen(const Ev::StructureConstructor &) { TODO(); }
};

}  // namespace

/// Generate an unknown location
M::Location Br::dummyLoc(M::MLIRContext *ctxt) {
  return M::UnknownLoc::get(ctxt);
}

// What do we need to convert a CharBlock to actual source locations?
// FIXME: replace with a map from a provenance to a source location
M::Location Br::parserPosToLoc(
    M::MLIRContext &context, const Pa::CharBlock &position) {
  return dummyLoc(&context);
}

M::Type Br::genTypeFromCategoryAndKind(
    M::MLIRContext *ctxt, Co::TypeCategory tc, int kind) {
  return TypeBuilder::genType(ctxt, tc, kind);
}

M::Type Br::genTypeFromCategory(
    M::MLIRContext *ctxt, Se::SemanticsContext &sc, Co::TypeCategory tc) {
  return TypeBuilder::genType(ctxt, sc, tc);
}

M::Type Br::translateDataRefToFIRType(M::MLIRContext *context,
    Se::SemanticsContext &semanticsContext, const Ev::DataRef &dataRef) {
  return TypeBuilder{context, semanticsContext}.gen(dataRef);
}

// Builds the FIR type from an instance of SomeExpr
M::Type Br::translateSomeExprToFIRType(M::MLIRContext *context,
    Se::SemanticsContext &semanticsContext, const SomeExpr *expr) {
  return TypeBuilder{context, semanticsContext}.gen(*expr);
}

// This entry point avoids gratuitously wrapping the Symbol instance in layers
// of Expr<T> that will then be immediately peeled back off and discarded.
M::Type Br::translateSymbolToFIRType(M::MLIRContext *context,
    Se::SemanticsContext &semanticsContext, const Se::Symbol *symbol) {
  return TypeBuilder{context, semanticsContext}.gen(symbol);
}

M::Type Br::convertReal(int KIND, M::MLIRContext *context) {
  return TypeBuilder::genReal(KIND, context);
}
