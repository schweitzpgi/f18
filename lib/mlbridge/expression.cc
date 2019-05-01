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

#include "expression.h"
#include "builder.h"
#include "fir-dialect.h"
#include "fir-type.h"
#include "../semantics/expression.h"
#include "../semantics/symbol.h"
#include "../semantics/tools.h"
#include "../semantics/type.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/StandardOps/Ops.h"

namespace Br = Fortran::mlbridge;
namespace Co = Fortran::common;
namespace Ev = Fortran::evaluate;
namespace M = mlir;
namespace Pa = Fortran::parser;
namespace Se = Fortran::semantics;

using namespace Fortran;
using namespace Fortran::mlbridge;

namespace {

template<typename A> int64_t toConstant(const Ev::Expr<A> &e) {
  auto opt = Ev::ToInt64(e);
  assert(opt.has_value() && "expression didn't resolve to a constant");
  return opt.value();
}

#undef TODO
#if 1
#define TODO() assert(false)
#else
#define TODO() \
  return {}
#endif

/// Recover the type of an evaluate::Expr<T> and convert it to an
/// mlir::Type. The type returned can be a MLIR standard or FIR type.
class TypeBuilder {
  M::MLIRContext *context;
  Se::SemanticsContext &semanticsContext;

  int getDefaultRealKind() const {
    return Ev::ExpressionAnalyzer{semanticsContext}.GetDefaultKind(
        Co::TypeCategory::Real);
  }

  int getDefaultIntegerBits() const {
    Ev::ExpressionAnalyzer analyzer(semanticsContext);
    return analyzer.GetDefaultKind(Co::TypeCategory::Integer) * 8;
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

  template<Co::TypeCategory TC, int KIND> M::Type genTy() {
    if constexpr (TC == IntegerCat) {
      return M::IntegerType::get(KIND * 8, context);
    } else if constexpr (TC == RealCat) {
      return genReal<KIND>();
    } else if constexpr (TC == ComplexCat) {
      return M::ComplexType::get(genReal<KIND>());
    } else if constexpr (TC == CharacterCat) {
      return FIRCharacterType::get(context, KIND);
    } else if constexpr (TC == LogicalCat) {
      return FIRLogicalType::get(context, KIND);
    }
    assert(false && "not implemented");
    return {};
  }

  template<Co::TypeCategory TC, int KIND>
  M::Type gen(const Ev::Designator<Ev::Type<TC, KIND>> &) {
    return genTy<TC, KIND>();
  }
  template<Co::TypeCategory TC, int KIND>
  M::Type gen(const Ev::Expr<Ev::Type<TC, KIND>> &) {
    return genTy<TC, KIND>();
  }
  template<Co::TypeCategory TC, int KIND>
  M::Type gen(const Ev::Constant<Ev::Type<TC, KIND>> &) {
    return genTy<TC, KIND>();
  }
  template<Co::TypeCategory TC, int KIND>
  M::Type gen(const Ev::FunctionRef<Ev::Type<TC, KIND>> &) {
    return genTy<TC, KIND>();
  }

  template<typename A> M::Type gen(const Ev::Expr<A> &expr) {
    return std::visit([&](const auto &x) { return gen(x); }, expr.u);
  }

  M::Type mkVoid() { return FIRTupleType::get(context, "(void)"); }

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
        case Co::TypeCategory::Integer:
          returnTy = M::IntegerType::get(kind * 8, context);
          break;
        case Co::TypeCategory::Real: {
          returnTy = genReal(kind);
        } break;
        case Co::TypeCategory::Complex:
          returnTy = M::ComplexType::get(genReal(kind));
          break;
        case Co::TypeCategory::Character:
          returnTy = FIRCharacterType::get(context, kind);
          break;
        case Co::TypeCategory::Logical:
          returnTy = FIRLogicalType::get(context, kind);
          break;
        case Co::TypeCategory::Derived: {
          TODO();
        } break;
        }
      }
    }
    if (symbol->IsObjectArray()) {
      // FIXME: add bounds info
      returnTy = FIRSequenceType::get(
          FIRSequenceType::Shape{FIRSequenceType::Unknown{}}, returnTy);
    } else if (Se::IsPointer(*symbol)) {
      // FIXME: what about allocatable?
      returnTy = FIRReferenceType::get(returnTy);
    }
    return returnTy;
  }

  M::Type gen(const Ev::DataRef &dref) {
    return std::visit([&](const auto &x) { return gen(x); }, dref.u);
  }

  M::Type gen(const Ev::ImpliedDoIndex &) {
    return M::IntegerType::get(getDefaultIntegerBits(), context);
  }

  // FIXME: to be implemented...
  template<Co::TypeCategory TC>
  M::Type gen(const Ev::Designator<Ev::SomeKind<TC>> &des) {
    TODO();
  }
  template<Co::TypeCategory TC>
  M::Type gen(const Ev::Constant<Ev::SomeKind<TC>> &) {
    if constexpr (TC == IntegerCat) {
      return M::IntegerType::get(getDefaultIntegerBits(), context);
    } else if constexpr (TC == RealCat) {
      return genReal(getDefaultRealKind());
    } else {
      TODO();
    }
  }
  template<Co::TypeCategory TC>
  M::Type gen(const Ev::FunctionRef<Ev::SomeKind<TC>> &funref) {
    TODO();
  }
  M::Type gen(const Ev::ProcedureRef &) { TODO(); }
  M::Type gen(const Ev::ProcedureDesignator &) { TODO(); }
  M::Type gen(const Ev::ArrayRef &) { TODO(); }
  M::Type gen(const Ev::NullPointer &) { TODO(); }
  M::Type gen(const Ev::CoarrayRef &) { TODO(); }
  M::Type gen(const Ev::Component &) { TODO(); }
  M::Type gen(const Ev::Substring &) { TODO(); }
  M::Type gen(const Ev::ComplexPart &) { TODO(); }
  M::Type gen(const Ev::DescriptorInquiry &) { TODO(); }
  M::Type gen(const Ev::StructureConstructor &) { TODO(); }
  M::Type gen(const Ev::BOZLiteralConstant &) { TODO(); }
  template<int KIND> M::Type gen(const Ev::TypeParamInquiry<KIND> &) { TODO(); }
  template<typename A> M::Type gen(const Ev::ArrayConstructor<A> &) { TODO(); }
  template<typename A> M::Type gen(const Ev::Relational<A> &) {
    return FIRLogicalType::get(context, 1);
  }
};

#undef TODO
#if 1
#define TODO() assert(false)
#else
#define TODO()
#endif

/// Collect the arguments and build the dictionary for FIR instructions with
/// embedded evaluate::Expr<T> attributes. These arguments strictly conform to
/// data flow between Fortran expressions.
class TreeArgsBuilder {
  FIRBuilder *builder;
  Se::SemanticsContext &semanticsContext;
  Args results;
  Dict dictionary;
  ExprType visited{ET_NONE};

  void addResult(M::Value *v, void *expr) {
    const auto index{results.size()};
    results.push_back(v);
    dictionary[index] = expr;
  }

  inline void addResult(M::Value *v, const void *expr) {
    addResult(v, const_cast<void *>(expr));
  }

  // FIXME: how do we map an evaluate::Expr<T> to a source location?
  M::Location dummyLoc() { return M::UnknownLoc::get(builder->getContext()); }

public:
  explicit TreeArgsBuilder(FIRBuilder *builder, Se::SemanticsContext &sc)
    : builder{builder}, semanticsContext{sc} {}

  /// What we generate for a `Symbol` depends on the category of the symbol
  void gen(const Se::Symbol *variable, void *des, M::Type ty) {
    auto *store = builder->lookupSymbol(variable);
    if (!store) {
      auto ip{builder->getInsertionBlock()};
      builder->setInsertionPointToStart(
          &builder->getBlock()->getFunction()->front());
      store = builder->create<AllocaExpr>(dummyLoc(), ty);
      builder->addSymbol(variable, store);
      builder->setInsertionPointToEnd(ip);
    }
    llvm::SmallVector<M::Value *, 2> loadArg{store};
    auto load{builder->create<LoadExpr>(dummyLoc(), loadArg, ty)};
    addResult(load.getResult(), des);
    visited = ET_Symbol;
  }
  void gen(const Se::Symbol *variable, const void *des, M::Type ty) {
    gen(variable, const_cast<void *>(des), ty);
  }

  // FIXME - need to handle the different cases
  template<typename A> void gen(const Ev::Designator<A> &designator) {
    std::visit(
        Co::visitors{
            [&](const Se::Symbol *x) {
              TypeBuilder tyBldr{builder->getContext(), semanticsContext};
              M::Type ty{tyBldr.gen(designator)};
              gen(x, &designator, ty);
            },
            [&](const auto &x) { gen(x); },
        },
        designator.u);
  }

  template<typename D, typename R, typename... A>
  void gen(const Ev::Operation<D, R, A...> &op) {
    gen(op.left());
    if constexpr (op.operands > 1) {
      gen(op.right());
    }
    visited = ET_Operation;
  }

  void gen(const Ev::DataRef &dref) {
    std::visit(
        Co::visitors{
            [&](const Se::Symbol *x) {
              TypeBuilder tyBldr{builder->getContext(), semanticsContext};
              M::Type ty{tyBldr.gen(dref)};
              gen(x, &dref, ty);
            },
            [&](const auto &x) { gen(x); },
        },
        dref.u);
  }

  void gen(const Ev::Triplet &triplet) {
    if (triplet.lower().has_value()) gen(triplet.lower().value());
    if (triplet.upper().has_value()) gen(triplet.upper().value());
    gen(triplet.stride());
  }

  /// Lower an array reference
  ///
  /// Visit each of the expressions in each dimension of the array access so as
  /// to add them to the data flow.
  void gen(const Ev::ArrayRef &aref) {
    // add the base
    std::visit(
        Co::visitors{
            [&](const Ev::Component &x) { gen(x); },
            [&](const Se::Symbol *x) {
              TypeBuilder tyBldr{builder->getContext(), semanticsContext};
              M::Type ty{tyBldr.gen(x)};
              gen(x, &aref.base(), ty);  // FIXME wrong type, use sequence!
            },
        },
        aref.base());
    // add each expression
    for (int i = 0, e = aref.size(); i < e; ++i)
      std::visit(Co::visitors{
                     [&](const Ev::Triplet &t) { gen(t); },
                     [&](const Ev::IndirectSubscriptIntegerExpr &x) {
                       gen(x.value());
                     },
                 },
          aref.at(i).u);
    visited = ET_ArrayRef;
  }

  void gen(const Ev::NullPointer &) { visited = ET_NullPointer; }
  void gen(const Ev::CoarrayRef &) {
    TODO();
    visited = ET_CoarrayRef;
  }
  void gen(const Ev::Component &) {
    TODO();
    visited = ET_Component;
  }
  void gen(const Ev::Substring &) {
    TODO();
    visited = ET_Substring;
  }
  void gen(const Ev::ComplexPart &) { visited = ET_ComplexPart; }
  void gen(const Ev::ImpliedDoIndex &) { TODO(); }
  void gen(const Ev::StructureConstructor &) {
    TODO();
    visited = ET_StructureCtor;
  }

  // Skip over constants
  void gen(const Ev::BOZLiteralConstant &) { visited = ET_Constant; }
  template<typename A> void gen(const Ev::Constant<A> &) {
    visited = ET_Constant;
  }

  void gen(const Ev::DescriptorInquiry &) {
    TODO();
    visited = ET_DescriptorInquiry;
  }
  template<int KIND> void gen(const Ev::TypeParamInquiry<KIND> &inquiry) {
    std::visit(Co::visitors{
                   [&](const Se::Symbol *x) {},  // FIXME
                   [&](auto &x) { gen(x); },
               },
        inquiry.base());
    visited = ET_TypeParamInquiry;
  }

  void gen(const Ev::ProcedureRef &) { TODO(); }
  void gen(const Ev::ProcedureDesignator &) { TODO(); }

  M::FunctionType genFunctionType(const Ev::ProcedureDesignator &proc) {
    return std::visit(Co::visitors{
                          [&](const Ev::SpecificIntrinsic &) {
                            TODO(); /* FIXME */
                            return M::Type{};
                          },
                          [&](const Se::Symbol *x) {
                            return translateSymbolToFIRType(
                                *builder->getContext(), semanticsContext, x);
                          },
                          [&](const Co::CopyableIndirection<Ev::Component> &x) {
                            return translateSymbolToFIRType(
                                *builder->getContext(), semanticsContext,
                                &x.value().GetLastSymbol());
                          },
                      },
        proc.u)
        .cast<M::FunctionType>();
  }

  /// Lower a function call
  ///
  /// A call is lowered to a sequence that evaluates the arguments and
  /// passes them to a CallOp to the named function.
  ///
  /// TODO: The function needs to be name-mangled by the front-end or some
  /// utility to avoid collisions.
  ///
  ///   %54 = FIR.ApplyExpr(...)
  ///   %55 = FIR.ApplyExpr(...)
  ///   %56 = call(@target_func, %54, %55)
  template<typename A> void gen(const Ev::FunctionRef<A> &funref) {
    // must be lowered to a call and data flow added for each actual arg
    auto *context = builder->getContext();
    M::Location loc = M::UnknownLoc::get(context);  // FIXME
    TypeBuilder tyBldr{context, semanticsContext};

    // lookup this function
    llvm::StringRef callee = funref.proc().GetName();
    auto *module{builder->getBlock()->getFunction()->getModule()};
    auto *func = module->getNamedFunction(callee);
    if (!func) {
      // create new function
      TypeBuilder bldr{context, semanticsContext};
      auto funTy{genFunctionType(funref.proc())};
      func = createFunction(module, callee, funTy);
    }
    assert(func);

    // build arguments
    llvm::SmallVector<M::Value *, 4> inputs;
    for (auto &arg : funref.arguments()) {
      if (arg.has_value()) {
        if (auto *aa{arg->UnwrapExpr()}) {
          auto eops{translateSomeExpr(builder, semanticsContext, aa)};
          auto aaType{FIRReferenceType::get(
              translateSomeExprToFIRType(*context, semanticsContext, aa))};
          M::Value *toLoc{nullptr};
          auto *defOp{getArgs(eops)[0]->getDefiningOp()};
          if (auto load = M::dyn_cast<LoadExpr>(defOp)) {
            toLoc = load.getOperand();
          } else {
            auto locate{builder->create<LocateExpr>(
                loc, aa, getDict(eops), getArgs(eops), aaType)};
            toLoc = locate.getResult();
          }
          inputs.push_back(toLoc);
        } else {
          auto *x{arg->GetAssumedTypeDummy()};
          auto xType{translateSymbolToFIRType(*context, semanticsContext, x)};
          gen(x, &arg, xType);
        }
      } else {
        assert(false && "argument is std::nullopt?");
      }
    }

    // generate a call
    auto call = builder->create<M::CallOp>(loc, func, inputs);
    addResult(call.getResult(0), &funref);
    visited = ET_FunctionRef;
  }

  template<typename A> void gen(const Ev::ImpliedDo<A> &impliedDo) { TODO(); }
  template<typename A> void gen(const Ev::ArrayConstructor<A> &arrayCtor) {
    for (auto i{arrayCtor.begin()}, end{arrayCtor.end()}; i != end; ++i) {
      std::visit([&](auto &x) { gen(x); }, i->u);
    }
    visited = ET_ArrayCtor;
  }

  template<typename A> void gen(const Ev::Relational<A> &op) {
    gen(op.left());
    gen(op.right());
    visited = ET_Relational;
  }
  void gen(const Ev::Relational<Ev::SomeType> &op) {
    std::visit([&](const auto &x) { gen(x); }, op.u);
  }

  template<typename A> void gen(const Ev::Expr<A> &expr) {
    std::visit([&](const auto &x) { gen(x); }, expr.u);
  }

  Args getResults() const { return results; }
  Dict getDictionary() const { return dictionary; }
  ExprType getExprType() const { return visited; }
};

}  // namespace

// Builds the `([inputs], {dict})` pair for constructing both `fir.apply_expr`
// and `fir.locate_expr` operations.
Values Br::translateSomeExpr(FIRBuilder *builder,
    Se::SemanticsContext &semanticsContext, const SomeExpr *expr) {
  TreeArgsBuilder argsBldr{builder, semanticsContext};
  argsBldr.gen(*expr);
  return {
      argsBldr.getResults(), argsBldr.getDictionary(), argsBldr.getExprType()};
}

// Builds the FIR type from an instance of SomeExpr
M::Type Br::translateSomeExprToFIRType(M::MLIRContext &context,
    Se::SemanticsContext &semanticsContext, const SomeExpr *expr) {
  TypeBuilder tyBldr{&context, semanticsContext};
  return tyBldr.gen(*expr);
}

// This entry point avoids gratuitously wrapping the Symbol instance in layers
// of Expr<T> that will then be immediately peeled back off and discarded.
M::Type Br::translateSymbolToFIRType(M::MLIRContext &context,
    Se::SemanticsContext &semanticsContext, const Se::Symbol *symbol) {
  TypeBuilder tyBldr{&context, semanticsContext};
  return tyBldr.gen(symbol);
}

M::Type Br::convertReal(int KIND, M::MLIRContext *context) {
  return TypeBuilder::genReal(KIND, context);
}
