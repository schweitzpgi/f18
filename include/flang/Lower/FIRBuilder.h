//===-- Lower/FirBuilder.h -- FIR operation builder -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Builder routines for constructing the FIR dialect of MLIR. As FIR is a
// dialect of MLIR, it makes extensive use of MLIR interfaces and MLIR's coding
// style (https://mlir.llvm.org/getting_started/DeveloperGuide/) is used in this
// module.
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_FIRBUILDER_H
#define FORTRAN_LOWER_FIRBUILDER_H

#include "flang/Common/reference.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"

namespace Fortran {
namespace semantics {
class Symbol;
using SymbolRef = common::Reference<const Symbol>;
} // namespace semantics

namespace lower {

class AbstractConverter;

/// Helper class to map front-end symbols to their MLIR representation. This
/// provides a way to lookup the fir.alloca location of a variable, for example.
class SymMap {
public:
  /// Add `symbol` to the current map and  bind to `value`.
  void addSymbol(semantics::SymbolRef symbol, mlir::Value value);

  /// Find `symbol` and return its value if it appears in the current mappings.
  mlir::Value lookupSymbol(semantics::SymbolRef symbol);

  void clear() { symbolMap.clear(); }

private:
  llvm::DenseMap<const semantics::Symbol *, mlir::Value> symbolMap;
};

/// Pair of ssa-values that compose a CHARACTER value. The buffer that holds the
/// data and the LEN type parameter.
struct CharValue {
  mlir::Value buffer;
  mlir::Value len;
};

//===----------------------------------------------------------------------===//
// FirOpBuilder interface extensions
//===----------------------------------------------------------------------===//

// TODO: Used CRTP to extend the FirOpBuilder interface, but this leads to some
// complex and downright ugly template code.

/// Extension class to facilitate lowering of CHARACTER operation
template <typename T>
class CharacterOpsBuilder {
public:
  fir::ReferenceType getReferenceType(const CharValue &pair);
  fir::CharacterType getCharacterType(const CharValue &pair);

  // access the implementation
  T &impl() { return *static_cast<T *>(this); }

  /// Copy the \p count first characters of \p src into \p dest.
  void createCopy(CharValue &dest, const CharValue &src, mlir::Value count);

  /// Set characters of \p str at position [\p lower, \p upper) to blanks.
  /// \p lower and \upper bounds are zero based.
  /// If \p upper <= \p lower, no padding is done.
  void createPadding(CharValue &str, mlir::Value lower, mlir::Value upper);

  /// Allocate storage (on the stack) for character given the kind and length.
  CharValue createCharacterTemp(fir::CharacterType type, mlir::Value len);

protected:
  mlir::Value createBlankConstant(fir::CharacterType type);
};

/// Extension class to facilitate lowering of COMPLEX manipulations in FIR.
template <typename T>
class ComplexOpsBuilder {
public:
  // The values of part enum members are meaningful for
  // InsertValueOp and ExtractValueOp so they are explicit.
  enum class Part { Real = 0, Imag = 1 };

  // access the implementation
  T &impl() { return *static_cast<T *>(this); }

  // Type helper. They do not create MLIR operations.
  mlir::Type getComplexPartType(mlir::Value cplx);
  mlir::Type getComplexPartType(mlir::Type complexType);
  mlir::Type getComplexPartType(fir::KindTy complexKind);

  // Complex operation creation helper. They create MLIR operations.
  mlir::Value createComplex(fir::KindTy kind, mlir::Value real,
                            mlir::Value imag);
  mlir::Value extractComplexPart(mlir::Value cplx, bool isImagPart) {
    return isImagPart ? extract<Part::Imag>(cplx) : extract<Part::Real>(cplx);
  }
  mlir::Value insertComplexPart(mlir::Value cplx, mlir::Value part,
                                bool isImagPart) {
    return isImagPart ? insert<Part::Imag>(cplx, part)
                      : insert<Part::Real>(cplx, part);
  }

  mlir::Value createComplexCompare(mlir::Value cplx1, mlir::Value cplx2,
                                   bool eq);

protected:
  template <Part partId>
  mlir::Value extract(mlir::Value cplx) {
    return impl().template createHere<fir::ExtractValueOp>(
        getComplexPartType(cplx), cplx, createPartId<partId>());
  }

  template <Part partId>
  mlir::Value insert(mlir::Value cplx, mlir::Value part) {
    return impl().template createHere<fir::InsertValueOp>(
        cplx.getType(), cplx, part, createPartId<partId>());
  }

  template <Part partId>
  mlir::Value createPartId() {
    return impl().createIntegerConstant(impl().getIntegerType(32),
                                        static_cast<int>(partId));
  }
};

//===----------------------------------------------------------------------===//
// FirOpBuilder
//===----------------------------------------------------------------------===//

/// Extends the MLIR OpBuilder to provide methods for building common FIR
/// patterns.
class FirOpBuilder : public mlir::OpBuilder,
                     public CharacterOpsBuilder<FirOpBuilder>,
                     public ComplexOpsBuilder<FirOpBuilder> {
public:
  using OpBuilder::OpBuilder;

  /// TODO: remove this as caching the location may have the location
  /// unexpectedly overridden along the way.
  /// Set the current location. Used by createHere template method, etc.
  void setLocation(mlir::Location loc) { currentLoc = loc; }

  /// Get the current location (if any) or return unknown location.
  mlir::Location getLoc() {
    return currentLoc.hasValue() ? currentLoc.getValue() : getUnknownLoc();
  }

  template <typename OP, typename... AS>
  auto createHere(AS... args) {
    return create<OP>(getLoc(), std::forward<AS>(args)...);
  }

  mlir::Region &getRegion() { return *getBlock()->getParent(); }

  /// Get the current Module
  mlir::ModuleOp getModule() {
    return getRegion().getParentOfType<mlir::ModuleOp>();
  }

  /// Get the current Function
  mlir::FuncOp getFunction() {
    return getRegion().getParentOfType<mlir::FuncOp>();
  }

  /// Get the entry block of the current Function
  mlir::Block *getEntryBlock() { return &getFunction().front(); }

  /// Create an integer constant of type \p type and value \p i.
  mlir::Value createIntegerConstant(mlir::Type integerType, std::int64_t i);

  /// Create a temporary. A temp is allocated using `fir.alloca` and can be read
  /// and written using `fir.load` and `fir.store`, resp.  The temporary can be
  /// given a name via a front-end `Symbol` or a `StringRef`.
  mlir::Value createTemporary(mlir::Location loc, SymMap &symMap,
                              mlir::Type type,
                              llvm::ArrayRef<mlir::Value> shape = {},
                              const Fortran::semantics::Symbol *symbol = {},
                              llvm::StringRef name = {});

  mlir::Value createTemporary(SymMap &symMap, mlir::Type type,
                              llvm::ArrayRef<mlir::Value> shape = {},
                              const Fortran::semantics::Symbol *symbol = {},
                              llvm::StringRef name = {}) {
    return createTemporary(getLoc(), symMap, type, shape, symbol, name);
  }

  /// Create an unnamed and untracked temporary on the stack.
  mlir::Value createTemporary(mlir::Type type,
                              llvm::ArrayRef<mlir::Value> shape) {
    SymMap fakeMap;
    return createTemporary(getLoc(), fakeMap, type, shape);
  }

  /// Create a global value.
  fir::GlobalOp createGlobal(mlir::Location loc, mlir::Type type,
                             llvm::StringRef name, mlir::Attribute value = {},
                             mlir::StringAttr linkage = {},
                             bool isConst = false);

  /// Create a global constant (read-only) value.
  fir::GlobalOp createGlobalConstant(mlir::Location loc, mlir::Type type,
                                     llvm::StringRef name,
                                     mlir::Attribute value = {},
                                     mlir::StringAttr linkage = {}) {
    return createGlobal(loc, type, name, value, linkage, /*isConst=*/true);
  }

  mlir::Block *createBlock() {
    return OpBuilder::createBlock(getBlock()->getParent());
  }

  /// Get a function by name. If the function exists in the current module, it
  /// is returned. Otherwise, a null FuncOp is returned.
  mlir::FuncOp getNamedFunction(llvm::StringRef name) {
    return getNamedFunction(getModule(), name);
  }

  static mlir::FuncOp getNamedFunction(mlir::ModuleOp module,
                                       llvm::StringRef name);

  /// Create a new FuncOp. If the function may have already been created, use
  /// `addNamedFunction` instead.
  mlir::FuncOp createFunction(mlir::Location loc, llvm::StringRef name,
                              mlir::FunctionType ty) {
    return createFunction(loc, getModule(), name, ty);
  }

  mlir::FuncOp createFunction(llvm::StringRef name, mlir::FunctionType ty) {
    return createFunction(getLoc(), name, ty);
  }

  static mlir::FuncOp createFunction(mlir::Location loc, mlir::ModuleOp module,
                                     llvm::StringRef name,
                                     mlir::FunctionType ty);

  /// Determine if the named function is already in the module. Return the
  /// instance if found, otherwise add a new named function to the module.
  mlir::FuncOp addNamedFunction(mlir::Location loc, llvm::StringRef name,
                                mlir::FunctionType ty) {
    if (auto func = getNamedFunction(name))
      return func;
    return createFunction(loc, name, ty);
  }

  mlir::FuncOp addNamedFunction(llvm::StringRef name, mlir::FunctionType ty) {
    if (auto func = getNamedFunction(name))
      return func;
    return createFunction(name, ty);
  }

  static mlir::FuncOp addNamedFunction(mlir::Location loc,
                                       mlir::ModuleOp module,
                                       llvm::StringRef name,
                                       mlir::FunctionType ty) {
    if (auto func = getNamedFunction(module, name))
      return func;
    return createFunction(loc, module, name, ty);
  }

  //===--------------------------------------------------------------------===//
  // LoopOp helpers
  //===--------------------------------------------------------------------===//

  using BodyGenerator = std::function<void(FirOpBuilder &, mlir::Value)>;

  /// Build loop [\p lb, \p ub) with step \p step.
  /// If \p step is an empty value, 1 is used for the step.
  void createLoop(mlir::Value lb, mlir::Value ub, mlir::Value step,
                  const BodyGenerator &bodyGenerator);

  /// Build loop [\p lb,  \p ub) with step 1.
  void createLoop(mlir::Value lb, mlir::Value ub,
                  const BodyGenerator &bodyGenerator);

  /// Build loop [0, \p count) with step 1.
  void createLoop(mlir::Value count, const BodyGenerator &bodyGenerator);

  /// Cast the input value to IndexType.
  mlir::Value convertToIndexType(mlir::Value integer);

private:
  llvm::Optional<mlir::Location> currentLoc{};
};

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_FIRBUILDER_H
