//===-- lib/lower/builder.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_BUILDER_H_
#define FORTRAN_LOWER_BUILDER_H_

#include "../../../lib/semantics/symbol.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "llvm/ADT/DenseMap.h"
#include <string>

namespace llvm {
class StringRef;
}

namespace fir {
class CharacterType;
class ReferenceType;
} // namespace fir

namespace Fortran {
namespace parser {
class CookedSource;
}

namespace evaluate {
struct ProcedureDesignator;
}

namespace lower {

/// Miscellaneous helper routines for building MLIR
///
/// [Coding style](https://llvm.org/docs/CodingStandards.html)

class AbstractConverter;

class SymMap {
  llvm::DenseMap<const semantics::Symbol *, mlir::Value> symbolMap;
  std::vector<std::pair<const semantics::Symbol *, mlir::Value>> shadowStack;

public:
  void addSymbol(semantics::SymbolRef symbol, mlir::Value value);

  mlir::Value lookupSymbol(semantics::SymbolRef symbol);

  void pushShadowSymbol(semantics::SymbolRef symbol, mlir::Value value);
  void popShadowSymbol() { shadowStack.pop_back(); }

  void clear() {
    symbolMap.clear();
    shadowStack.clear();
  }
};

/// Helper class that can be inherited from in order to
/// facilitate mlir::OpBuilder usage.
class OpBuilderHandler {
public:
  OpBuilderHandler(mlir::OpBuilder &b, mlir::Location l) : builder{b}, loc{l} {}
  template <typename T, typename... Args>
  auto create(Args... args) {
    return builder.create<T>(loc, std::forward<Args>(args)...);
  }
  mlir::Value getIntegerConstant(mlir::Type integerType, std::int64_t);

protected:
  mlir::OpBuilder &builder;
  mlir::Location loc;
};

/// Facilitate lowering to fir::loop
class LoopCreator : public OpBuilderHandler {
public:
  LoopCreator(mlir::OpBuilder &b, mlir::Location l) : OpBuilderHandler{b, l} {}
  LoopCreator(OpBuilderHandler &b) : OpBuilderHandler{b} {}
  // In genLoop functions, lb, ub, and count arguments must have integer types.
  // They will be automatically converted the IndexType if needed.
  // Build loop [lb, ub) with step "step".
  using BodyGenerator = std::function<void(OpBuilderHandler &, mlir::Value)>;
  void genLoop(mlir::Value lb, mlir::Value ub, mlir::Value step,
               const BodyGenerator &bodyGenerator);
  /// Build loop [lb, ub) with step 1.
  void genLoop(mlir::Value lb, mlir::Value ub,
               const BodyGenerator &bodyGenerator);
  /// Build loop [0, count) with step 1.
  void genLoop(mlir::Value count, const BodyGenerator &bodyGenerator);

private:
  mlir::Type getIndexType();
  mlir::Value convertToIndexType(mlir::Value integer);
};

/// Facilitate lowering of CHARACTER operation
class CharacterOpsCreator : public OpBuilderHandler {
public:
  CharacterOpsCreator(mlir::OpBuilder &b, mlir::Location l)
      : OpBuilderHandler{b, l} {}
  CharacterOpsCreator(OpBuilderHandler &b) : OpBuilderHandler{b} {}
  /// Interchange format to avoid inserting unbox/embox everywhere while
  /// evaluating character expressions.
  struct CharValue {
    fir::ReferenceType getReferenceType();
    fir::CharacterType getCharacterType();

    mlir::Value reference;
    mlir::Value len;
  };

  /// Copy count first characters of src into dest.
  void genCopy(CharValue &dest, CharValue &src, mlir::Value count);
  /// Pad str(from:to) with blanks. If to <= from, no padding is done.
  void genPadding(CharValue &str, mlir::Value from, mlir::Value to);
  /// allocate (on the stack) storage for character given the kind and length.
  CharValue createTemp(fir::CharacterType type, mlir::Value len);

private:
  mlir::Value getBlankConstant(fir::CharacterType type);
};

/// Get the current Module
inline mlir::ModuleOp getModule(mlir::OpBuilder *bldr) {
  return bldr->getBlock()->getParent()->getParentOfType<mlir::ModuleOp>();
}

/// Get the current Function
inline mlir::FuncOp getFunction(mlir::OpBuilder *bldr) {
  return bldr->getBlock()->getParent()->getParentOfType<mlir::FuncOp>();
}

/// Get the entry block of the current Function
inline mlir::Block *getEntryBlock(mlir::OpBuilder *bldr) {
  return &getFunction(bldr).front();
}

/// Create a new basic block
inline mlir::Block *createBlock(mlir::OpBuilder *bldr, mlir::Region *region) {
  return bldr->createBlock(region, region->end());
}

inline mlir::Block *createBlock(mlir::OpBuilder *bldr) {
  return createBlock(bldr, bldr->getBlock()->getParent());
}

/// Get a function by name (or null)
mlir::FuncOp getNamedFunction(mlir::ModuleOp, llvm::StringRef name);

/// Create a new FuncOp
mlir::FuncOp createFunction(AbstractConverter &converter, llvm::StringRef name,
                            mlir::FunctionType funcTy);

/// Create a new FuncOp
/// The function is created with no Location information
mlir::FuncOp createFunction(mlir::ModuleOp module, llvm::StringRef name,
                            mlir::FunctionType funcTy);

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_BUILDER_H_
