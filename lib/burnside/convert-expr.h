//===-- lib/burnside/convert-expr.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_BURNSIDE_CONVERT_EXPR_H_
#define FORTRAN_BURNSIDE_CONVERT_EXPR_H_

#include "intrinsics.h"

/// [Coding style](https://llvm.org/docs/CodingStandards.html)

namespace mlir {
class Location;
class OpBuilder;
class Type;
class Value;
}  // mlir

namespace fir {
class AllocaExpr;
}  // fir

namespace Fortran {
namespace common {
class IntrinsicTypeDefaultKinds;
}  //  common
namespace evaluate {
template<typename> class Expr;
struct SomeType;
}  // evaluate
namespace semantics {
class Symbol;
}  // semantics

namespace burnside {

class AbstractConverter;
class SymMap;

mlir::Value *createSomeExpression(mlir::Location loc,
    AbstractConverter &converter,
    const evaluate::Expr<evaluate::SomeType> &expr, SymMap &symMap,
    const IntrinsicLibrary &intrinsics);

mlir::Value *createI1LogicalExpression(mlir::Location loc,
    AbstractConverter &converter,
    const evaluate::Expr<evaluate::SomeType> &expr, SymMap &symMap,
    const IntrinsicLibrary &intrinsics);

mlir::Value *createSomeAddress(mlir::Location loc, AbstractConverter &converter,
    const evaluate::Expr<evaluate::SomeType> &expr, SymMap &symMap,
    const IntrinsicLibrary &intrinsics);

mlir::Value *createTemporary(mlir::Location loc, mlir::OpBuilder &builder,
    SymMap &symMap, mlir::Type type, const semantics::Symbol *symbol);

}  // burnside
}  // Fortran

#endif  // FORTRAN_BURNSIDE_CONVERT_EXPR_H_
