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

#ifndef FORTRAN_MLBRIDGE_FORTRAN_IR_DIALECT_H_
#define FORTRAN_MLBRIDGE_FORTRAN_IR_DIALECT_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "../evaluate/expression.h"
#include <string>
#include <variant>

namespace Fortran::mlbridge {

// In the Fortran::mlbridge namespace, the code will default follow the
// LLVM/MLIR coding standards

using StringRef = llvm::StringRef;

/// Fortran dialect types
class FIRDialect : public mlir::Dialect {
public:
  explicit FIRDialect(mlir::MLIRContext *ctx);
  virtual ~FIRDialect();

  mlir::Type parseType(StringRef ty, mlir::Location loc) const override;
  void printType(mlir::Type ty, llvm::raw_ostream &os) const override;
};

// FIR dialect operations

/// The "fir.apply_expr" operation evaluates a Fortran expression to obtain the
/// value that the expression.  The following computes the product of the two
/// ssa-values %72 and %75.
///
///   %82 = fir.apply_expr(%72, %75){"!0!_8 * !1!_8"} : i64
class ApplyExpr : public mlir::Op<ApplyExpr, mlir::OpTrait::VariadicOperands,
                      mlir::OpTrait::OneResult> {
public:
  explicit ApplyExpr(const evaluate::Expr<evaluate::SomeType> &expr)
    : Op{}, expr{expr} {}
  explicit ApplyExpr(StringRef expr) : Op{}, expr{expr.str()} {}
  using Op::Op;

  static StringRef getOperationName() { return "fir.apply_expr"; }

  mlir::LogicalResult verify();

  static void build(mlir::FuncBuilder *builder, mlir::OperationState *state,
      StringRef lambda, llvm::ArrayRef<mlir::Value *> args);

  StringRef getExpr();

private:
  std::variant<std::string, evaluate::Expr<evaluate::SomeType>> expr;
};

/// The "fir.locate_expr" operation evaluates a Fortran expression to obtain the
/// address to which it refers.  The following computes the address of the value
/// in a Fortran array at position (1,4).
///
///   %82 = fir.locate_expr(%75){"!0!(1,4)"} : i64
class LocateExpr : public mlir::Op<LocateExpr, mlir::OpTrait::VariadicOperands,
                       mlir::OpTrait::OneResult> {
public:
  explicit LocateExpr(const evaluate::Expr<evaluate::SomeType> &expr)
    : Op{}, expr{expr} {}
  explicit LocateExpr(StringRef expr) : Op{}, expr{expr.str()} {}
  using Op::Op;

  static StringRef getOperationName() { return "fir.locate_expr"; }

  mlir::LogicalResult verify();

  static void build(mlir::FuncBuilder *builder, mlir::OperationState *state,
      StringRef lambda, llvm::ArrayRef<mlir::Value *> args);

  StringRef getExpr();

private:
  std::variant<std::string, evaluate::Expr<evaluate::SomeType>> expr;
};

/// The "fir.unreachable" terminator represents an instruction that should
/// never be reached.  This can happen if a preceeding "call" is known to
/// not return to the caller. Lowers to an LLVM unreachable terminator.
///
///   call @fortran_stop(%5)
///   fir.unreachable
class UnreachableOp
  : public mlir::Op<UnreachableOp, mlir::OpTrait::ZeroOperands,
        mlir::OpTrait::ZeroResult, mlir::OpTrait::IsTerminator> {
public:
  using Op::Op;

  static StringRef getOperationName() { return "fir.unreachable"; }
  static void build(mlir::FuncBuilder *builder, mlir::OperationState *state);
};

}  // Fortran::mlbridge

#endif  // FORTRAN_MLBRIDGE_FORTRAN_IR_DIALECT_H_
