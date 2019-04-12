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
#include <variant>

namespace Fortran::mlbridge {

// In the Fortran::mlbridge namespace, the code will default follow the
// LLVM/MLIR coding standards

/// Fortran dialect types
class FIRDialect : public mlir::Dialect {
public:
  explicit FIRDialect(mlir::MLIRContext *ctx);

  mlir::Type parseType(llvm::StringRef ty, mlir::Location loc) const override;
  void printType(mlir::Type ty, llvm::raw_ostream &os) const override;
};

// FIR dialect operations

/// Application of a FIR expression
class ApplyExpr : public mlir::Op<ApplyExpr, mlir::OpTrait::VariadicOperands,
                      mlir::OpTrait::OneResult> {
public:
  explicit ApplyExpr(const evaluate::Expr<evaluate::SomeType> &expr)
    : Op{}, expr{expr} {}
  explicit ApplyExpr() : Op{}, expr{std::monostate{}} {}

  static llvm::StringRef getOperationName() { return "fir.apply_expr"; }

  mlir::LogicalResult verify();

  static void build(mlir::FuncBuilder *builder, mlir::OperationState *state,
      llvm::StringRef lambda, llvm::ArrayRef<mlir::Value *> args);

  llvm::StringRef getExpr();

private:
  std::variant<std::monostate, evaluate::Expr<evaluate::SomeType>> expr;

  // boilerplate for adding operation
  friend class mlir::Operation;
  using Op::Op;
};

/// Location implied by FIR expression
class LocateExpr : public mlir::Op<LocateExpr, mlir::OpTrait::VariadicOperands,
                       mlir::OpTrait::OneResult> {
public:
  explicit LocateExpr(const evaluate::Expr<evaluate::SomeType> &expr)
    : Op{}, expr{expr} {}
  explicit LocateExpr() : Op{}, expr{std::monostate{}} {}

  static llvm::StringRef getOperationName() { return "fir.locate_expr"; }

  mlir::LogicalResult verify();

  static void build(mlir::FuncBuilder *builder, mlir::OperationState *state,
      llvm::StringRef lambda, llvm::ArrayRef<mlir::Value *> args);

  llvm::StringRef getExpr();

private:
  std::variant<std::monostate, evaluate::Expr<evaluate::SomeType>> expr;

  // boilerplate for adding operation
  friend class mlir::Operation;
  using Op::Op;
};

}  // Fortran::mlbridge

#endif  // FORTRAN_MLBRIDGE_FORTRAN_IR_DIALECT_H_
