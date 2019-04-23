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
#include <string>
#include <variant>

namespace Fortran::evaluate {
struct SomeType;
template<typename A> class Expr;
}

namespace Fortran::mlbridge {

// In the Fortran::mlbridge namespace, the code will default follow the
// LLVM/MLIR coding standards

using SomeExpr = evaluate::Expr<evaluate::SomeType>;
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
///   %82 = "fir.apply_expr"(%72, %75){"!0!_8 * !1!_8"} : i64
class ApplyExpr : public mlir::Op<ApplyExpr, mlir::OpTrait::VariadicOperands,
                      mlir::OpTrait::OneResult> {
public:
  using Op::Op;

  static StringRef getOperationName() { return "fir.apply_expr"; }

  mlir::LogicalResult verify();

  static void build(mlir::FuncBuilder *builder, mlir::OperationState *result,
      const SomeExpr *expr, llvm::ArrayRef<mlir::Value *> operands,
      mlir::Type opTy);

  StringRef getExpr();
};

/// The "fir.locate_expr" operation evaluates a Fortran expression to obtain the
/// address to which it refers.  The following computes the address of the value
/// in a Fortran array at position (1,4).
///
///   %82 = "fir.locate_expr"(%75){"!0!(1,4)"} : i64
class LocateExpr : public mlir::Op<LocateExpr, mlir::OpTrait::VariadicOperands,
                       mlir::OpTrait::OneResult> {
public:
  using Op::Op;

  static StringRef getOperationName() { return "fir.locate_expr"; }

  mlir::LogicalResult verify();

  static void build(mlir::FuncBuilder *builder, mlir::OperationState *result,
      const SomeExpr *expr, llvm::ArrayRef<mlir::Value *> operands,
      mlir::Type opTy);

  StringRef getExpr();
};

/// The "fir.alloca_expr" operation is used to allocate local temporary space
/// for object of a specified type.  The result of "fir.alloca_expr" is a
/// pointer to an uninitialized ssa-object of the designated type.  Together
/// with "fir.load_expr" and "fir.store_expr", these operations are used to
/// construct a mem-ssa representation, which will be lowered to the MLIR
/// standard dialect.
///
///   %86 = "fir.alloca_expr"(){expr: "i"} : i64*
class AllocaExpr : public mlir::Op<AllocaExpr, mlir::OpTrait::ZeroOperands,
                       mlir::OpTrait::OneResult> {
public:
  using Op::Op;

  static StringRef getOperationName() { return "fir.alloca_expr"; }

  mlir::LogicalResult verify();

  static void build(mlir::FuncBuilder *builder, mlir::OperationState *result,
      const SomeExpr *expr, llvm::ArrayRef<mlir::Value *> operands,
      mlir::Type opTy);

  StringRef getExpr();
};

/// The "fir.store_expr" operation is used to store the result of a
/// "fir.apply_expr" operation to memory specified by the result of a
/// "fir.locate_expr" operation.
///
///   %83 = "fir.store_expr"(%75, %76) : (i64, i64*) -> ()
class StoreExpr : public mlir::Op<StoreExpr, mlir::OpTrait::NOperands<2>::Impl,
                      mlir::OpTrait::ZeroResult> {
public:
  using Op::Op;

  static StringRef getOperationName() { return "fir.store_expr"; }

  mlir::LogicalResult verify();

  static void build(mlir::FuncBuilder *builder, mlir::OperationState *result,
      llvm::ArrayRef<mlir::Value *> operands);
};

/// The "fir.load_expr" operation is used to load the result of a
/// "fir.locate_expr" operation from memory.  The result of "fir.load_expr"
/// is an ssa-object with a type.
///
///   %85 = "fir.load_expr"(%77) : (tensor<10xf32>*) -> tensor<10xf32>
class LoadExpr : public mlir::Op<LoadExpr, mlir::OpTrait::OneOperand,
                     mlir::OpTrait::OneResult> {
public:
  using Op::Op;

  static StringRef getOperationName() { return "fir.load_expr"; }

  mlir::LogicalResult verify();

  static void build(mlir::FuncBuilder *builder, mlir::OperationState *result,
      llvm::ArrayRef<mlir::Value *> operands, mlir::Type opTy);
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
  static void build(mlir::FuncBuilder *builder, mlir::OperationState *result);
};

}  // Fortran::mlbridge

#endif  // FORTRAN_MLBRIDGE_FORTRAN_IR_DIALECT_H_
