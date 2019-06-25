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

#ifndef FORTRAN_LIB_MLBRIDGE_FIR_DIALECT_H
#define FORTRAN_LIB_MLBRIDGE_FIR_DIALECT_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include <map>
#include <string>
#include <variant>  // TODO: remove

namespace mlir {
class ConstantOp;
}

namespace Fortran::evaluate {
struct SomeType;
template<typename> class Expr;
}

namespace Fortran::mlbridge {

// In the Fortran::mlbridge namespace, the code will default follow the
// LLVM/MLIR coding standards

using SomeExpr = evaluate::Expr<evaluate::SomeType>;

/// Fortran dialect types
class FIROpsDialect : public mlir::Dialect {
public:
  explicit FIROpsDialect(mlir::MLIRContext *ctx);
  virtual ~FIROpsDialect();

  static llvm::StringRef getDialectNamespace() { return "fir"; }

  mlir::Type parseType(
      llvm::StringRef rawData, mlir::Location loc) const override;
  void printType(mlir::Type ty, llvm::raw_ostream &os) const override;

  mlir::Attribute parseAttribute(
      llvm::StringRef attrData, mlir::Location loc) const override;
  void printAttribute(
      mlir::Attribute attr, llvm::raw_ostream &os) const override;
};

// FIR dialect operations

/// The "fir.address" operation is used to compute the address of a well-typed
/// expression.  The first argument is the base object.  This is compositional
/// so multiple "fir.address" operations can be chained.  "fir.address" with
/// a single argument devolves to a NOP.
///
///   %45 = ... : !fir.ref<!fir.type<"T", {"A" : !fir.array<1 10 1:i64>}>>
///   %46 = "fir.field_value"("A") : !fir.field_type
///   %47 = ... : i32   ; index `i` into array `A`
///   %48 = "fir.address"(%45, %46, %47) : !fir.ref<i64>
class AddressOp
  : public mlir::Op<AddressOp, mlir::OpTrait::VariadicOperands,
        mlir::OpTrait::HasNoSideEffect, mlir::OpTrait::OneResult> {
public:
  using Op::Op;

  static llvm::StringRef getOperationName() { return "fir.address"; }

  mlir::LogicalResult verify();

  // `opTy` will be wrapped with !fir.ref in `build`
  static void build(mlir::OpBuilder *builder, mlir::OperationState *result,
      llvm::ArrayRef<mlir::Value *> operands, mlir::Type opTy);
};

/// The "fir.alloca_expr" operation is used to allocate local temporary space
/// for object of a specified type.  The result of "fir.alloca_expr" is a
/// pointer to an uninitialized ssa-object of the designated type.  Together
/// with "fir.load_expr" and "fir.store_expr", these operations are used to
/// construct a mem-ssa representation, which will be lowered to the MLIR
/// standard dialect.
///
///   %86 = fir.alloca_expr(){name: "i"} : () -> !fir.ref<i64>
class AllocaExpr : public mlir::Op<AllocaExpr, mlir::OpTrait::ZeroOperands,
                       mlir::OpTrait::OneResult> {
public:
  using Op::Op;

  static llvm::StringRef getOperationName() { return "fir.alloca_expr"; }

  mlir::LogicalResult verify();

  // `opTy` will be wrapped with !fir.ref in `build`
  static void build(
      mlir::OpBuilder *builder, mlir::OperationState *result, mlir::Type opTy);
  static void build(mlir::OpBuilder *builder, mlir::OperationState *result,
      llvm::StringRef name, mlir::Type opTy);

  mlir::Type getAllocatedType();
  llvm::StringRef getExpr();
};

/// The "fir.allocmem" operation creates space for an object of a specified
/// type in heap memory.
///
///   %8 = "fir.allocmem"(){name: "j"} : !"fir.ref"<i32>
class AllocMemOp : public mlir::Op<AllocMemOp, mlir::OpTrait::ZeroOperands,
                       mlir::OpTrait::OneResult> {
public:
  using Op::Op;

  static llvm::StringRef getOperationName() { return "fir.allocmem"; }

  mlir::LogicalResult verify();
  static void build(
      mlir::OpBuilder *builder, mlir::OperationState *result, mlir::Type opTy);
  static void build(mlir::OpBuilder *builder, mlir::OperationState *result,
      llvm::StringRef name, mlir::Type opTy);

  mlir::Type getAllocatedType();
  llvm::StringRef getExpr();
};

/// The "fir.apply_expr" operation evaluates a Fortran expression to obtain the
/// value that the expression.  The following computes the product of the two
/// ssa-values %72 and %75.
///
///   %82 = fir.apply_expr(%72, %75){expr: "!0!_8 * !1!_8",
///            dict: {!0! -> %72, !1! -> %75}} : i64
class ApplyExpr : public mlir::Op<ApplyExpr, mlir::OpTrait::VariadicOperands,
                      mlir::OpTrait::OneResult> {
public:
  using Op::Op;

  static llvm::StringRef getOperationName() { return "fir.apply_expr"; }

  mlir::LogicalResult verify();

  static void build(mlir::OpBuilder *builder, mlir::OperationState *result,
      const SomeExpr *expr, const std::map<unsigned, void *> &dict,
      llvm::ArrayRef<mlir::Value *> operands, mlir::Type opTy);

  llvm::StringRef getExpr();
  SomeExpr *getRawExpr();
  std::map<unsigned, void *> *getDict();
};

class ConvertOp
  : public mlir::Op<ConvertOp, mlir::OpTrait::OneOperand,
        mlir::OpTrait::OneResult, mlir::OpTrait::HasNoSideEffect> {
public:
  using Op::Op;

  static llvm::StringRef getOperationName() { return "fir.convert"; }

  mlir::LogicalResult verify();
  static void build(mlir::OpBuilder *builder, mlir::OperationState *result,
      mlir::Value *val, mlir::Type toType);
};

class ExtractValueOp
  : public mlir::Op<ExtractValueOp, mlir::OpTrait::HasNoSideEffect,
        mlir::OpTrait::VariadicOperands, mlir::OpTrait::OneResult> {
public:
  using Op::Op;

  static llvm::StringRef getOperationName() { return "fir.extract_value"; }

  mlir::LogicalResult verify();
  static void build(mlir::OpBuilder *builder, mlir::OperationState *result,
      llvm::ArrayRef<mlir::Value *> args, mlir::Type opTy);
};

/// Encoding of a field (part reference)
/// Always yield a value of type `!fir.field_type`
class FieldValueOp
  : public mlir::Op<FieldValueOp, mlir::OpTrait::HasNoSideEffect,
        mlir::OpTrait::ZeroOperands, mlir::OpTrait::OneResult> {
public:
  using Op::Op;

  static llvm::StringRef getOperationName() { return "fir.field_value"; }

  mlir::LogicalResult verify();
  static void build(mlir::OpBuilder *builder, mlir::OperationState *result,
      llvm::StringRef name);
};

/// The "fir.freemem" operation frees the heap memory object referenced by
/// its argument.
///
///   %24 = "fir.freemem"(%19) : (!"fir.ref"<i32>) -> ()
class FreeMemOp : public mlir::Op<FreeMemOp, mlir::OpTrait::OneOperand,
                      mlir::OpTrait::ZeroResult> {
public:
  using Op::Op;

  static llvm::StringRef getOperationName() { return "fir.freemem"; }

  mlir::LogicalResult verify();
  static void build(mlir::OpBuilder *builder, mlir::OperationState *result,
      llvm::ArrayRef<mlir::Value *> args);

  mlir::Type getAllocatedType();
  llvm::StringRef getExpr();
};

/// The "fir.global_expr" operation is used to access global space for specified
/// object.  The result of "fir.global_expr" is a pointer to a persistent
/// ssa-object. The object has a type and is initialized. If no `value`
/// attribute is given the default initialization value is a bitstring of all
/// zero bits of the same bit-size as the type.
///
/// NB: MLIR will begin supporting global variables as first-class abstractions
/// at some point.
///
///   %86 = fir.global_expr(){name: "global", value: 4} : !fir.ref<i64>
class GlobalExpr : public mlir::Op<GlobalExpr, mlir::OpTrait::ZeroOperands,
                       mlir::OpTrait::OneResult> {
public:
  using Op::Op;

  static llvm::StringRef getOperationName() { return "fir.global_expr"; }

  mlir::LogicalResult verify();

  static void build(mlir::OpBuilder *builder, mlir::OperationState *result,
      llvm::StringRef name, mlir::Attribute value, mlir::Type opTy);

  llvm::StringRef getName();
};

class InsertValueOp
  : public mlir::Op<InsertValueOp, mlir::OpTrait::HasNoSideEffect,
        mlir::OpTrait::VariadicOperands, mlir::OpTrait::OneResult> {
public:
  using Op::Op;

  static llvm::StringRef getOperationName() { return "fir.insert_value"; }

  mlir::LogicalResult verify();
  static void build(mlir::OpBuilder *builder, mlir::OperationState *result,
      llvm::ArrayRef<mlir::Value *> args, mlir::Type opTy);
};

/// The "fir.load_expr" operation is used to load the result of a
/// "fir.locate_expr" operation from memory.  The result of "fir.load_expr"
/// is an ssa-object with a type.
///
///   %85 = fir.load_expr(%77) : (!fir.ref<tensor<10xf32>>) -> tensor<10xf32>
class LoadExpr : public mlir::Op<LoadExpr, mlir::OpTrait::OneOperand,
                     mlir::OpTrait::OneResult> {
public:
  using Op::Op;

  static llvm::StringRef getOperationName() { return "fir.load_expr"; }

  mlir::LogicalResult verify();

  static void build(mlir::OpBuilder *builder, mlir::OperationState *result,
      llvm::ArrayRef<mlir::Value *> operands, mlir::Type opTy);
  static void build(mlir::OpBuilder *builder, mlir::OperationState *result,
      mlir::Value *store);
};

/// The "fir.locate_expr" operation evaluates a Fortran expression to obtain the
/// address to which it refers.  The following computes the address of the value
/// in a Fortran array at position (1,4).
///
///   %82 = fir.locate_expr(%75){addr: "!0!(1,4)", dict: {!0! -> %75}} : i64
class LocateExpr : public mlir::Op<LocateExpr, mlir::OpTrait::VariadicOperands,
                       mlir::OpTrait::OneResult> {
public:
  using Op::Op;

  static llvm::StringRef getOperationName() { return "fir.locate_expr"; }

  mlir::LogicalResult verify();

  static void build(mlir::OpBuilder *builder, mlir::OperationState *result,
      const SomeExpr *expr, const std::map<unsigned, void *> &dict,
      llvm::ArrayRef<mlir::Value *> operands, mlir::Type opTy);

  llvm::StringRef getExpr();
  SomeExpr *getRawExpr();
  std::map<unsigned, void *> *getDict();
};

/// The "fir.select" operation is used to model a control flow analogous to a
/// switch statement in C, where matching is done based on an equivalence test
/// of the selector expression's value. The values tested against must be
/// constants of the same type as the selector or a default.
///
///   fir.select %56 : i32, (0, ^bb1), (1 , ^bb2(%45 : f32))
class SelectOp
  : public mlir::Op<SelectOp, mlir::OpTrait::AtLeastNOperands<1>::Impl,
        mlir::OpTrait::ZeroResult, mlir::OpTrait::IsTerminator> {
public:
  using Op::Op;

  // std::monostate is used to encode DEFAULT in all select op variants as well
  // as the "no match" condition which is a default DEFAULT to an empty
  // block. This implies that *all* select ops must have a std::monostate tuple
  // in the list to be correctly constructed.
  using Conditions = std::variant<mlir::ConstantOp *, std::monostate>;
  using BranchTuple =
      std::tuple<Conditions, mlir::Block *, llvm::ArrayRef<mlir::Value *>>;

  static llvm::StringRef getOperationName() { return "fir.select"; }

  mlir::LogicalResult verify();

  static void build(mlir::OpBuilder *builder, mlir::OperationState *result,
      mlir::Value *condition, llvm::ArrayRef<BranchTuple> tuples);
};

// CRTP in Op<> decls so did not encapsulate these within a subtype
#define SHARED_SELECT_CONSTRUCT \
  unsigned getNumConditions() { return getNumDest(); } \
  mlir::Value *getSelector() { return getOperand(0); } \
  mlir::Value *getCondition(unsigned dest) { \
    assert(dest < getNumConditions()); \
    return getOperand(getDestOperandIndex(dest) - 1); \
  } \
  unsigned getNumDest() { return getOperation()->getNumSuccessors(); } \
  mlir::Block *getDest(unsigned dest) { \
    return getOperation()->getSuccessor(dest); \
  } \
  unsigned getNumDestOperands(unsigned dest) { \
    return getOperation()->getNumSuccessorOperands(dest); \
  } \
  mlir::Value *getDestOperand(unsigned dest, unsigned i) { \
    assert(dest < getNumDest()); \
    assert(i < getNumDestOperands(dest)); \
    return getOperand(getDestOperandIndex(dest) + i); \
  } \
  unsigned getDestOperandIndex(unsigned dest) { \
    if (dest == 0) { \
      return 2; \
    } \
    return getNumDestOperands(dest) + getDestOperandIndex(dest - 1); \
  } \
  operand_iterator dest_operand_begin(unsigned dest) { \
    return operand_begin() + getDestOperandIndex(dest); \
  } \
  operand_iterator dest_operand_end(unsigned dest) { \
    return dest_operand_begin(dest) + getNumDestOperands(dest); \
  } \
  operand_range getDestOperands(unsigned dest) { \
    return {dest_operand_begin(dest), dest_operand_end(dest)}; \
  }

/// The "fir.select_case" operation is used to model a Fortran SELECT CASE
/// statement. This is analogous to a switch statement in C, but matching can be
/// on ranges and not just equivalence. Ranges can be one value, a range with
/// both an upper and lower bound, a range with only an upper bound, a range
/// with only a lower bound, or a default.
///
///   fir.select_case %57 : i32, ([1..5], ^bb1), (? , ^bb2)
class SelectCaseOp
  : public mlir::Op<SelectCaseOp, mlir::OpTrait::AtLeastNOperands<1>::Impl,
        mlir::OpTrait::ZeroResult, mlir::OpTrait::IsTerminator> {
public:
  using Op::Op;

  // assuming R1146 (case range well-formed) and C1149 (ranges cannot overlap)
  // will be checked by the front-end
  using Conditions = mlir::Value *;
  using BranchTuple =
      std::tuple<Conditions, mlir::Block *, llvm::ArrayRef<mlir::Value *>>;

  static llvm::StringRef getOperationName() { return "fir.select_case"; }

  mlir::LogicalResult verify();

  static void build(mlir::OpBuilder *builder, mlir::OperationState *result,
      mlir::Value *condition, llvm::ArrayRef<BranchTuple> tuples);

  SHARED_SELECT_CONSTRUCT
};

/// The "fir.select_rank" operation is used to model a Fortran SELECT RANK
/// statement. This is analogous to a switch statement in C, but matching is
/// done on the rank of the selector expression, which can be a non-negative
/// integer, assumed shape, or a default.
///
///   %58 = call @fortran_get_rank(%54) : (!fir.array<? x f32>) -> i32
///   fir.select_rank %58 : i32, (4, ^bb4), (*, ^bb5), (?, ^bb6)
class SelectRankOp
  : public mlir::Op<SelectRankOp, mlir::OpTrait::AtLeastNOperands<1>::Impl,
        mlir::OpTrait::ZeroResult, mlir::OpTrait::IsTerminator> {
public:
  using Op::Op;

  // assuming that the runtime will encode selectors with assumed shape as a
  // sentinel with a negative integer value
  using Conditions = mlir::Value *;
  using BranchTuple =
      std::tuple<Conditions, mlir::Block *, llvm::ArrayRef<mlir::Value *>>;

  static llvm::StringRef getOperationName() { return "fir.select_rank"; }

  mlir::LogicalResult verify();

  static void build(mlir::OpBuilder *builder, mlir::OperationState *result,
      mlir::Value *condition, llvm::ArrayRef<BranchTuple> tuples);

  SHARED_SELECT_CONSTRUCT
};

/// The "fir.select_type" operation is used to model a Fortran SELECT TYPE
/// statement. This is analogous to a switch statement in C, but matching is
/// done on the type of the selector expression, which can be an exact type, a
/// most-recent ancestor class in an inheritance hierarchy, or a default.
///
///   %td3 = call @fortran_get_typedesc(%foo) : !foo_type -> !typedesc
///   fir.select_type %td3 : !typedesc, (%td1, ^bb6), (<:%td2, ^bb7), (?, ^bb8)
class SelectTypeOp
  : public mlir::Op<SelectTypeOp, mlir::OpTrait::AtLeastNOperands<1>::Impl,
        mlir::OpTrait::ZeroResult, mlir::OpTrait::IsTerminator> {
public:
  using Op::Op;

  // assuming that tuple list is partially ordered by the front-end such that a
  // subclass will always come earlier in the list then its superclass(es) from
  // the type hierarchy
  using Conditions = mlir::Value *;
  using BranchTuple =
      std::tuple<Conditions, mlir::Block *, llvm::ArrayRef<mlir::Value *>>;

  static llvm::StringRef getOperationName() { return "fir.select_type"; }

  mlir::LogicalResult verify();

  static void build(mlir::OpBuilder *builder, mlir::OperationState *result,
      mlir::Value *condition, llvm::ArrayRef<BranchTuple> tuples);

  SHARED_SELECT_CONSTRUCT
};

/// The "fir.store_expr" operation is used to store the result of a
/// "fir.apply_expr" operation to memory specified by the result of a
/// "fir.locate_expr" operation.
///
///   %83 = fir.store_expr(%75, %76) : (i64, !fir.ref<i64>) -> ()
class StoreExpr : public mlir::Op<StoreExpr, mlir::OpTrait::NOperands<2>::Impl,
                      mlir::OpTrait::ZeroResult> {
public:
  using Op::Op;

  static llvm::StringRef getOperationName() { return "fir.store_expr"; }

  mlir::LogicalResult verify();

  static void build(mlir::OpBuilder *builder, mlir::OperationState *result,
      llvm::ArrayRef<mlir::Value *> operands);
  static void build(mlir::OpBuilder *builder, mlir::OperationState *result,
      mlir::Value *value, mlir::Value *store);
};

/// The "fir.undefined" operation is used to create an LLVM::UndefOp.
class UndefOp : public mlir::Op<UndefOp, mlir::OpTrait::HasNoSideEffect,
                    mlir::OpTrait::ZeroOperands, mlir::OpTrait::OneResult> {
public:
  using Op::Op;

  static llvm::StringRef getOperationName() { return "fir.undefined"; }

  mlir::LogicalResult verify();

  static void build(
      mlir::OpBuilder *builder, mlir::OperationState *result, mlir::Type opTy);
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

  static llvm::StringRef getOperationName() { return "fir.unreachable"; }

  mlir::LogicalResult verify();

  static void build(mlir::OpBuilder *builder, mlir::OperationState *result);
};

}  // Fortran::mlbridge

#endif  // FORTRAN_LIB_MLBRIDGE_FIR_DIALECT_H
