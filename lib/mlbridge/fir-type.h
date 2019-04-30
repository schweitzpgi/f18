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

#ifndef FORTRAN_MLBRIDGE_FORTRAN_IR_TYPE_H_
#define FORTRAN_MLBRIDGE_FORTRAN_IR_TYPE_H_

#include "mlir/IR/Types.h"
#include <variant>

namespace Fortran::mlbridge {

namespace detail {
struct FIRLogicalTypeStorage;
struct FIRCharacterTypeStorage;
struct FIRTupleTypeStorage;
struct FIRReferenceTypeStorage;
struct FIRSequenceTypeStorage;
}

/// LLVM-style RTTI: one entry per subclass to allow dyn_cast/isa.
enum FIRTypeKind {
  // The enum starts at the range reserved for this dialect.
  FIR_TYPE = mlir::Type::FIRST_FIR_TYPE,
  FIR_LOGICAL,
  FIR_CHARACTER,
  FIR_TUPLE,
  FIR_REFERENCE,
  FIR_SEQUENCE,
};

// Intrinsic Fortran type mappings
// INTEGER*k      --> `i` k*8  where k in {1, 2, 4, 8, 16}
// REAL*k         --> `f` k*8  where k in {2, 4, 8}
// REAL*3         --> `bf16`
// COMPLEX*k      --> `complex<f` k*8 `>`  where k in {2, 4, 8}
// COMPLEX*3      --> `complex<bf16>`

/// LOGICAL*k
/// These will be lowered to `i8`, etc.
class FIRLogicalType : public mlir::Type::TypeBase<FIRLogicalType, mlir::Type,
                           detail::FIRLogicalTypeStorage> {
public:
  static FIRLogicalType get(mlir::MLIRContext *ctxt, unsigned kind);
  static bool kindof(unsigned kind) { return kind == FIRTypeKind::FIR_LOGICAL; }
};

/// CHARACTER*k(n)
/// These will be lowered to `[` n ` x i` k*8 `]` if n is constant
///                       or `{i` k*8 `*, i64}`   if n is variable
class FIRCharacterType : public mlir::Type::TypeBase<FIRCharacterType,
                             mlir::Type, detail::FIRCharacterTypeStorage> {
public:
  static FIRCharacterType get(mlir::MLIRContext *ctxt, unsigned kind);
  static bool kindof(unsigned kind) {
    return kind == FIRTypeKind::FIR_CHARACTER;
  }
};

/// Derived Fortran types:
/// TYPE :: name ... END TYPE name
/// These will be lowered to `tuple <` list `>` (standard dialect)
/// and then `{` type-list `}` (LLVM-IR)
class FIRTupleType : public mlir::Type::TypeBase<FIRTupleType, mlir::Type,
                         detail::FIRTupleTypeStorage> {
public:
  static FIRTupleType get(mlir::MLIRContext *ctxt, llvm::StringRef name);
  static FIRTupleType get(mlir::MLIRContext *ctxt, llvm::StringRef name,
      llvm::ArrayRef<mlir::Type> elementTypes);
  static bool kindof(unsigned kind) { return kind == FIRTypeKind::FIR_TUPLE; }
};

/// FIR support types
/// Pointer-like objects
/// These will be lowered to   type `*` (LLVM-IR)
class FIRReferenceType : public mlir::Type::TypeBase<FIRReferenceType,
                             mlir::Type, detail::FIRReferenceTypeStorage> {
public:
  static FIRReferenceType get(mlir::MLIRContext *ctxt, mlir::Type elementType);
  static bool kindof(unsigned kind) {
    return kind == FIRTypeKind::FIR_REFERENCE;
  }
};

// function types  -->  `(` type-list `) ->` `(`? result-type-list `)`?
// and in LLVM-IR  `{`? result-type-list `}`? `(` type-list `)`

/// Sequence-like objects
/// These will be lowered to  (`tensor` | `memref`) `<` dim-list `x` type `>`
class FIRSequenceType : public mlir::Type::TypeBase<FIRSequenceType, mlir::Type,
                            detail::FIRSequenceTypeStorage> {
public:
  struct Unknown {};
  struct BoundInfo {
    int lower;
    int count;
    int stride;
  };
  struct Extent {
    std::variant<Unknown, BoundInfo> u;
  };
  struct Shape {
    std::variant<Unknown, std::vector<Extent>> u;
    bool operator==(const Shape &) const;
    size_t hash_value() const;
  };

  BoundInfo simpleBoundInfo(int size) { return {1, size, 1}; }

  static FIRSequenceType get(
      mlir::MLIRContext *ctxt, const Shape &shape, mlir::Type elementType);
  static bool kindof(unsigned kind) {
    return kind == FIRTypeKind::FIR_SEQUENCE;
  }
};

}  // Fortran::mlbridge

#endif  // FORTRAN_MLBRIDGE_FORTRAN_IR_TYPE_H_
