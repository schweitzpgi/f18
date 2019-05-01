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

namespace llvm {
class StringRef;
template<typename> class ArrayRef;
}

namespace Fortran::mlbridge {

// In the Fortran::mlbridge namespace, the code will default follow the
// LLVM/MLIR coding standards

namespace detail {
struct FIRRealTypeStorage;
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
  FIR_REAL,
  FIR_LOGICAL,
  FIR_CHARACTER,
  FIR_TUPLE,
  FIR_REFERENCE,
  FIR_SEQUENCE,
};

/// CHARACTER*k(n)  --> `char {kind:` k*8 `}`
/// These will be lowered to `[` n ` x i` k*8 `]` if n is constant
///                       or `{i` k*8 `*, i64}`   if n is variable
class FIRCharacterType : public mlir::Type::TypeBase<FIRCharacterType,
                             mlir::Type, detail::FIRCharacterTypeStorage> {
public:
  using Base::Base;
  static FIRCharacterType get(mlir::MLIRContext *ctxt, int kind);
  static bool kindof(unsigned kind) {
    return kind == FIRTypeKind::FIR_CHARACTER;
  }

  int getSizeInBits() const;
  int getFKind() const;
};

// COMPLEX*k      --> `complex<f` k*8 `>`                  where k in {2, 4, 8}
///                   `complex<!fir<"real {kind:` k*8 `}">>`  otherwise
// COMPLEX*3      --> `complex<bf16>`

// function types  -->  `(` type-list `) ->` `(`? result-type-list `)`?
// and in LLVM-IR  `{`? result-type-list `}`? `(` type-list `)`

// Intrinsic Fortran type mappings
// INTEGER*k      --> `i` k*8  where k in {1, 2, 4, 8, 16}

/// LOGICAL*k     --> `logical {kind:` k*8 `}`
/// These will be lowered to `i1`, `i8`, etc.
class FIRLogicalType : public mlir::Type::TypeBase<FIRLogicalType, mlir::Type,
                           detail::FIRLogicalTypeStorage> {
public:
  using Base::Base;
  static FIRLogicalType get(mlir::MLIRContext *ctxt, int kind);
  static bool kindof(unsigned kind) { return kind == FIRTypeKind::FIR_LOGICAL; }

  int getSizeInBits() const;
  int getFKind() const;
};

/// REAL*k         --> `f` k*8                where k in {2, 4, 8}
///                    `real {kind:` k*8 `}`  otherwise
/// REAL*3         --> `bf16`                 (not 3/4 precision)
class FIRRealType : public mlir::Type::TypeBase<FIRRealType, mlir::Type,
                        detail::FIRRealTypeStorage> {
public:
  using Base::Base;
  static FIRRealType get(mlir::MLIRContext *ctxt, int kind);
  static bool kindof(unsigned kind) { return kind == FIRTypeKind::FIR_REAL; }

  int getSizeInBits() const;
  int getFKind() const;
};

/// FIR support types
/// Pointer-like objects  -->  `ref` `<` type `>`
/// These will be lowered to   type `*` (LLVM-IR)
class FIRReferenceType : public mlir::Type::TypeBase<FIRReferenceType,
                             mlir::Type, detail::FIRReferenceTypeStorage> {
public:
  using Base::Base;
  static FIRReferenceType get(mlir::Type elementType);
  static bool kindof(unsigned kind) {
    return kind == FIRTypeKind::FIR_REFERENCE;
  }

  mlir::Type getEleTy() const;
};

/// Sequence-like objects
/// The FIR type:
///      --> bounds ::= (lo `:`)? extent (`;` stride)? | `?`
///          `array` `<` bounds (`,` bounds)* `x` type `>`
/// These will be lowered to
///      --> (`tensor` | `memref`) `<` dim-list `x` type `>`
class FIRSequenceType : public mlir::Type::TypeBase<FIRSequenceType, mlir::Type,
                            detail::FIRSequenceTypeStorage> {
public:
  using Base::Base;
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

  static FIRSequenceType get(const Shape &shape, mlir::Type elementType);
  static bool kindof(unsigned kind) {
    return kind == FIRTypeKind::FIR_SEQUENCE;
  }
};

/// Derived Fortran types:
/// TYPE :: name ... END TYPE name
/// The FIR type:
///     --> `type` name `<` type (`,` type)* `>` (`{` kind (`,` kind)* `}`)?
///         `type<` type-list `> {name:` name , kinds: ` kind-list `}`
/// These will be lowered to `tuple <` list `>` (standard dialect)
/// and then `{` type-list `}` (LLVM-IR)
class FIRTupleType : public mlir::Type::TypeBase<FIRTupleType, mlir::Type,
                         detail::FIRTupleTypeStorage> {
public:
  using Base::Base;
  static FIRTupleType get(mlir::MLIRContext *ctxt, llvm::StringRef name);
  static FIRTupleType get(mlir::MLIRContext *ctxt, llvm::StringRef name,
      llvm::ArrayRef<mlir::Type> elementTypes);  // TODO: add kinds
  static bool kindof(unsigned kind) { return kind == FIRTypeKind::FIR_TUPLE; }
};

}  // Fortran::mlbridge

#endif  // FORTRAN_MLBRIDGE_FORTRAN_IR_TYPE_H_
