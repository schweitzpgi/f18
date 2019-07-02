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

#ifndef FORTRAN_LIB_MLBRIDGE_FIR_TYPE_H
#define FORTRAN_LIB_MLBRIDGE_FIR_TYPE_H

#include "mlir/IR/Types.h"
#include <variant>

namespace llvm {
class StringRef;
template<typename> class ArrayRef;
class hash_code;
}

namespace Fortran::mlbridge {

class FIROpsDialect;

using KindTy = int;

// In the Fortran::mlbridge namespace, the code will default follow the
// LLVM/MLIR coding standards

namespace detail {
struct FIRBoxTypeStorage;
struct FIRBoxCharTypeStorage;
struct FIRBoxProcTypeStorage;
struct FIRCharacterTypeStorage;
struct FIRDimsTypeStorage;
struct FIRFieldTypeStorage;
struct FIRLogicalTypeStorage;
struct FIRRealTypeStorage;
struct FIRReferenceTypeStorage;
struct FIRSequenceTypeStorage;
struct FIRTupleTypeStorage;
struct FIRTypeDescTypeStorage;
}

// COMPLEX*k      --> `complex<f` k*8 `>`            where k in {2, 4, 8}
//                    `complex<!fir.real<` k*8 `>>`  otherwise
// COMPLEX*3      --> `complex<bf16>`

// function types  -->  `(` type-list `) ->` `(`? result-type-list `)`?
// and in LLVM-IR  `{`? result-type-list `}`? `(` type-list `)`

// Intrinsic Fortran type mappings
// INTEGER*k      --> `i` k*8    where k in {1, 2, 4, 8, 16}

/// LLVM-style RTTI: one entry per subclass to allow dyn_cast/isa.
enum FIRTypeKind {
  // The enum starts at the range reserved for this dialect.
  FIR_TYPE = mlir::Type::FIRST_FIR_TYPE,
  FIR_BOX,
  FIR_BOXCHAR,
  FIR_BOXPROC,
  FIR_CHARACTER,
  FIR_DIMS,
  FIR_FIELD,
  FIR_LOGICAL,
  FIR_REAL,
  FIR_REFERENCE,
  FIR_SEQUENCE,
  FIR_TUPLE,
  FIR_TYPEDESC,
};

/// CHARACTER*k(n)  --> `!fir.char<` k `>`
/// These will be lowered to `[` n ` x i` k*8 `]` if n is constant
///                       or `{i` k*8 `*, i64}`   if n is variable
class FIRCharacterType : public mlir::Type::TypeBase<FIRCharacterType,
                             mlir::Type, detail::FIRCharacterTypeStorage> {
public:
  using Base::Base;
  static FIRCharacterType get(mlir::MLIRContext *ctxt, KindTy kind);
  static bool kindof(unsigned kind) {
    return kind == FIRTypeKind::FIR_CHARACTER;
  }

  int getSizeInBits() const;
  KindTy getFKind() const;
};

/// A field in a PART-REF represented as a FieldOp. A FieldOp has type
/// FIRFieldType.
///          --> `!fir.field`
/// lowering to an integer
class FIRFieldType : public mlir::Type::TypeBase<FIRFieldType, mlir::Type,
                         detail::FIRFieldTypeStorage> {
public:
  using Base::Base;
  static FIRFieldType get(mlir::MLIRContext *ctxt, KindTy _ = 0);
  static bool kindof(unsigned kind) { return kind == FIRTypeKind::FIR_FIELD; }
};

/// LOGICAL*k     --> `logical<` k `>`
/// These will be lowered to `i1`, `i8`, etc.
class FIRLogicalType : public mlir::Type::TypeBase<FIRLogicalType, mlir::Type,
                           detail::FIRLogicalTypeStorage> {
public:
  using Base::Base;
  static FIRLogicalType get(mlir::MLIRContext *ctxt, KindTy kind);
  static bool kindof(unsigned kind) { return kind == FIRTypeKind::FIR_LOGICAL; }

  int getSizeInBits() const;
  KindTy getFKind() const;
};

/// REAL*k         --> `f` k*8               where k in {2, 4, 8}
///                    `!fir.real<` k `>`  otherwise
/// REAL*3         --> `bf16`                (not 3/4 precision)
class FIRRealType : public mlir::Type::TypeBase<FIRRealType, mlir::Type,
                        detail::FIRRealTypeStorage> {
public:
  using Base::Base;
  static FIRRealType get(mlir::MLIRContext *ctxt, KindTy kind);
  static bool kindof(unsigned kind) { return kind == FIRTypeKind::FIR_REAL; }

  int getSizeInBits() const;
  KindTy getFKind() const;
};

// FIR support types

/// Boxed object (a Fortran descriptor)
///
/// A boxed object is a Fortran descriptor. An descriptor is a tuple of
/// information that describes objects that are arrays, allocatables, or
/// pointers. The tuple is of the form: (base-addr, type-descriptor, attributes,
/// dims).  Element-size and rank can be derived.
class FIRBoxType : public mlir::Type::TypeBase<FIRBoxType, mlir::Type,
                       detail::FIRBoxTypeStorage> {
public:
  using Base::Base;
  static FIRBoxType get(mlir::Type eleTy);
  static bool kindof(unsigned kind) { return kind == FIRTypeKind::FIR_BOX; }
  mlir::Type getEleTy() const;
};

/// Boxed CHARACTER object type
///
/// A boxed character object is an object with a pair of values:
/// (pointer-of-buffer, length-of-buffer)
class FIRBoxCharType : public mlir::Type::TypeBase<FIRBoxCharType, mlir::Type,
                           detail::FIRBoxCharTypeStorage> {
public:
  using Base::Base;
  static FIRBoxCharType get(FIRCharacterType charTy);
  static bool kindof(unsigned kind) { return kind == FIRTypeKind::FIR_BOXCHAR; }
  FIRCharacterType getEleTy() const;
};

/// Boxed PROCEDURE POINTER object type
///
/// A boxed procedure pointer object is an object with a pair of values:
/// (pointer-to-function, opt-pointer-to-host-context)
class FIRBoxProcType : public mlir::Type::TypeBase<FIRBoxProcType, mlir::Type,
                           detail::FIRBoxProcTypeStorage> {
public:
  using Base::Base;
  static FIRBoxProcType get(mlir::Type eleTy);
  static bool kindof(unsigned kind) { return kind == FIRTypeKind::FIR_BOXPROC; }
  mlir::Type getEleTy() const;
};

class FIRDimsType : public mlir::Type::TypeBase<FIRDimsType, mlir::Type,
                        detail::FIRDimsTypeStorage> {
public:
  using Base::Base;
  static FIRDimsType get(mlir::MLIRContext *ctx, unsigned rank);
  static bool kindof(unsigned kind) { return kind == FIRTypeKind::FIR_DIMS; }
  unsigned getRank() const;
};

/// Pointer-like objects
///
///  REAL, POINTER :: ptr
///
/// The FIR type:
///      -->  `!fir.ref` `<` type `>`
///
/// These will be lowered to
///      --> `memref` `<` type `>`
///      --> type `*` (LLVM-IR)
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

/// Sequence-like object values
///
///  REAL :: A(n,2:m)
///
/// The FIR type:
///      --> bounds ::= lo extent stride | `?`
///          `!fir.array` `<` bounds (`,` bounds)* `:` type `>`
///
/// These will be lowered to
///      --> (`tensor` | `memref`) `<` dim-list `x` type `>`
///      --> [n x [... type ...] ] (LLVM-IR)
class FIRSequenceType : public mlir::Type::TypeBase<FIRSequenceType, mlir::Type,
                            detail::FIRSequenceTypeStorage> {
public:
  using Base::Base;
  struct Unknown {};
  struct BoundInfo {
    int64_t lower;
    int64_t count;
    int64_t stride;
  };
  using Extent = std::variant<Unknown, BoundInfo>;
  using Bounds = std::vector<Extent>;
  using Shape = std::variant<Unknown, Bounds>;

  mlir::Type getEleTy() const;
  Shape getShape() const;

  static FIRSequenceType get(const Shape &shape, mlir::Type elementType);
  static bool kindof(unsigned kind) {
    return kind == FIRTypeKind::FIR_SEQUENCE;
  }
};

bool operator==(const FIRSequenceType::Shape &, const FIRSequenceType::Shape &);
llvm::hash_code hash_value(const FIRSequenceType::Extent &);
llvm::hash_code hash_value(const FIRSequenceType::Shape &);

/// Derived Fortran types
///
///  TYPE :: name ... END TYPE name
///
/// The FIR type:
///    --> `!fir.type` `<` name
///              (`,` `{` str `:` type (`,` str `:` type)* `}`)?
///              (`,` `[` str `:` kind (`,` str `:` kind)* `]`)? '>'
///
/// These will be lowered to `tuple <` list `>` (standard dialect)
/// and then `{` type-list `}` (LLVM-IR)
class FIRTupleType : public mlir::Type::TypeBase<FIRTupleType, mlir::Type,
                         detail::FIRTupleTypeStorage> {
public:
  using Base::Base;
  using TypePair = std::pair<std::string, mlir::Type>;
  using TypeList = std::vector<TypePair>;
  using KindPair = std::pair<std::string, KindTy>;
  using KindList = std::vector<KindPair>;

  llvm::StringRef getName();
  TypeList getTypeList();
  KindList getKindList();

  static FIRTupleType get(mlir::MLIRContext *ctxt, llvm::StringRef name,
      llvm::ArrayRef<KindPair> kindList = {},
      llvm::ArrayRef<TypePair> typeList = {});
  static bool kindof(unsigned kind) { return kind == FIRTypeKind::FIR_TUPLE; }
};

/// The type of a type descriptor object
///
/// Each Fortran type has a type descriptor object, which may (or may not) be
/// reified. A type descriptor object is a constant "dope vector" that may be
/// used by the Fortran runtime. The type of that type descriptor object is a
/// FIRTypeDescType. A FIRTypeDescType has one argument, the Type that the
/// descriptor describes.
///
/// The FIR type:
///     ---> `!fir.typedesc` type
///
/// Code gen can render an object of this type into the appropriate LLVM
/// constant(s)
class FIRTypeDescType : public mlir::Type::TypeBase<FIRTypeDescType, mlir::Type,
                            detail::FIRTypeDescTypeStorage> {
public:
  using Base::Base;
  static FIRTypeDescType get(mlir::Type ofType);
  static bool kindof(unsigned kind) {
    return kind == FIRTypeKind::FIR_TYPEDESC;
  }
  mlir::Type getOfTy() const;
};

mlir::Type parseFirType(
    FIROpsDialect *dialect, llvm::StringRef rawData, mlir::Location loc);

}  // Fortran::mlbridge

#endif  // FORTRAN_LIB_MLBRIDGE_FIR_TYPE_H
