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

#include "fir-type.h"

namespace Fortran::mlbridge {
namespace detail {

// `REAL` storage (for reals of unsupported sizes)
struct FIRRealTypeStorage : public mlir::TypeStorage {
  using KeyTy = int;

  static unsigned hashKey(const KeyTy &key) { return llvm::hash_combine(key); }

  bool operator==(const KeyTy &key) const { return key == getFKind(); }

  static FIRRealTypeStorage *construct(
      mlir::TypeStorageAllocator &allocator, int kind) {
    auto *storage = allocator.allocate<FIRRealTypeStorage>();
    return new (storage) FIRRealTypeStorage{kind};
  }

  int getFKind() const { return kind; }

protected:
  int kind;

private:
  FIRRealTypeStorage() = delete;
  explicit FIRRealTypeStorage(int kind) : kind{kind} {}
};

// `LOGICAL` storage
struct FIRLogicalTypeStorage : public mlir::TypeStorage {
  using KeyTy = int;

  static unsigned hashKey(const KeyTy &key) { return llvm::hash_combine(key); }

  bool operator==(const KeyTy &key) const { return key == getFKind(); }

  static FIRLogicalTypeStorage *construct(
      mlir::TypeStorageAllocator &allocator, int kind) {
    auto *storage = allocator.allocate<FIRLogicalTypeStorage>();
    return new (storage) FIRLogicalTypeStorage{kind};
  }

  int getFKind() const { return kind; }

protected:
  int kind;

private:
  FIRLogicalTypeStorage() = delete;
  explicit FIRLogicalTypeStorage(int kind) : kind{kind} {}
};

// `CHARACTER` storage
struct FIRCharacterTypeStorage : public mlir::TypeStorage {
  using KeyTy = int;

  static unsigned hashKey(const KeyTy &key) { return llvm::hash_combine(key); }

  bool operator==(const KeyTy &key) const { return key == getFKind(); }

  static FIRCharacterTypeStorage *construct(
      mlir::TypeStorageAllocator &allocator, int kind) {
    auto *storage = allocator.allocate<FIRCharacterTypeStorage>();
    return new (storage) FIRCharacterTypeStorage{kind};
  }

  int getFKind() const { return kind; }

protected:
  int kind;

private:
  FIRCharacterTypeStorage() = delete;
  explicit FIRCharacterTypeStorage(int kind) : kind{kind} {}
};

// `TYPE :: name` storage
struct FIRTupleTypeStorage : public mlir::TypeStorage {
  using KeyTy = std::tuple<llvm::StringRef, llvm::ArrayRef<int>,
      llvm::ArrayRef<mlir::Type>>;

  static unsigned hashKey(const KeyTy &key) {
    const llvm::ArrayRef<int> &vec = std::get<llvm::ArrayRef<int>>(key);
    return llvm::hash_combine(std::get<llvm::StringRef>(key).str(),
        llvm::hash_combine_range(vec.begin(), vec.end()));
  }

  bool operator==(const KeyTy &key) const {
    return std::get<llvm::StringRef>(key) == getName() &&
        std::get<llvm::ArrayRef<int>>(key) == getFKinds();
  }

  static FIRTupleTypeStorage *construct(
      mlir::TypeStorageAllocator &allocator, const KeyTy &key) {
    auto *storage = allocator.allocate<FIRTupleTypeStorage>();
    auto &name = std::get<llvm::StringRef>(key);
    auto &kinds = std::get<llvm::ArrayRef<int>>(key);
    auto &members = std::get<llvm::ArrayRef<mlir::Type>>(key);
    return new (storage) FIRTupleTypeStorage{name, kinds, members};
  }

  llvm::StringRef getName() const { return name; }
  llvm::ArrayRef<int> getFKinds() const { return kinds; }
  void setMembers(llvm::ArrayRef<mlir::Type> mems) { members = mems; }
  llvm::ArrayRef<mlir::Type> getMembers() const { return members; }

protected:
  std::string name;
  std::vector<int> kinds;
  std::vector<mlir::Type> members;

private:
  FIRTupleTypeStorage() = delete;
  explicit FIRTupleTypeStorage(llvm::StringRef name, llvm::ArrayRef<int> kinds,
      llvm::ArrayRef<mlir::Type> members)
    : name{name}, kinds{kinds}, members{members} {}
};

// Pointer-like object storage
struct FIRReferenceTypeStorage : public mlir::TypeStorage {
  using KeyTy = mlir::Type;

  static unsigned hashKey(const KeyTy &key) { return llvm::hash_combine(key); }

  bool operator==(const KeyTy &key) const { return key == getElementType(); }

  static FIRReferenceTypeStorage *construct(
      mlir::TypeStorageAllocator &allocator, mlir::Type eleTy) {
    assert(eleTy && "element type is null");
    auto *storage = allocator.allocate<FIRReferenceTypeStorage>();
    return new (storage) FIRReferenceTypeStorage{eleTy};
  }

  mlir::Type getElementType() const { return eleTy; }

protected:
  mlir::Type eleTy;

private:
  FIRReferenceTypeStorage() = delete;
  explicit FIRReferenceTypeStorage(mlir::Type eleTy) : eleTy{eleTy} {}
};

// Sequence-like object storage
struct FIRSequenceTypeStorage : public mlir::TypeStorage {
  using KeyTy = std::pair<FIRSequenceType::Shape, mlir::Type>;

  static unsigned hashKey(const KeyTy &key) {
    auto shapeHash{std::get<FIRSequenceType::Shape>(key).hash_value()};
    return llvm::hash_combine(shapeHash, std::get<mlir::Type>(key));
  }

  bool operator==(const KeyTy &key) const {
    return key == KeyTy{getShape(), getElementType()};
  }

  static FIRSequenceTypeStorage *construct(
      mlir::TypeStorageAllocator &allocator, const KeyTy &key) {
    auto *storage = allocator.allocate<FIRSequenceTypeStorage>();
    return new (storage) FIRSequenceTypeStorage{key.first, key.second};
  }

  FIRSequenceType::Shape getShape() const { return shape; }
  mlir::Type getElementType() const { return eleTy; }

protected:
  FIRSequenceType::Shape shape;
  mlir::Type eleTy;

private:
  FIRSequenceTypeStorage() = delete;
  explicit FIRSequenceTypeStorage(
      const FIRSequenceType::Shape &shape, mlir::Type eleTy)
    : shape{shape}, eleTy{eleTy} {}
};
}  // detail

FIRRealType FIRRealType::get(mlir::MLIRContext *ctxt, int kind) {
  return Base::get(ctxt, FIR_REAL, kind * 8);
}

int FIRRealType::getSizeInBits() const { return getImpl()->getFKind(); }
int FIRRealType::getFKind() const { return getSizeInBits() / 8; }

FIRLogicalType FIRLogicalType::get(mlir::MLIRContext *ctxt, int kind) {
  return Base::get(ctxt, FIR_LOGICAL, kind * 8);
}

int FIRLogicalType::getSizeInBits() const { return getImpl()->getFKind(); }
int FIRLogicalType::getFKind() const { return getSizeInBits() / 8; }

FIRCharacterType FIRCharacterType::get(mlir::MLIRContext *ctxt, int kind) {
  return Base::get(ctxt, FIR_CHARACTER, kind * 8);
}

int FIRCharacterType::getSizeInBits() const { return getImpl()->getFKind(); }
int FIRCharacterType::getFKind() const { return getSizeInBits() / 8; }

bool FIRSequenceType::Shape::operator==(
    const FIRSequenceType::Shape &shape) const {
  return false;  // FIXME
}

size_t FIRSequenceType::Shape::hash_value() const {
  return 0;  // FIXME
}

FIRSequenceType FIRSequenceType::get(
    const Shape &shape, mlir::Type elementType) {
  auto *ctxt = elementType.getContext();
  return Base::get(ctxt, FIR_SEQUENCE, shape, elementType);
}

FIRReferenceType FIRReferenceType::get(mlir::Type elementType) {
  return Base::get(elementType.getContext(), FIR_REFERENCE, elementType);
}

mlir::Type FIRReferenceType::getEleTy() const {
  return getImpl()->getElementType();
}

FIRTupleType FIRTupleType::get(mlir::MLIRContext *ctxt, llvm::StringRef name,
    llvm::ArrayRef<mlir::Type> eleTypes) {
  llvm::SmallVector<int, 1> empty;  // FIXME
  return Base::get(ctxt, FIR_TUPLE, name, empty, eleTypes);
}

FIRTupleType FIRTupleType::get(mlir::MLIRContext *ctxt, llvm::StringRef name) {
  llvm::SmallVector<int, 1> empty;  // FIXME
  llvm::SmallVector<mlir::Type, 1> noTypes;
  return Base::get(ctxt, FIR_TUPLE, name, empty, noTypes);
}

}  // mlbridge
