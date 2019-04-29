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

// `LOGICAL` storage
struct FIRLogicalTypeStorage : public mlir::TypeStorage {
  using KeyTy = int;

  static unsigned hashKey(const KeyTy &key) { return llvm::hash_combine(key); }

  bool operator==(const KeyTy &key) const { return key == getFKind(); }

  static FIRLogicalTypeStorage *construct(
      mlir::TypeStorageAllocator &allocator, llvm::StringRef name, int kind) {
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
      mlir::TypeStorageAllocator &allocator, llvm::StringRef name, int kind) {
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
  using KeyTy =
      std::tuple<llvm::StringRef, std::vector<int>, std::vector<mlir::Type>>;

  static unsigned hashKey(const KeyTy &key) {
    const std::vector<int> &vec = std::get<std::vector<int>>(key);
    return llvm::hash_combine(std::get<llvm::StringRef>(key).str(),
        llvm::hash_combine_range(vec.begin(), vec.end()));
  }

  bool operator==(const KeyTy &key) const {
    return std::get<llvm::StringRef>(key) == getName() &&
        std::get<std::vector<int>>(key) == getFKinds();
  }

  static FIRTupleTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
      llvm::StringRef name, llvm::ArrayRef<int> kinds) {
    auto *storage = allocator.allocate<FIRTupleTypeStorage>();
    return new (storage) FIRTupleTypeStorage{name, kinds};
  }

  llvm::StringRef getName() const { return name; }
  std::vector<int> getFKinds() const { return kinds; }
  void setMembers(llvm::ArrayRef<mlir::Type> mems) { members = mems; }
  std::vector<mlir::Type> getMembers() const { return members; }

protected:
  std::string name;
  std::vector<int> kinds;
  std::vector<mlir::Type> members;

private:
  FIRTupleTypeStorage() = delete;
  explicit FIRTupleTypeStorage(llvm::StringRef name, llvm::ArrayRef<int> kinds)
    : name{name}, kinds{kinds} {}
};

// Pointer-like object storage
struct FIRReferenceTypeStorage : public mlir::TypeStorage {
  using KeyTy = mlir::Type;

  static unsigned hashKey(const KeyTy &key) { return llvm::hash_combine(key); }

  bool operator==(const KeyTy &key) const { return key == getElementType(); }

  static FIRReferenceTypeStorage *construct(
      mlir::TypeStorageAllocator &allocator, mlir::Type eleTy) {
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
  using KeyTy = std::tuple<FIRSequenceType::Shape, mlir::Type>;

  static unsigned hashKey(const KeyTy &key) {
    auto shapeHash{std::get<FIRSequenceType::Shape>(key).hash_value()};
    return llvm::hash_combine(shapeHash, std::get<mlir::Type>(key));
  }

  bool operator==(const KeyTy &key) const {
    return (std::get<FIRSequenceType::Shape>(key) == getShape()) ||
        (std::get<mlir::Type>(key) == getElementType());
  }

  static FIRSequenceTypeStorage *construct(
      mlir::TypeStorageAllocator &allocator,
      const FIRSequenceType::Shape &shape, mlir::Type eleTy) {
    auto *storage = allocator.allocate<FIRSequenceTypeStorage>();
    return new (storage) FIRSequenceTypeStorage{shape, eleTy};
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

bool FIRSequenceType::Shape::operator==(
    const FIRSequenceType::Shape &shape) const {
  return false;  // FIXME
}

size_t FIRSequenceType::Shape::hash_value() const {
  return 0;  // FIXME
}

FIRSequenceType FIRSequenceType::get(
    mlir::MLIRContext *ctxt, const Shape &shape, mlir::Type elementType) {
  return {};
}

}  // mlbridge
