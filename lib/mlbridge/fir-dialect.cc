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

#include "fir-dialect.h"
#include "fir-type.h"
#include "../evaluate/expression.h"
#include "mlir/LLVMIR/LLVMDialect.h"

namespace Br = Fortran::mlbridge;
namespace Co = Fortran::common;
namespace M = mlir;

using namespace Fortran;
using namespace Fortran::mlbridge;

using llvm::StringRef;
using M::MLIRContext;

namespace {
std::string someExprToString(const SomeExpr *expr) {
  std::stringstream ss;
  expr->AsFortran(ss);
  return ss.str();
}

template<typename A>
void selectBuild(M::OpBuilder *builder, M::OperationState *result,
    M::Value *condition, llvm::ArrayRef<typename A::BranchTuple> tuples) {
  result->addOperands(condition);
  for (auto &tup : tuples) {
    auto *cond{std::get<typename A::Conditions>(tup)};
    result->addOperands(cond);
  }
  // Note: succs must be added *after* operands
  for (auto &tup : tuples) {
    auto *block{std::get<M::Block *>(tup)};
    assert(block);
    auto blkArgs{std::get<llvm::ArrayRef<M::Value *>>(tup)};
    result->addSuccessor(block, blkArgs);
  }
}
}  // namespace

Br::FIROpsDialect::FIROpsDialect(MLIRContext *ctx) : M::Dialect("fir", ctx) {
  addTypes<FIRLogicalType, FIRCharacterType, FIRTupleType, FIRReferenceType,
      FIRSequenceType>();
  addOperations<AllocaExpr, GlobalExpr, LoadExpr, StoreExpr, ApplyExpr,
      LocateExpr, UnreachableOp, UndefOp, SelectCaseOp, SelectRankOp,
      SelectTypeOp, SelectOp, AllocMemOp, FreeMemOp>();
}

Br::FIROpsDialect::~FIROpsDialect() = default;

M::Type Br::FIROpsDialect::parseType(StringRef rawData, M::Location loc) const {
  if (rawData.startswith("logical")) {
    // `logical` `{` kind `}`
    rawData = rawData.drop_front(StringRef("logical").size());
  } else if (rawData.startswith("char")) {
    // `char` `{` kind `}`
    rawData = rawData.drop_front(StringRef("char").size());
  } else if (rawData.startswith("type")) {
    // `type` name `<` type (`,` type)* `>` (`{` kind (`,` kind)* `}`)?
    rawData = rawData.drop_front(StringRef("type").size());
  } else if (rawData.startswith("ref")) {
    // `ref` `<` type `>`
    rawData = rawData.drop_front(StringRef("ref").size());
  } else if (rawData.startswith("array")) {
    // bounds ::= (lo `:`)? extent (`;` stride)? | `?`
    // `array` `(` bounds (`,` bounds)* `)`
    rawData = rawData.drop_front(StringRef("array").size());
  } else {
    // error
    assert(false && "not a known type");
  }
  return {};
}

void Br::FIROpsDialect::printType(M::Type ty, llvm::raw_ostream &os) const {
  if (auto type = ty.dyn_cast<FIRReferenceType>()) {
    os << "ref<";
    type.getEleTy().print(os);
    os << ">";
  } else if (auto type = ty.dyn_cast<FIRLogicalType>()) {
    os << "logical {kind: " << type.getFKind() << "}";
  } else if (auto type = ty.dyn_cast<FIRRealType>()) {
    os << "real {kind: " << type.getFKind() << "}";
  } else if (auto type = ty.dyn_cast<FIRCharacterType>()) {
    os << "char {kind: " << type.getFKind() << "}";
  } else {
    assert(false);
  }
}

M::Attribute Br::FIROpsDialect::parseAttribute(
    StringRef attrData, M::Location loc) const {
  return M::SerializableAttr::get(nullptr, getContext());
}

void Br::FIROpsDialect::printAttribute(
    M::Attribute attr, llvm::raw_ostream &os) const {
  os << '?';
}

/// ApplyExpr
void Br::ApplyExpr::build(M::OpBuilder *builder, M::OperationState *result,
    const SomeExpr *expr, const std::map<unsigned, void *> &dict,
    llvm::ArrayRef<M::Value *> operands, M::Type opTy) {
  result->addOperands(operands);
  result->addAttribute(
      "expr", builder->getSerializableAttr(const_cast<SomeExpr *>(expr)));
  void *dictCopy = new std::map<unsigned, void *>{dict};  // FIXME
  result->addAttribute("dict", builder->getSerializableAttr(dictCopy));
  result->addTypes(opTy);
}

M::LogicalResult Br::ApplyExpr::verify() { return M::success(); }

SomeExpr *Br::ApplyExpr::getRawExpr() {
  void *barePtr{getAttrOfType<M::SerializableAttr>("expr").getValue()};
  return reinterpret_cast<SomeExpr *>(barePtr);
}

/// string representation of the expression
StringRef Br::ApplyExpr::getExpr() { return someExprToString(getRawExpr()); }

std::map<unsigned, void *> *Br::ApplyExpr::getDict() {
  void *barePtr{getAttrOfType<M::SerializableAttr>("dict").getValue()};
  return reinterpret_cast<std::map<unsigned, void *> *>(barePtr);
}

M::LogicalResult Br::LocateExpr::verify() { return M::success(); }

/// LocateExpr
void Br::LocateExpr::build(M::OpBuilder *builder, M::OperationState *result,
    const SomeExpr *expr, const std::map<unsigned, void *> &dict,
    llvm::ArrayRef<M::Value *> operands, M::Type opTy) {
  result->addOperands(operands);
  result->addAttribute(
      "addr", builder->getSerializableAttr(const_cast<SomeExpr *>(expr)));
  void *dictCopy = new std::map<unsigned, void *>{dict};  // FIXME
  result->addAttribute("dict", builder->getSerializableAttr(dictCopy));
  result->addTypes(opTy);
}

SomeExpr *Br::LocateExpr::getRawExpr() {
  void *barePtr{getAttrOfType<M::SerializableAttr>("addr").getValue()};
  return reinterpret_cast<SomeExpr *>(barePtr);
}

/// string representation of the expression
StringRef Br::LocateExpr::getExpr() { return someExprToString(getRawExpr()); }

std::map<unsigned, void *> *Br::LocateExpr::getDict() {
  void *barePtr{getAttrOfType<M::SerializableAttr>("dict").getValue()};
  return reinterpret_cast<std::map<unsigned, void *> *>(barePtr);
}

/// AllocaExpr
void Br::AllocaExpr::build(
    M::OpBuilder *builder, M::OperationState *result, M::Type opTy) {
  result->addTypes(FIRReferenceType::get(opTy));
}

void Br::AllocaExpr::build(M::OpBuilder *builder, M::OperationState *result,
    llvm::StringRef name, M::Type opTy) {
  result->addAttribute("name", builder->getStringAttr(name));
  result->addTypes(FIRReferenceType::get(opTy));
}

M::Type Br::AllocaExpr::getAllocatedType() {
  auto ty{*getOperation()->result_type_begin()};
  return ty.cast<FIRReferenceType>().getEleTy();
}

M::LogicalResult Br::AllocaExpr::verify() { return M::success(); }

/// LoadExpr
void Br::LoadExpr::build(M::OpBuilder *builder, M::OperationState *result,
    llvm::ArrayRef<M::Value *> operands, M::Type opTy) {
  result->addOperands(operands);
  result->addTypes(opTy);
}

void Br::LoadExpr::build(
    M::OpBuilder *builder, M::OperationState *result, M::Value *store) {
  auto opTy{store->getType().cast<FIRReferenceType>().getEleTy()};
  llvm::SmallVector<M::Value *, 1> loadArg{store};
  result->addOperands(std::move(loadArg));
  result->addTypes(opTy);
}

M::LogicalResult Br::LoadExpr::verify() { return M::success(); }

/// StoreExpr
void Br::StoreExpr::build(M::OpBuilder *builder, M::OperationState *result,
    llvm::ArrayRef<M::Value *> operands) {
  result->addOperands(operands);
}

void Br::StoreExpr::build(M::OpBuilder *builder, M::OperationState *result,
    M::Value *value, M::Value *store) {
  llvm::SmallVector<M::Value *, 2> storeArgs{value, store};
  result->addOperands(std::move(storeArgs));
}

M::LogicalResult Br::StoreExpr::verify() { return M::success(); }

/// Unreachable
/// an unreachable takes no operands or attributes and has no type
void Br::UnreachableOp::build(
    M::OpBuilder *builder, M::OperationState *result) {
  // do nothing
}

M::LogicalResult Br::UnreachableOp::verify() { return M::success(); }

/// GlobalExpr
void Br::GlobalExpr::build(M::OpBuilder *builder, M::OperationState *result,
    llvm::StringRef name, M::Attribute value, M::Type opTy) {
  result->addAttribute("name", builder->getStringAttr(name));
  result->addAttribute("value", value);
  result->addTypes(FIRReferenceType::get(opTy));
}

M::LogicalResult Br::GlobalExpr::verify() { return M::success(); }

/// Undef
M::LogicalResult Br::UndefOp::verify() { return M::success(); }

void Br::UndefOp::build(
    M::OpBuilder *builder, M::OperationState *result, M::Type opTy) {
  result->addTypes(opTy);
}

/// SelectCase
void Br::SelectCaseOp::build(M::OpBuilder *builder, M::OperationState *result,
    M::Value *condition, llvm::ArrayRef<BranchTuple> tuples) {
  selectBuild<SelectCaseOp>(builder, result, condition, tuples);
}

M::LogicalResult Br::SelectCaseOp::verify() { return M::success(); }

/// SelectRank
void Br::SelectRankOp::build(M::OpBuilder *builder, M::OperationState *result,
    M::Value *condition, llvm::ArrayRef<BranchTuple> tuples) {
  selectBuild<SelectRankOp>(builder, result, condition, tuples);
}

M::LogicalResult Br::SelectRankOp::verify() { return M::success(); }

/// SelectType
void Br::SelectTypeOp::build(M::OpBuilder *builder, M::OperationState *result,
    M::Value *condition, llvm::ArrayRef<BranchTuple> tuples) {
  selectBuild<SelectTypeOp>(builder, result, condition, tuples);
}

M::LogicalResult Br::SelectTypeOp::verify() { return M::success(); }

/// Select
M::LogicalResult Br::SelectOp::verify() { return M::success(); }

/// AllocMem
M::LogicalResult Br::AllocMemOp::verify() { return M::success(); }

/// FreeMem
M::LogicalResult Br::FreeMemOp::verify() { return M::success(); }
