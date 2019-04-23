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
#include "../evaluate/expression.h"

using namespace Fortran;
using namespace Fortran::mlbridge;
using MLIRContext = mlir::MLIRContext;
using StringRef = llvm::StringRef;

namespace {
std::string someExprToString(const SomeExpr *expr) {
  std::stringstream ss;
  expr->AsFortran(ss);
  return ss.str();
}
}  // namespace

FIRDialect::FIRDialect(MLIRContext *ctx) : mlir::Dialect("fir", ctx) {
  // addTypes<T>();
  addOperations<ApplyExpr, LocateExpr, UnreachableOp>();
}

FIRDialect::~FIRDialect() = default;

mlir::Type FIRDialect::parseType(StringRef ty, mlir::Location loc) const {
  return {};
}

void FIRDialect::printType(mlir::Type ty, llvm::raw_ostream &os) const {}

mlir::LogicalResult ApplyExpr::verify() { return mlir::success(); }

void ApplyExpr::build(mlir::FuncBuilder *builder, mlir::OperationState *result,
    const SomeExpr *expr, llvm::ArrayRef<mlir::Value *> operands,
    mlir::Type opTy) {
  result->addOperands(operands);
  result->addAttribute(
      "expr", builder->getSerializableAttr(const_cast<SomeExpr *>(expr)));
  result->addTypes(opTy);
}

/// string representation of the expression
StringRef ApplyExpr::getExpr() {
  void *barePtr{getAttrOfType<mlir::SerializableAttr>("expr").getValue()};
  return someExprToString(reinterpret_cast<SomeExpr *>(barePtr));
}

mlir::LogicalResult LocateExpr::verify() { return mlir::success(); }

void LocateExpr::build(mlir::FuncBuilder *builder, mlir::OperationState *result,
    const SomeExpr *expr, llvm::ArrayRef<mlir::Value *> operands,
    mlir::Type opTy) {
  result->addOperands(operands);
  result->addAttribute(
      "addr", builder->getSerializableAttr(const_cast<SomeExpr *>(expr)));
  result->addTypes(opTy);
}

/// string representation of the expression
StringRef LocateExpr::getExpr() {
  void *barePtr{getAttrOfType<mlir::SerializableAttr>("addr").getValue()};
  return someExprToString(reinterpret_cast<SomeExpr *>(barePtr));
}

void StoreExpr::build(mlir::FuncBuilder *builder, mlir::OperationState *result,
    llvm::ArrayRef<mlir::Value *> operands) {
  result->addOperands(operands);
}

/// an unreachable takes no operands or attributes and has no type
void UnreachableOp::build(
    mlir::FuncBuilder *builder, mlir::OperationState *result) {
  // do nothing
}
