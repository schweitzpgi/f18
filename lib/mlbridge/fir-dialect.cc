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

using namespace Fortran;
using namespace Fortran::mlbridge;

FIRDialect::FIRDialect(mlir::MLIRContext *ctx) : mlir::Dialect("fir", ctx) {
  // addTypes<T>();
  addOperations<ApplyExpr, LocateExpr>();
}

mlir::LogicalResult ApplyExpr::verify() { return mlir::failure(); }

void ApplyExpr::build(mlir::FuncBuilder *builder, mlir::OperationState *state,
    llvm::StringRef lambda, llvm::ArrayRef<mlir::Value *> args) {
  // FIXME
}

/// string representation of the expression
llvm::StringRef ApplyExpr::getExpr() {
  std::string s{std::visit(common::visitors{
                               [](std::monostate &) { return "opaque-value"s; },
                               [](evaluate::Expr<evaluate::SomeType> &e) {
                                 std::stringstream ss;
                                 e.AsFortran(ss);
                                 return ss.str();
                               },
                           },
      expr)};
  return s;
}

mlir::LogicalResult LocateExpr::verify() { return mlir::failure(); }

void LocateExpr::build(mlir::FuncBuilder *builder, mlir::OperationState *state,
    llvm::StringRef lambda, llvm::ArrayRef<mlir::Value *> args) {
  // FIXME
}

/// string representation of the expression
llvm::StringRef LocateExpr::getExpr() {
  std::string s{
      std::visit(common::visitors{
                     [](std::monostate &) { return "opaque-address"s; },
                     [](evaluate::Expr<evaluate::SomeType> &e) {
                       std::stringstream ss;
                       e.AsFortran(ss);
                       return ss.str();
                     },
                 },
          expr)};
  return s;
}

std::unique_ptr<mlir::MLIRContext> getFortranMLIRContext() {
  mlir::registerDialect<FIRDialect>();
  return std::make_unique<mlir::MLIRContext>();
}
