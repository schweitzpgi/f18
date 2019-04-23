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

#include "expression.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Value.h"
// clang-format off
#include "../semantics/type.h"
#include "../evaluate/traversal.h"
// clang-format on

using namespace Fortran;
using namespace Fortran::mlbridge;

llvm::SmallVector<mlir::Value *, 8> Fortran::mlbridge::translateApplyExpr(
    const SomeExpr *expr) {
  return {};
}

llvm::SmallVector<mlir::Value *, 8> Fortran::mlbridge::translateLocateExpr(
    const SomeExpr *expr) {
  return {};
}
