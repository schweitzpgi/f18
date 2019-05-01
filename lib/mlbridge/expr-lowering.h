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

#ifndef FORTRAN_MLBRIDGE_EXPRESSION_LOWERING_H_
#define FORTRAN_MLBRIDGE_EXPRESSION_LOWERING_H_

#include "llvm/ADT/ArrayRef.h"
#include <variant>

/// This pass lowers the FIR.ApplyExpr and FIR.LocateExpr operations,
/// specifically the evaluate::Expr<T> data structures, to MLIR standard
/// operations.

namespace mlir {
class OpBuilder;
class Value;
}

namespace Fortran::mlbridge {

// In the Fortran::mlbridge namespace, the code will default follow the
// LLVM/MLIR coding standards

class FIRBuilder;
class ApplyExpr;
class LocateExpr;

using RewriteVals = mlir::Value *;
using OperandTy = llvm::ArrayRef<mlir::Value *>;

RewriteVals lowerSomeExpr(mlir::OpBuilder *bldr, OperandTy operands,
    std::variant<ApplyExpr, LocateExpr> &&operation);

}  // Fortran::mlbridge

#endif  // FORTRAN_MLBRIDGE_EXPRESSION_LOWERING_H_
