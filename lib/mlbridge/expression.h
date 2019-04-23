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

#ifndef FORTRAN_MLBRIDGE_EXPRESSION_H_
#define FORTRAN_MLBRIDGE_EXPRESSION_H_

#include "llvm/ADT/SmallVector.h"

/// Conversion of expressions with type Fortran::evaluate::Expr<A> into the FIR
/// dialect of MLIR.
///
/// Fortran expressions appear in syntactic contexts such as, for example,
/// assignment statements.  The semantics of these expressions fall into two
/// cases: (1) an expression that computes a value and (2) an expression that
/// computes an address in storage.  In the FIR dialect, case (1) is represented
/// as a "fir.apply_expr" operation. Case (2) is represented as a
/// "fir.locate_expr" operation.

namespace mlir {
class Value;
}
namespace Fortran::evaluate {
template<typename A> class Expr;
struct SomeType;
}

namespace Fortran::mlbridge {

// In the Fortran::mlbridge namespace, the code will default follow the
// LLVM/MLIR coding standards

using SomeExpr = evaluate::Expr<evaluate::SomeType>;

llvm::SmallVector<mlir::Value *, 8> translateApplyExpr(const SomeExpr *expr);
llvm::SmallVector<mlir::Value *, 8> translateLocateExpr(const SomeExpr *expr);

}  // Fortran::mlbridge

#endif  // FORTRAN_MLBRIDGE_EXPRESSION_H_
