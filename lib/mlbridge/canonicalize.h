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

#ifndef FORTRAN_MLBRIDGE_CANONICALIZE_H_
#define FORTRAN_MLBRIDGE_CANONICALIZE_H_

/// Canonicalize the Expr<T> trees embedded in both `fir.apply_expr` and
/// `fir.locate_expr` operations into discrete MLIR operations. After this pass,
/// all `fir.apply_expr` and `fir.locate_expr` will be erased.

namespace llvm {
template<typename> class ArrayRef;
}

namespace mlir {
class OpBuilder;
class Pass;
class Value;
}

namespace Fortran::mlbridge {

// In the Fortran::mlbridge namespace, the code will default follow the
// LLVM/MLIR coding standards

class ApplyExpr;
class LocateExpr;

using RewriteVals = mlir::Value *;
using OperandTy = llvm::ArrayRef<mlir::Value *>;

/// Lower FIR to a canonical representation (suitable as .fir files)
mlir::Pass *createFIRLoweringPass();

/// Allow for selective lowering of ApplyExpr ops
RewriteVals lowerSomeExpr(
    mlir::OpBuilder *bldr, OperandTy operands, ApplyExpr &operation);
/// Allow for selective lowering of LocateExpr ops
RewriteVals lowerSomeExpr(
    mlir::OpBuilder *bldr, OperandTy operands, LocateExpr &operation);

}  // Fortran::mlbridge

#endif  // FORTRAN_MLBRIDGE_CANONICALIZE_H_
