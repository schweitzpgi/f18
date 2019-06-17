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

#ifndef FORTRAN_LIB_MLBRIDGE_FIR_CONVERSION_H
#define FORTRAN_LIB_MLBRIDGE_FIR_CONVERSION_H

namespace mlir {
class Pass;
}

namespace Fortran::mlbridge {

// In the Fortran::mlbridge namespace, the code will default follow the
// LLVM/MLIR coding standards

/// Convert FIR to the standard dialect
mlir::Pass *createFIRToStdPass();

/// Convert the standard dialect to LLVM IR dialect
mlir::Pass *createStdToLLVMPass();

/// Convert the LLVM IR dialect to LLVM-IR proper
mlir::Pass *createLLVMDialectToLLVMPass();

}  // Fortran::mlbridge

#endif  // FORTRAN_LIB_MLBRIDGE_FIR_CONVERSION_H
