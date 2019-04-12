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

#ifndef FORTRAN_MLBRIDGE_VIADUCT_H_
#define FORTRAN_MLBRIDGE_VIADUCT_H_

#include <memory>

// Implement the viaduct from Fortran to MLIR
// https://github.com/tensorflow/mlir

namespace Fortran::parser {
struct Program;
}

namespace Fortran::semantics {
class SemanticsContext;
}

namespace mlir {
class MLIRContext;
class Module;
}

namespace llvm {
class Module;
}

namespace Fortran::mlbridge {

// In the Fortran::mlbridge namespace, the code will default follow the
// LLVM/MLIR coding standards

/// Viaduct from Fortran parse-tree, etc. to MLIR
std::unique_ptr<mlir::Module> MLIRViaduct(mlir::MLIRContext &context,
    const parser::Program &program,
    semantics::SemanticsContext &semanticsContext);

/// Viaduct from MLIR to LLVM-IR
std::unique_ptr<llvm::Module> LLVMViaduct(mlir::Module &module);

}  // Fortran::mlbridge

std::unique_ptr<mlir::MLIRContext> getFortranMLIRContext();

#endif  // FORTRAN_MLBRIDGE_VIADUCT_H_
