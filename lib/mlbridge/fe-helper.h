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

#ifndef FORTRAN_MLBRIDGE_FE_HELPER_H_
#define FORTRAN_MLBRIDGE_FE_HELPER_H_

/// Traversal and coversion of various Fortran::parser data structures into the
/// FIR dialect of MLIR. These traversals are isolated in this file to hopefully
/// make maintenance easier.

namespace mlir {
class Location;
class MLIRContext;
}

namespace Fortran::parser {
class CharBlock;
}

namespace Fortran::mlbridge {

// In the Fortran::mlbridge namespace, the code will default follow the
// LLVM/MLIR coding standards

mlir::Location dummyLoc(mlir::MLIRContext *ctxt);

/// Translate a CharBlock position to (source-file, line, column)
mlir::Location parserPosToLoc(
    mlir::MLIRContext &context, const parser::CharBlock &position);

}  // Fortran::mlbridge

#endif  // FORTRAN_MLBRIDGE_FE_HELPER_H_
