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

#ifndef FORTRAN_MLBRIDGE_BUILDER_H_
#define FORTRAN_MLBRIDGE_BUILDER_H_

#include "../semantics/symbol.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/Builders.h"

namespace Fortran::mlbridge {

// In the Fortran::mlbridge namespace, the code will default follow the
// LLVM/MLIR coding standards

/// FIRBuilder extends the MLIR OpBuilder to track context information in the
/// conversion from the front-end to FIR dialect.
class FIRBuilder : public mlir::OpBuilder {
  llvm::DenseMap<const semantics::Symbol *, mlir::Value *> symbolMap;

public:
  explicit FIRBuilder(mlir::Function *func)
    : mlir::OpBuilder{&func->getBody()} {}

  // map a Fortran symbol to its abstract store
  void addSymbol(const semantics::Symbol *symbol, mlir::Value *value);

  mlir::Value *lookupSymbol(const semantics::Symbol *symbol);
};

mlir::Function *createFunction(
    mlir::Module *module, const std::string &name, mlir::FunctionType funcTy);

}  // Fortran::mlbridge

#endif  // FORTRAN_MLBRIDGE_BUILDER_H_
