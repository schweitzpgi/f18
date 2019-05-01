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

#include "builder.h"
#include "fe-helper.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Value.h"

namespace Br = Fortran::mlbridge;
namespace M = mlir;
namespace Se = Fortran::semantics;

using namespace Fortran;
using namespace Fortran::mlbridge;

void FIRBuilder::addSymbol(const Se::Symbol *symbol, M::Value *value) {
  symbolMap.try_emplace(symbol, value);
}

M::Value *FIRBuilder::lookupSymbol(const Se::Symbol *symbol) {
  auto iter = symbolMap.find(symbol);
  return (iter != symbolMap.end()) ? iter->second : nullptr;
}

M::Function *Br::createFunction(
    M::Module *module, const std::string &name, M::FunctionType funcTy) {
  M::MLIRContext *ctxt{module->getContext()};
  auto *func{new M::Function(dummyLoc(ctxt), name, funcTy)};
  module->getFunctions().push_back(func);
  return func;
}
