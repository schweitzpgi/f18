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
#include "bridge.h"
#include "convert-type.h"
#include "fir/FIROpsSupport.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Value.h"

namespace B = Fortran::burnside;
namespace Ev = Fortran::evaluate;
namespace M = mlir;
namespace Se = Fortran::semantics;

using namespace Fortran;
using namespace Fortran::burnside;

M::FuncOp B::createFunction(B::AbstractConverter &converter,
    llvm::StringRef name, M::FunctionType funcTy) {
  return fir::createFuncOp(
      converter.getCurrentLocation(), converter.getModuleOp(), name, funcTy);
}

M::FuncOp B::createFunction(
    M::ModuleOp module, llvm::StringRef name, M::FunctionType funcTy) {
  return fir::createFuncOp(
      M::UnknownLoc::get(module.getContext()), module, name, funcTy);
}

M::FuncOp B::getNamedFunction(M::ModuleOp module, llvm::StringRef name) {
  return module.lookupSymbol<M::FuncOp>(name);
}

void B::SymMap::addSymbol(Se::SymbolRef symbol, M::Value *value) {
  symbolMap.try_emplace(&*symbol, value);
}

M::Value *B::SymMap::lookupSymbol(Se::SymbolRef symbol) {
  auto iter{symbolMap.find(&*symbol)};
  return (iter == symbolMap.end()) ? nullptr : iter->second;
}

void B::SymMap::pushShadowSymbol(Se::SymbolRef symbol, M::Value *value) {
  // find any existing mapping for symbol
  auto iter{symbolMap.find(&*symbol)};
  const Se::Symbol *sym{nullptr};
  M::Value *val{nullptr};
  // if mapping exists, save it on the shadow stack
  if (iter != symbolMap.end()) {
    sym = iter->first;
    val = iter->second;
    symbolMap.erase(iter);
  }
  shadowStack.emplace_back(sym, val);
  // insert new shadow mapping
  auto r{symbolMap.try_emplace(&*symbol, value)};
  assert(r.second && "unexpected insertion failure");
  (void)r;
}
