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

#ifndef OPTIMIZER_FIROPS_SUPPORT_H
#define OPTIMIZER_FIROPS_SUPPORT_H

#include "mlir/Dialect/StandardOps/Ops.h"
#include "optimizer/FIROps.h"

namespace fir {

/// return true iff the Operation is a non-volatile LoadOp
inline bool nonVolatileLoad(mlir::Operation *op) {
  if (auto load = dyn_cast<fir::LoadOp>(op))
    return !load.getAttr("volatile");
  return false;
}

/// return true iff the Operation is a fir::CallOp, fir::DispatchOp,
/// mlir::CallOp, or mlir::CallIndirectOp and not pure
/// NB: this is not the same as `!pureCall(op)`
inline bool impureCall(mlir::Operation *op) {
  // Should we also auto-detect that the called function is pure if its
  // arguments are not references?  For now, rely on a "pure" attribute.
  if (auto call = dyn_cast<fir::CallOp>(op))
    return !call.getAttr("pure");
  if (auto dispatch = dyn_cast<fir::DispatchOp>(op))
    return !dispatch.getAttr("pure");
  if (auto call = dyn_cast<mlir::CallOp>(op))
    return !call.getAttr("pure");
  if (auto icall = dyn_cast<mlir::CallIndirectOp>(op))
    return !icall.getAttr("pure");
  return false;
}

/// return true iff the Operation is a fir::CallOp, fir::DispatchOp,
/// mlir::CallOp, or mlir::CallIndirectOp and is also pure.
/// NB: this is not the same as `!impureCall(op)`
inline bool pureCall(mlir::Operation *op) {
  // Should we also auto-detect that the called function is pure if its
  // arguments are not references?  For now, rely on a "pure" attribute.
  if (auto call = dyn_cast<fir::CallOp>(op))
    return bool(call.getAttr("pure"));
  if (auto dispatch = dyn_cast<fir::DispatchOp>(op))
    return bool(dispatch.getAttr("pure"));
  if (auto call = dyn_cast<mlir::CallOp>(op))
    return bool(call.getAttr("pure"));
  if (auto icall = dyn_cast<mlir::CallIndirectOp>(op))
    return bool(icall.getAttr("pure"));
  return false;
}

/// Get or create a FuncOp in a module.
///
/// If `module` already contains FuncOp `name`, it is returned. Otherwise, a new
/// FuncOp is created, and that new FuncOp is returned.
mlir::FuncOp createFuncOp(mlir::Location loc, mlir::ModuleOp module,
                          llvm::StringRef name, mlir::FunctionType type,
                          llvm::ArrayRef<mlir::NamedAttribute> attrs = {});

/// Get or create a GlobalOp in a module.
fir::GlobalOp createGlobalOp(mlir::Location loc, mlir::ModuleOp module,
                             llvm::StringRef name, mlir::Type type,
                             llvm::ArrayRef<mlir::NamedAttribute> attrs = {});

} // namespace fir

#endif // OPTIMIZER_FIROPS_SUPPORT_H
