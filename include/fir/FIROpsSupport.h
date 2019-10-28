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

#ifndef FIR_FIROPS_SUPPORT_H
#define FIR_FIROPS_SUPPORT_H

#include "fir/FIROps.h"
#include "mlir/Dialect/StandardOps/Ops.h"

namespace fir {

/// return true iff the Operation is a non-volatile LoadOp
inline bool nonVolatileLoad(mlir::Operation *op) {
  if (auto load = dyn_cast<fir::LoadOp>(op))
    return !load.getAttr("volatile");
  return false;
}

/// return true iff the Operation is a fir::CallOp, fir::DispatchOp,
/// mlir::CallOp, or mlir::CallIndirectOp and not pure
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

} // namespace fir

#endif // FIR_FIROPS_SUPPORT_H
