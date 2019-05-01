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

#include "fe-helper.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"

namespace Br = Fortran::mlbridge;
namespace M = mlir;
namespace Pa = Fortran::parser;

/// Generate an unknown location
M::Location Br::dummyLoc(M::MLIRContext *ctxt) {
  return M::UnknownLoc::get(ctxt);
}

// What do we need to convert a CharBlock to actual source locations?
// FIXME: replace with a map from a provenance to a source location
M::Location Br::parserPosToLoc(
    M::MLIRContext &context, const Pa::CharBlock &position) {
  return dummyLoc(&context);
}
