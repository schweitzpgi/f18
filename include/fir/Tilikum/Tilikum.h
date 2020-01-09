//===-- include/fir/Tilikum/Tilikum.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//----------------------------------------------------------------------------//

#ifndef FIR_TILIKUM_TILIKUM_H
#define FIR_TILIKUM_TILIKUM_H

#include <memory>

namespace llvm {
class StringRef;
}
namespace mlir {
class Pass;
}

namespace fir {

struct NameUniquer;

/// Convert FIR to the LLVM IR dialect
std::unique_ptr<mlir::Pass> createFIRToLLVMPass(NameUniquer &uniquer);

/// Convert the LLVM IR dialect to LLVM-IR proper
std::unique_ptr<mlir::Pass>
createLLVMDialectToLLVMPass(llvm::StringRef output);

} // namespace fir

#endif // FIR_TILIKUM_TILIKUM_H
