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

#ifndef FORTRAN_MLBRIDGE_BRIDGE_H_
#define FORTRAN_MLBRIDGE_BRIDGE_H_

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include <memory>

// Implement the bridge from Fortran to MLIR
// https://github.com/tensorflow/mlir

namespace Fortran::common {
class IntrinsicTypeDefaultKinds;
}

namespace Fortran::parser {
struct Program;
}

namespace llvm {
class Module;
class SourceMgr;
}

namespace Fortran::mlbridge {

// In the Fortran::mlbridge namespace, the code will default follow the
// LLVM/MLIR coding standards

/// An instance of FirBridge is a singleton that owns the state of the bridge
class FirBridge {
public:
  static std::unique_ptr<FirBridge> create(
      const common::IntrinsicTypeDefaultKinds &defaultKinds) {
    FirBridge *p = new FirBridge{defaultKinds};
    return std::unique_ptr<FirBridge>{p};
  }

  mlir::MLIRContext &getMLIRContext() { return *context.get(); }
  mlir::ModuleManager &getManager() { return *manager.get(); }
  mlir::ModuleOp getModule() { return module; }

  void parseSourceFile(llvm::SourceMgr &);

  const common::IntrinsicTypeDefaultKinds &getDefaultKinds() {
    return defaultKinds;
  }

  bool validModule() { return getModule(); }

private:
  explicit FirBridge(const common::IntrinsicTypeDefaultKinds &defaultKinds);
  FirBridge() = delete;
  FirBridge(const FirBridge &) = delete;

  const common::IntrinsicTypeDefaultKinds &defaultKinds;
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::ModuleOp module;
  std::unique_ptr<mlir::ModuleManager> manager;
};

/// Bridge from Fortran parse-tree, etc. to FIR
void FIRBridge(FirBridge &bridge, const parser::Program &program);

/// Bridge from MLIR to LLVM-IR
std::unique_ptr<llvm::Module> LLVMBridge(mlir::ModuleOp &module);

/// instantiate the FIR bridge singleton
void instantiateFIRBridge(
    const common::IntrinsicTypeDefaultKinds &defaultKinds);

/// access to the default kinds class (for MLIR bridge)
const common::IntrinsicTypeDefaultKinds &getDefaultKinds();

/// get the FIR bridge singleton
FirBridge &getBridge();

}  // Fortran::mlbridge

#endif  // FORTRAN_MLBRIDGE_BRIDGE_H_
