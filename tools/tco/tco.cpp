//===- tco.cpp - Tilikum Crossing Opt ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is to be like LLVM's opt program, only for FIR.  Such a program is
// required for roundtrip testing, etc.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/CodeGen/CodeGen.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Optimizer/Support/KindMapping.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "flang/Optimizer/Transforms/StdConverter.h"
#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::opt<std::string>
    inputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));

static cl::opt<std::string> outputFilename("o",
                                           cl::desc("Specify output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));

static cl::opt<bool> emitFir("emit-fir",
                             cl::desc("Parse and pretty-print the input"),
                             cl::init(false));

static void printModuleBody(mlir::ModuleOp mod, raw_ostream &output) {
  for (auto &op : mod.getBody()->without_terminator())
    output << op << '\n';
}

// compile a .fir file
static int compileFIR() {
  // check that there is a file to load
  ErrorOr<std::unique_ptr<MemoryBuffer>> fileOrErr =
      MemoryBuffer::getFileOrSTDIN(inputFilename);

  if (std::error_code EC = fileOrErr.getError()) {
    errs() << "Could not open file: " << EC.message() << '\n';
    return 1;
  }

  // load the file into a module
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), SMLoc());
  auto context = std::make_unique<mlir::MLIRContext>();
  auto owningRef = mlir::parseSourceFile(sourceMgr, context.get());

  if (!owningRef) {
    errs() << "Error can't load file " << inputFilename << '\n';
    return 2;
  }
  if (mlir::failed(owningRef->verify())) {
    errs() << "Error verifying FIR module\n";
    return 4;
  }

  std::error_code ec;
  ToolOutputFile out(outputFilename, ec, sys::fs::OF_None);

  // run passes
  fir::NameUniquer uniquer;
  fir::KindMapping kindMap{context.get()};
  mlir::PassManager pm{context.get()};
  mlir::applyPassManagerCLOptions(pm);
  if (emitFir) {
    // parse the input and pretty-print it back out
    // -emit-fir intentionally disables all the passes
  } else {
    // add all the passes
    // the user can disable them individually
    pm.addPass(fir::createMemToRegPass());
    pm.addPass(fir::createCSEPass());
    // convert fir dialect to affine
    pm.addPass(fir::createPromoteToAffinePass());
    // convert fir dialect to loop
    pm.addPass(fir::createLowerToLoopPass());
    pm.addPass(fir::createFIRToStdPass(kindMap));
    // convert loop dialect to standard
    pm.addPass(mlir::createLowerToCFGPass());
    pm.addPass(fir::createFIRToLLVMPass(uniquer));
    pm.addPass(fir::createLLVMDialectToLLVMPass(out.os()));
  }

  // run the pass manager
  if (mlir::succeeded(pm.run(*owningRef))) {
    // passes ran successfully, so keep the output
    if (emitFir)
      printModuleBody(*owningRef, out.os());
    out.keep();
    return 0;
  }

  // pass manager failed
  printModuleBody(*owningRef, errs());
  errs() << "\n\nFAILED: " << inputFilename << '\n';
  return 8;
}

int main(int argc, char **argv) {
  mlir::registerAllDialects();
  fir::registerFIR();
  fir::registerFIRPasses();
  [[maybe_unused]] InitLLVM y(argc, argv);
  mlir::registerPassManagerCLOptions();
  mlir::PassPipelineCLParser passPipe("", "Compiler passes to run");
  cl::ParseCommandLineOptions(argc, argv, "Tilikum Crossing Optimizer\n");
  return compileFIR();
}
