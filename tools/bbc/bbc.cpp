//===- bbc.cpp - Burnside Bridge Compiler -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This is a tool for translating Fortran sources to the FIR dialect of MLIR.
///
//===----------------------------------------------------------------------===//

#include "flang/common/Fortran-features.h"
#include "flang/common/default-kinds.h"
#include "flang/lower/Bridge.h"
#include "flang/lower/ConvertExpr.h"
#include "flang/optimizer/CodeGen/CodeGen.h"
#include "flang/optimizer/Dialect/FIRDialect.h"
#include "flang/optimizer/Support/InternalNames.h"
#include "flang/optimizer/Support/KindMapping.h"
#include "flang/optimizer/Transforms/Passes.h"
#include "flang/optimizer/Transforms/StdConverter.h"
#include "flang/parser/characters.h"
#include "flang/parser/dump-parse-tree.h"
#include "flang/parser/message.h"
#include "flang/parser/parse-tree-visitor.h"
#include "flang/parser/parse-tree.h"
#include "flang/parser/parsing.h"
#include "flang/parser/provenance.h"
#include "flang/parser/unparse.h"
#include "flang/semantics/expression.h"
#include "flang/semantics/semantics.h"
#include "flang/semantics/unparse-with-symbols.h"
#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
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
#include <iostream>

namespace Br = Fortran::lower;

using namespace llvm;

namespace {

// Some basic command-line options
cl::opt<std::string> inputFilename(cl::Positional, cl::Required,
                                   cl::desc("<input file>"));

cl::opt<std::string> outputFilename("o",
                                    cl::desc("Specify the output filename"),
                                    cl::value_desc("filename"),
                                    cl::init("a.mlir"));

cl::list<std::string> includeDirs("I", cl::desc("include search paths"));

cl::list<std::string> moduleDirs("module", cl::desc("module search paths"));

cl::opt<std::string> moduleSuffix("module-suffix",
                                  cl::desc("module file suffix override"),
                                  cl::init(".mod"));

cl::opt<bool> emitLLVM("emit-llvm",
                       cl::desc("Add passes to lower to and emit LLVM IR"),
                       cl::init(false));
cl::opt<bool> emitFIR("emit-fir",
                      cl::desc("Dump the FIR created by lowering and exit"),
                      cl::init(false));

// vestigal struct that should be deleted
struct DriverOptions {
  bool forcedForm{false};             // -Mfixed or -Mfree appeared
  bool warnOnNonstandardUsage{false}; // -Mstandard
  bool warningsAreErrors{false};      // -Werror
  Fortran::parser::Encoding encoding{Fortran::parser::Encoding::UTF_8};
  std::string prefix;
};

int exitStatus{EXIT_SUCCESS};

// Convert Fortran input to MLIR (target is FIR dialect)
void convertFortranSourceToMLIR(
    std::string path, Fortran::parser::Options options, DriverOptions &driver,
    Fortran::semantics::SemanticsContext &semanticsContext) {
  if (!driver.forcedForm) {
    auto dot{path.rfind(".")};
    if (dot != std::string::npos) {
      std::string suffix{path.substr(dot + 1)};
      options.isFixedForm = suffix == "f" || suffix == "F" || suffix == "ff";
    }
  }
  options.searchDirectories = includeDirs;
  Fortran::parser::Parsing parsing{semanticsContext.allSources()};
  parsing.Prescan(path, options);
  if (!parsing.messages().empty() &&
      (driver.warningsAreErrors || parsing.messages().AnyFatalError())) {
    errs() << driver.prefix << "could not scan " << path << '\n';
    parsing.messages().Emit(std::cerr, parsing.cooked());
    exitStatus = EXIT_FAILURE;
    return;
  }
  parsing.Parse(&std::cout);
  parsing.messages().Emit(std::cerr, parsing.cooked());
  if (!parsing.consumedWholeFile()) {
    parsing.EmitMessage(std::cerr, parsing.finalRestingPlace(),
                        "parser FAIL (final position)");
    exitStatus = EXIT_FAILURE;
    return;
  }
  if ((!parsing.messages().empty() &&
       (driver.warningsAreErrors || parsing.messages().AnyFatalError())) ||
      !parsing.parseTree().has_value()) {
    errs() << driver.prefix << "could not parse " << path << '\n';
    exitStatus = EXIT_FAILURE;
    return;
  }
  auto &parseTree{*parsing.parseTree()};
  Fortran::semantics::Semantics semantics{semanticsContext, parseTree,
                                          parsing.cooked()};
  semantics.Perform();
  semantics.EmitMessages(std::cerr);
  if (semantics.AnyFatalError()) {
    errs() << driver.prefix << "semantic errors in " << path << '\n';
    exitStatus = EXIT_FAILURE;
    return;
  }

  // MLIR+FIR
  fir::NameUniquer nameUniquer;
  auto burnside = Br::BurnsideBridge::create(semanticsContext.defaultKinds(),
                                             &parsing.cooked());
  fir::KindMapping kindMap{&burnside.getMLIRContext()};
  burnside.lower(parseTree, nameUniquer);
  mlir::ModuleOp mlirModule = burnside.getModule();
  std::error_code ec;
  raw_fd_ostream out(outputFilename, ec);
  if (ec) {
    errs() << "could not open output file " << outputFilename << '\n';
    return;
  }
  if (emitFIR) {
    // dump FIR and exit
    mlirModule.print(out);
    return;
  }

  mlir::PassManager pm = mlirModule.getContext();
  pm.addPass(fir::createMemToRegPass());
  pm.addPass(fir::createCSEPass());
  pm.addPass(fir::createLowerToLoopPass());
  pm.addPass(fir::createFIRToStdPass(kindMap));
  pm.addPass(mlir::createLowerToCFGPass());

  if (emitLLVM) {
    pm.addPass(fir::createFIRToLLVMPass(nameUniquer));
    std::error_code ec;
    llvm::ToolOutputFile out(outputFilename + ".ll", ec,
                             llvm::sys::fs::OF_None);
    if (ec) {
      errs() << "can't open output file " + outputFilename + ".ll";
      return;
    }
    pm.addPass(fir::createLLVMDialectToLLVMPass(out.os()));
  }

  if (mlir::succeeded(pm.run(mlirModule))) {
    mlirModule.print(out);
  } else {
    errs() << "oops, pass manager reported failure\n";
    mlirModule.dump();
  }
}

} // namespace

int main(int argc, char **argv) {
  [[maybe_unused]] InitLLVM y(argc, argv);

  mlir::registerPassManagerCLOptions();
  mlir::PassPipelineCLParser passPipe("", "Compiler passes to run");
  cl::ParseCommandLineOptions(argc, argv, "bbc\n");

  DriverOptions driver;
  driver.prefix = argv[0] + ": "s;

  if (includeDirs.size() == 0) {
    includeDirs.push_back(".");
  }
  if (moduleDirs.size() == 0) {
    moduleDirs.push_back(".");
  }

  Fortran::parser::Options options;
  options.predefinitions.emplace_back("__F18", "1");
  options.predefinitions.emplace_back("__F18_MAJOR__", "1");
  options.predefinitions.emplace_back("__F18_MINOR__", "1");
  options.predefinitions.emplace_back("__F18_PATCHLEVEL__", "1");
#if __x86_64__
  options.predefinitions.emplace_back("__x86_64__", "1");
#endif

  Fortran::common::IntrinsicTypeDefaultKinds defaultKinds;
  Fortran::parser::AllSources allSources;
  Fortran::semantics::SemanticsContext semanticsContext{
      defaultKinds, options.features, allSources};
  semanticsContext.set_moduleDirectory(moduleDirs.front())
      .set_moduleFileSuffix(moduleSuffix)
      .set_searchDirectories(includeDirs)
      .set_warnOnNonstandardUsage(driver.warnOnNonstandardUsage)
      .set_warningsAreErrors(driver.warningsAreErrors);

  convertFortranSourceToMLIR(inputFilename, options, driver, semanticsContext);
  return exitStatus;
}
