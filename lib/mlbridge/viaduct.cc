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

#include "viaduct.h"
#include "fir-dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Module.h"
#include "mlir/StandardOps/Ops.h"
#include "mlir/Target/LLVMIR.h"
#include "../FIR/flattened.h"
#include "../parser/parse-tree-visitor.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Module.h"

using namespace Fortran;
using namespace Fortran::mlbridge;
using LabelRef = FIR::flat::LabelRef;
using MLIRContext = mlir::MLIRContext;

namespace {

/// Generate an unknown location
mlir::Location dummyLoc(MLIRContext *ctxt) {
  // FIXME: replace with a map from a provenance to a source location
  return mlir::UnknownLoc::get(ctxt);
}

/// Converter from Fortran to MLIR
class MLIRConverter {
  using LabelMapType = std::map<LabelRef, mlir::Block *>;
  using Closure = std::function<void(const LabelMapType &)>;

  MLIRContext &mlirContext;
  semantics::SemanticsContext &semanticsContext;
  std::unique_ptr<mlir::Module> module;
  std::unique_ptr<mlir::FuncBuilder> builder;
  std::list<Closure> edgesToAdd;
  LabelMapType blockMap;  // map from flattened labels to MLIR blocks

  /// complete the remaining terminators and edges
  void drawRemainingEdges() {
    for (auto &edge : edgesToAdd) {
      edge(blockMap);
    }
  }

  mlir::Location dummyLocation() { return dummyLoc(&mlirContext); }

  void genMLIR(FIR::AnalysisData &ad, std::list<FIR::flat::Op> &operations);
  void genMLIR(bool lastWasLabel, const FIR::flat::LabelOp &op);
  void genMLIR(const FIR::flat::GotoOp &op);
  void genMLIR(const FIR::flat::ReturnOp &op);
  void genMLIR(const FIR::flat::ConditionalGotoOp &op);
  void genMLIR(const FIR::flat::SwitchIOOp &op);
  void genMLIR(const FIR::flat::SwitchOp &op);
  void genMLIR(FIR::AnalysisData &ad, const FIR::flat::ActionOp &op);
  void genMLIR(const FIR::flat::BeginOp &op);
  void genMLIR(const FIR::flat::EndOp &op);
  void genMLIR(FIR::AnalysisData &ad, const FIR::flat::IndirectGotoOp &op);
  void genMLIR(const FIR::flat::DoIncrementOp &op);
  void genMLIR(const FIR::flat::DoCompareOp &op);
  void genMLIR(const parser::FailImageStmt &stmt);
  void genMLIR(const parser::ReturnStmt &stmt);
  void genMLIR(const parser::StopStmt &stmt);
  template<typename A>
  void genMLIR(const A &tuple, LabelRef trueLabel, LabelRef falseLabel);
  void genMLIR(const parser::Statement<parser::IfThenStmt> &stmt,
      LabelRef trueLabel, LabelRef falseLabel) {
    genMLIR(stmt.statement.t, trueLabel, falseLabel);
  }
  void genMLIR(const parser::Statement<parser::ElseIfStmt> &stmt,
      LabelRef trueLabel, LabelRef falseLabel) {
    genMLIR(stmt.statement.t, trueLabel, falseLabel);
  }
  void genMLIR(
      const parser::IfStmt &stmt, LabelRef trueLabel, LabelRef falseLabel) {
    genMLIR(stmt.t, trueLabel, falseLabel);
  }
  void genMLIR(const parser::Statement<parser::NonLabelDoStmt> &stmt,
      LabelRef trueLabel, LabelRef falseLabel);

  template<typename A>
  void translateRoutine(const A &routine, const std::string &name);

public:
  MLIRConverter(MLIRContext &mlirCtxt, semantics::SemanticsContext &semCtxt)
    : mlirContext{mlirCtxt}, semanticsContext{semCtxt},
      module{llvm::make_unique<mlir::Module>(&mlirCtxt)} {}
  MLIRConverter() = delete;

  std::unique_ptr<mlir::Module> acquireModule() { return std::move(module); }

  template<typename A> constexpr bool Pre(const A &) { return true; }
  template<typename A> constexpr void Post(const A &) {}

  /// Translate the various routines from the parse tree
  void Post(const parser::MainProgram &mainp) {
    std::string mainName{"_MAIN"s};
    if (auto &ps{
            std::get<std::optional<parser::Statement<parser::ProgramStmt>>>(
                mainp.t)}) {
      mainName = ps->statement.v.ToString();
    }
    translateRoutine(mainp, mainName);
  }
  void Post(const parser::FunctionSubprogram &subp) {
    std::string name{std::get<parser::Name>(
        std::get<parser::Statement<parser::FunctionStmt>>(subp.t).statement.t)
                         .ToString()};
    translateRoutine(subp, name);
  }
  void Post(const parser::SubroutineSubprogram &subp) {
    std::string name{std::get<parser::Name>(
        std::get<parser::Statement<parser::SubroutineStmt>>(subp.t).statement.t)
                         .ToString()};
    translateRoutine(subp, name);
  }
};

// Control flow destination
void MLIRConverter::genMLIR(bool lastWasLabel, const FIR::flat::LabelOp &op) {
  if (lastWasLabel) {
    blockMap.insert({op.get(), builder->getInsertionBlock()});
  } else {
    auto *newBlock{builder->createBlock()};
    blockMap.insert({op.get(), newBlock});
    if (builder->getInsertionBlock()) {
      builder->create<mlir::BranchOp>(dummyLocation(), newBlock);
    }
    builder->setInsertionPointToStart(newBlock);
  }
}

// Goto statements
void MLIRConverter::genMLIR(const FIR::flat::GotoOp &op) {
  auto iter{blockMap.find(op.target)};
  if (iter != blockMap.end()) {
    builder->create<mlir::BranchOp>(dummyLocation(), iter->second);
  } else {
    using namespace std::placeholders;
    edgesToAdd.emplace_back(std::bind(
        [](mlir::FuncBuilder *builder, mlir::Block *block, LabelRef dest,
            mlir::Location location, const LabelMapType &map) {
          builder->setInsertionPointToEnd(block);
          assert(map.find(dest) != map.end() && "no destination");
          builder->create<mlir::BranchOp>(location, map.find(dest)->second);
        },
        builder.get(), builder->getInsertionBlock(), op.target, dummyLocation(),
        _1));
  }
  builder->clearInsertionPoint();
}

// Return-like statements
void MLIRConverter::genMLIR(const parser::FailImageStmt &stmt) {
  auto calleeName{"Fortran_fail_image"s};  // FIXME
  auto *callee{module->getNamedFunction(calleeName)};
  llvm::SmallVector<mlir::Value *, 8> operands;  // FIXME: argument(s)?
  builder->create<mlir::CallOp>(dummyLocation(), callee, operands);
  builder->create<UnreachableOp>(dummyLocation());
}
void MLIRConverter::genMLIR(const parser::ReturnStmt &stmt) {
  builder->create<mlir::ReturnOp>(dummyLocation());  // FIXME: argument(s)?
}
void MLIRConverter::genMLIR(const parser::StopStmt &stmt) {
  auto calleeName{"Fortran_stop"s};  // FIXME
  auto *callee{module->getNamedFunction(calleeName)};
  llvm::SmallVector<mlir::Value *, 8> operands;  // FIXME: argument(s)?
  builder->create<mlir::CallOp>(dummyLocation(), callee, operands);
  builder->create<UnreachableOp>(dummyLocation());
}
void MLIRConverter::genMLIR(const FIR::flat::ReturnOp &op) {
  std::visit([&](const auto *stmt) { genMLIR(*stmt); }, op.u);
  builder->clearInsertionPoint();
}

// Conditional branch-like statements
template<typename A>
void MLIRConverter::genMLIR(
    const A &tuple, LabelRef trueLabel, LabelRef falseLabel) {
#if 0
  auto exprRef{
      ExprRef(std::get<parser::ScalarLogicalExpr>(tuple).thing.thing.value())};
  assert(exprRef && "condition expression missing");
  auto *cond{builder->create<>(exprRef)};
  AddOrQueueCGoto(cond, trueLabel, falseLabel);
#endif
}
void MLIRConverter::genMLIR(
    const parser::Statement<parser::NonLabelDoStmt> &stmt, LabelRef trueLabel,
    LabelRef falseLabel) {
#if 0
  auto *cond{BuildLoopLatchExpression(stmt.statement)};
  AddOrQueueCGoto(cond, trueLabel, falseLabel);
#endif
}
void MLIRConverter::genMLIR(const FIR::flat::ConditionalGotoOp &op) {
  std::visit(
      [&](const auto *stmt) { genMLIR(*stmt, op.trueLabel, op.falseLabel); },
      op.u);
  builder->clearInsertionPoint();
}

void MLIRConverter::genMLIR(const FIR::flat::SwitchIOOp &op) {}
void MLIRConverter::genMLIR(const FIR::flat::SwitchOp &op) {}
void MLIRConverter::genMLIR(const FIR::flat::BeginOp &op) {}
void MLIRConverter::genMLIR(const FIR::flat::EndOp &op) {}
void MLIRConverter::genMLIR(const FIR::flat::DoIncrementOp &op) {}
void MLIRConverter::genMLIR(const FIR::flat::DoCompareOp &op) {}

void MLIRConverter::genMLIR(
    FIR::AnalysisData &ad, const FIR::flat::ActionOp &op) {}

void MLIRConverter::genMLIR(
    FIR::AnalysisData &ad, const FIR::flat::IndirectGotoOp &op) {
  // add or queue an igoto
}

void MLIRConverter::genMLIR(
    FIR::AnalysisData &ad, std::list<FIR::flat::Op> &operations) {
  bool lastWasLabel{false};
  for (auto &op : operations) {
    std::visit(common::visitors{
                   [&](const FIR::flat::IndirectGotoOp &oper) {
                     genMLIR(ad, oper);
                     lastWasLabel = false;
                   },
                   [&](const FIR::flat::ActionOp &oper) {
                     genMLIR(ad, oper);
                     lastWasLabel = false;
                   },
                   [&](const FIR::flat::LabelOp &oper) {
                     genMLIR(lastWasLabel, oper);
                     lastWasLabel = true;
                   },
                   [&](const FIR::flat::BeginOp &oper) {
                     genMLIR(oper);
                     lastWasLabel = true;
                   },
                   [&](const auto &oper) {
                     genMLIR(oper);
                     lastWasLabel = false;
                   },
               },
        op.u);
  }
  drawRemainingEdges();
}

mlir::Function *createFunction(MLIRContext *ctxt, const std::string &name) {
  // FIXME: generate the correct type
  llvm::SmallVector<mlir::Type, 4> retTy;
  llvm::SmallVector<mlir::Type, 4> argTy;
  auto funcTy{mlir::FunctionType::get(argTy, retTy, ctxt)};
  return new mlir::Function(dummyLoc(ctxt), name, funcTy, /*attrs=*/{});
}

/// Translate the routine to MLIR
template<typename A>
void MLIRConverter::translateRoutine(
    const A &routine, const std::string &name) {
  assert(!module->getNamedFunction(name));
  std::unique_ptr<mlir::Function> function{createFunction(&mlirContext, name)};
  builder = llvm::make_unique<mlir::FuncBuilder>(function.get());
  FIR::AnalysisData ad;
  std::list<FIR::flat::Op> operations;
  CreateFlatIR(routine, operations, ad);
  genMLIR(ad, operations);
}

}  // namespace

std::unique_ptr<mlir::Module> Fortran::mlbridge::MLIRViaduct(
    mlir::MLIRContext &mlirCtxt, const parser::Program &prg,
    semantics::SemanticsContext &semCtxt) {
  MLIRConverter converter{mlirCtxt, semCtxt};
  Walk(prg, converter);
  return converter.acquireModule();
}

std::unique_ptr<llvm::Module> Fortran::mlbridge::LLVMViaduct(
    mlir::Module &module) {
  return mlir::translateModuleToLLVMIR(module);
}

std::unique_ptr<mlir::MLIRContext> Fortran::mlbridge::getFortranMLIRContext() {
  mlir::registerDialect<FIRDialect>();
  return std::make_unique<MLIRContext>();
}
