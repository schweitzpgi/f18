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
#include "builder.h"
#include "expression.h"
#include "fe-helper.h"
#include "fir-dialect.h"
#include "fir-type.h"
#include "flattened.h"
#include "runtime.h"
#include "../evaluate/expression.h"
#include "../parser/parse-tree-visitor.h"
#include "../semantics/tools.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/Module.h"
#include "mlir/StandardOps/Ops.h"
#include "mlir/Target/LLVMIR.h"

namespace Br = Fortran::mlbridge;
namespace Co = Fortran::common;
namespace Ev = Fortran::evaluate;
namespace Ff = Fortran::FIR::flat;
namespace M = mlir;
namespace Pa = Fortran::parser;
namespace Se = Fortran::semantics;

using namespace Fortran;
using namespace Fortran::mlbridge;

namespace {

using SomeExpr = Ev::Expr<Ev::SomeType>;

constexpr bool isStopStmt(Pa::StopStmt::Kind kind) {
  return kind == Pa::StopStmt::Kind::Stop;
}

/// Converter from Fortran to MLIR
class MLIRConverter {
  using LabelMapType = std::map<Ff::LabelRef, M::Block *>;
  using Closure = std::function<void(const LabelMapType &)>;

  struct DoBoundsInfo {
    M::Operation *doVar;
    M::Operation *counter;
    M::Value *stepExpr;
    M::Operation *condition;
  };

  M::MLIRContext &mlirContext;
  Se::SemanticsContext &semanticsContext;
  std::unique_ptr<M::Module> module;
  std::unique_ptr<FIRBuilder> builder;
  LabelMapType blockMap;  // map from flattened labels to MLIR blocks
  std::list<Closure> edgeQ;
  std::map<const Pa::NonLabelDoStmt *, DoBoundsInfo> doMap;
  Pa::CharBlock lastKnownPos;
  bool noInsPt{false};

  /// Convert a parser CharBlock to a Location
  M::Location toLocation(const Pa::CharBlock &cb) {
    return parserPosToLoc(mlirContext, cb);
  }
  M::Location toLocation() { return toLocation(lastKnownPos); }

  /// Construct the type of an Expr<A> expression
  M::Type exprType(const SomeExpr *expr) {
    return translateSomeExprToFIRType(&mlirContext, semanticsContext, expr);
  }
  M::Type refExprType(const SomeExpr *expr) {
    auto type{translateSomeExprToFIRType(&mlirContext, semanticsContext, expr)};
    return FIRReferenceType::get(type);
  }

  int getDefaultIntegerKind() {
    return Ev::ExpressionAnalyzer{semanticsContext}.GetDefaultKind(
        Co::TypeCategory::Integer);
  }
  M::Type getDefaultIntegerType() {
    return M::IntegerType::get(8 * getDefaultIntegerKind(), &mlirContext);
  }
  int getDefaultLogicalKind() {
    return Ev::ExpressionAnalyzer{semanticsContext}.GetDefaultKind(
        Co::TypeCategory::Logical);
  }
  M::Type getDefaultLogicalType() {
    return FIRLogicalType::get(&mlirContext, getDefaultLogicalKind());
  }

  M::Value *createFIRExpr(M::Location loc, const SomeExpr *expr) {
    auto exprOps{translateSomeExpr(builder.get(), semanticsContext, expr)};
    if (getExprType(exprOps) == ET_FunctionRef) {
      const auto &args{getArgs(exprOps)};
      if (args.size() == 1) {
        return args[0];
      }
    }
    return builder->create<ApplyExpr>(
        loc, expr, getDict(exprOps), getArgs(exprOps), exprType(expr));
  }

  AllocaExpr createTemp(M::Type type, Se::Symbol *symbol = nullptr) {
    if (symbol)
      if (auto *val{builder->lookupSymbol(symbol)}) {
        return M::cast<AllocaExpr>(*val->getDefiningOp());
      }
    auto insPt(builder->saveInsertionPoint());
    builder->setInsertionPointToStart(&builder->getRegion()->front());
    AllocaExpr ae;
    if (symbol) {
      ae = builder->create<AllocaExpr>(
          toLocation(), symbol->name().ToString(), type);
      builder->addSymbol(symbol, ae);
    } else {
      ae = builder->create<AllocaExpr>(toLocation(), type);
    }
    builder->restoreInsertionPoint(insPt);
    return ae;
  }

  M::Function *genFunctionMLIR(llvm::StringRef callee, M::FunctionType funcTy) {
    if (auto *func{module->getNamedFunction(callee)}) {
      return func;
    }
    return createFunction(module.get(), callee, funcTy);
  }

  M::Function *genRuntimeFunction(RuntimeEntryCode rec, int kind) {
    return genFunctionMLIR(
        getRuntimeEntryName(rec), getRuntimeEntryType(rec, mlirContext, kind));
  }

  template<typename T> DoBoundsInfo *getBoundsInfo(const T &linearOp) {
    auto &st{std::get<Pa::Statement<Pa::NonLabelDoStmt>>(linearOp.v->t)};
    lastKnownPos = st.source;
    auto *s{&st.statement};
    auto iter{doMap.find(s)};
    if (iter != doMap.end()) {
      return &iter->second;
    }
    assert(false && "DO context not present");
    return nullptr;
  }

  // Simple scalar expression builders
  // TODO: handle REAL and COMPLEX (iff needed)
  template<M::CmpIPredicate ICMPOPC>
  M::Value *genCompare(M::Value *lhs, M::Value *rhs) {
    auto lty{lhs->getType()};
    assert(lty == rhs->getType());
    if (lty.isIntOrIndex()) {
      return builder->create<M::CmpIOp>(lhs->getLoc(), ICMPOPC, lhs, rhs);
    }
    if (FIRLogicalType::kindof(lty.getKind())) {
      return builder->create<M::CmpIOp>(lhs->getLoc(), ICMPOPC, lhs, rhs);
    }
    if (FIRCharacterType::kindof(lty.getKind())) {
      // return builder->create<M::CallOp>(lhs->getLoc(), );
      return {};
    }
    assert(false && "cannot generate operation on this type");
    return {};
  }
  M::Value *genGE(M::Value *lhs, M::Value *rhs) {
    return genCompare<M::CmpIPredicate::SGE>(lhs, rhs);
  }
  M::Value *genLE(M::Value *lhs, M::Value *rhs) {
    return genCompare<M::CmpIPredicate::SLE>(lhs, rhs);
  }
  M::Value *genEQ(M::Value *lhs, M::Value *rhs) {
    return genCompare<M::CmpIPredicate::EQ>(lhs, rhs);
  }
  M::Value *genAND(M::Value *lhs, M::Value *rhs) {
    return builder->create<M::AndOp>(lhs->getLoc(), lhs, rhs);
  }

  void genMLIR(FIR::AnalysisData &ad, std::list<Ff::Op> &operations);

  // Control flow destination
  void genMLIR(bool lastWasLabel, const Ff::LabelOp &op) {
    if (lastWasLabel) {
      blockMap.insert({op.get(), builder->getInsertionBlock()});
    } else {
      auto *currBlock{builder->getInsertionBlock()};
      auto *newBlock{builder->createBlock()};
      blockMap.insert({op.get(), newBlock});
      if (!noInsPt) {
        builder->setInsertionPointToEnd(currBlock);
        builder->create<M::BranchOp>(toLocation(), newBlock);
      }
      builder->setInsertionPointToStart(newBlock);
    }
  }

  // Goto statements
  void genMLIR(const Ff::GotoOp &op) {
    auto iter{blockMap.find(op.target)};
    if (iter != blockMap.end()) {
      builder->create<M::BranchOp>(toLocation(), iter->second);
    } else {
      using namespace std::placeholders;
      edgeQ.emplace_back(std::bind(
          [](FIRBuilder *builder, M::Block *block, Ff::LabelRef dest,
              M::Location location, const LabelMapType &map) {
            builder->setInsertionPointToEnd(block);
            assert(map.find(dest) != map.end() && "no destination");
            builder->create<M::BranchOp>(location, map.find(dest)->second);
          },
          builder.get(), builder->getInsertionBlock(), op.target, toLocation(),
          _1));
    }
    noInsPt = true;
  }
  void genMLIR(const Ff::ReturnOp &op) {
    std::visit([&](const auto *stmt) { genMLIR(*stmt); }, op.u);
    noInsPt = true;
  }
  void genMLIR(const Ff::ConditionalGotoOp &op) {
    std::visit(
        [&](const auto *stmt) { genMLIR(*stmt, op.trueLabel, op.falseLabel); },
        op.u);
    noInsPt = true;
  }

  void genMLIR(const Ff::SwitchIOOp &op);

  // CALL with alt-return value returned
  void genMLIR(const Ff::SwitchOp &op, const Pa::CallStmt &stmt) {
    auto loc{toLocation(op.source)};
    // FIXME
    (void)loc;
  }
  void genMLIR(const Ff::SwitchOp &op, const Pa::ComputedGotoStmt &stmt) {
    auto loc{toLocation(op.source)};
    auto *exp{Se::GetExpr(std::get<Pa::ScalarIntExpr>(stmt.t))};
    auto *e1{createFIRExpr(loc, exp)};
    // FIXME
    (void)e1;
  }
  void genMLIR(const Ff::SwitchOp &op, const Pa::ArithmeticIfStmt &stmt) {
    auto loc{toLocation(op.source)};
    auto *exp{Se::GetExpr(std::get<Pa::Expr>(stmt.t))};
    auto *e1{createFIRExpr(loc, exp)};
    // FIXME
    (void)e1;
  }
  M::Value *fromCaseValue(const M::Location &locs, const Pa::CaseValue &val) {
    return createFIRExpr(locs, Se::GetExpr(val));
  }
  void genMLIR(const Ff::SwitchOp &op, const Pa::CaseConstruct &stmt);
  void genMLIR(const Ff::SwitchOp &op, const Pa::SelectRankConstruct &stmt);
  void genMLIR(const Ff::SwitchOp &op, const Pa::SelectTypeConstruct &stmt);
  void genMLIR(const Ff::SwitchOp &op) {
    std::visit([&](auto *construct) { genMLIR(op, *construct); }, op.u);
    noInsPt = true;
  }

  void genMLIR(FIR::AnalysisData &ad, const Ff::ActionOp &op);

  void pushDoContext(const Pa::NonLabelDoStmt *doStmt,
      M::Operation *doVar = nullptr, M::Operation *counter = nullptr,
      M::Value *stepExpr = nullptr) {
    doMap.emplace(doStmt, DoBoundsInfo{doVar, counter, stepExpr});
  }

  void genLoopEnterMLIR(const Pa::LoopControl::Bounds &bounds,
      const Pa::NonLabelDoStmt *stmt, const Pa::CharBlock &source) {
    auto loc{toLocation(source)};
    auto *nameExpr{bounds.name.thing.symbol};
    auto name{createTemp(getDefaultIntegerType(), nameExpr)};
    // evaluate e1, e2 [, e3] ...
    auto *lowerExpr{Se::GetExpr(bounds.lower)};
    auto *e1{createFIRExpr(loc, lowerExpr)};
    auto *upperExpr{Se::GetExpr(bounds.upper)};
    auto *e2{createFIRExpr(loc, upperExpr)};
    M::Value *e3;
    if (bounds.step.has_value()) {
      auto *stepExpr{Se::GetExpr(bounds.step)};
      e3 = createFIRExpr(loc, stepExpr);
    } else {
      auto attr = builder->getIntegerAttr(e2->getType(), 1);
      e3 = builder->create<M::ConstantOp>(loc, attr);
    }
    // name <- e1
    builder->create<StoreExpr>(loc, e1, name);
    auto tripCounter{createTemp(getDefaultIntegerType())};
    // See 11.1.7.4.1, para. 1, item (3)
    // totalTrips ::= iteration count = a
    //   where a = (e2 - e1 + e3) / e3 if a > 0 and 0 otherwise
    auto c1{builder->create<M::SubIOp>(loc, e2, e1)};
    auto c2{builder->create<M::AddIOp>(loc, c1.getResult(), e3)};
    auto c3{builder->create<M::DivISOp>(loc, c2.getResult(), e3)};
    auto *totalTrips{c3.getResult()};
    builder->create<StoreExpr>(loc, totalTrips, tripCounter);
    pushDoContext(stmt, name, tripCounter, e3);
  }

  void genLoopEnterMLIR(const Pa::ScalarLogicalExpr &logicalExpr,
      const Pa::NonLabelDoStmt *stmt, const Pa::CharBlock &source) {
    // See 11.1.7.4.1, para. 2
    // See BuildLoopLatchExpression()
    pushDoContext(stmt);
  }
  void genLoopEnterMLIR(const Pa::LoopControl::Concurrent &concurrent,
      const Pa::NonLabelDoStmt *stmt, const Pa::CharBlock &source) {
    // See 11.1.7.4.2
    // FIXME
  }
  void genEnterMLIR(const Pa::DoConstruct &construct) {
    auto &stmt{std::get<Pa::Statement<Pa::NonLabelDoStmt>>(construct.t)};
    lastKnownPos = stmt.source;
    const Pa::NonLabelDoStmt &ss{stmt.statement};
    auto &ctrl{std::get<std::optional<Pa::LoopControl>>(ss.t)};
    if (ctrl.has_value()) {
      std::visit([&](const auto &x) { genLoopEnterMLIR(x, &ss, stmt.source); },
          ctrl->u);
    } else {
      // loop forever (See 11.1.7.4.1, para. 2)
      pushDoContext(&ss);
    }
  }
  template<typename A> void genEnterMLIR(const A &construct) {
    // FIXME: add other genEnterMLIR() members
  }
  void genMLIR(const Ff::BeginOp &op) {
    std::visit([&](auto *construct) { genEnterMLIR(*construct); }, op.u);
  }

  void genExitMLIR(const Pa::DoConstruct &construct) {
    auto &stmt{std::get<Pa::Statement<Pa::NonLabelDoStmt>>(construct.t)};
    lastKnownPos = stmt.source;
    const Pa::NonLabelDoStmt &ss{stmt.statement};
    auto &ctrl{std::get<std::optional<parser::LoopControl>>(ss.t)};
    if (ctrl.has_value() &&
        std::holds_alternative<parser::LoopControl::Bounds>(ctrl->u)) {
      doMap.erase(&ss);
    }
    noInsPt = true;  // backedge already processed
  }
  void genMLIR(const Ff::EndOp &op) {
    if (auto *construct{std::get_if<const Pa::DoConstruct *>(&op.u)})
      genExitMLIR(**construct);
  }

  void genMLIR(FIR::AnalysisData &ad, const Ff::IndirectGotoOp &op);
  void genMLIR(const Ff::DoIncrementOp &op) {
    auto *info{getBoundsInfo(op)};
    if (info->doVar && info->stepExpr) {
      // add: do_var = do_var + e3
      auto load{builder->create<LoadExpr>(
          info->doVar->getLoc(), info->doVar->getResult(0))};
      auto incremented{builder->create<M::AddIOp>(
          load.getLoc(), load.getResult(), info->stepExpr)};
      builder->create<StoreExpr>(
          load.getLoc(), incremented, info->doVar->getResult(0));
      // add: counter--
      auto loadCtr{builder->create<LoadExpr>(
          info->counter->getLoc(), info->counter->getResult(0))};
      auto one{builder->create<M::ConstantOp>(
          loadCtr.getLoc(), builder->getIntegerAttr(loadCtr.getType(), 1))};
      auto decremented{builder->create<M::SubIOp>(
          loadCtr.getLoc(), loadCtr.getResult(), one)};
      builder->create<StoreExpr>(
          loadCtr.getLoc(), decremented, info->counter->getResult(0));
    }
  }
  void genMLIR(const Ff::DoCompareOp &op) {
    auto *info{getBoundsInfo(op)};
    if (info->doVar && info->stepExpr) {
      // add: cond = counter > 0 (signed)
      auto load{builder->create<LoadExpr>(
          info->counter->getLoc(), info->counter->getResult(0))};
      auto zero{builder->create<M::ConstantOp>(
          load.getLoc(), builder->getIntegerAttr(load.getType(), 0))};
      auto cond{builder->create<M::CmpIOp>(
          load.getLoc(), M::CmpIPredicate::SGT, load, zero)};
      info->condition = cond;
    }
  }
  void genMLIR(const Pa::FailImageStmt &stmt) {
    auto *callee{genRuntimeFunction(FIRT_FAIL_IMAGE, 0)};
    llvm::SmallVector<M::Value *, 1> operands;  // FAIL IMAGE has no args
    builder->create<M::CallOp>(toLocation(), callee, operands);
    builder->create<UnreachableOp>(toLocation());
  }
  void genMLIR(const Pa::ReturnStmt &stmt) {
    builder->create<M::ReturnOp>(toLocation());  // FIXME: argument(s)?
  }
  void genMLIR(const Pa::StopStmt &stmt) {
    auto *callee{genRuntimeFunction(
        isStopStmt(std::get<Pa::StopStmt::Kind>(stmt.t)) ? FIRT_STOP
                                                         : FIRT_ERROR_STOP,
        getDefaultIntegerKind())};
    // 2 args: stop-code-opt, quiet-opt
    llvm::SmallVector<M::Value *, 8> operands;
    builder->create<M::CallOp>(toLocation(), callee, operands);
    builder->create<UnreachableOp>(toLocation());
  }

  // Conditional branch-like statements
  template<typename A>
  void genMLIR(
      const A &tuple, Ff::LabelRef trueLabel, Ff::LabelRef falseLabel) {
    auto *exprRef{Se::GetExpr(std::get<Pa::ScalarLogicalExpr>(tuple))};
    assert(exprRef && "condition expression missing");
    auto *cond{createFIRExpr(toLocation(), exprRef)};
    genCondBranch(cond, trueLabel, falseLabel);
  }
  void genMLIR(const Pa::Statement<Pa::IfThenStmt> &stmt,
      Ff::LabelRef trueLabel, Ff::LabelRef falseLabel) {
    lastKnownPos = stmt.source;
    genMLIR(stmt.statement.t, trueLabel, falseLabel);
  }
  void genMLIR(const Pa::Statement<Pa::ElseIfStmt> &stmt,
      Ff::LabelRef trueLabel, Ff::LabelRef falseLabel) {
    lastKnownPos = stmt.source;
    genMLIR(stmt.statement.t, trueLabel, falseLabel);
  }
  void genMLIR(
      const Pa::IfStmt &stmt, Ff::LabelRef trueLabel, Ff::LabelRef falseLabel) {
    genMLIR(stmt.t, trueLabel, falseLabel);
  }

  M::Value *getTrueConstant() {
    auto attr{builder->getBoolAttr(true)};
    return builder->create<M::ConstantOp>(toLocation(), attr).getResult();
  }

  // Conditional branch to enter loop body or exit
  void genMLIR(const Pa::Statement<Pa::NonLabelDoStmt> &stmt,
      Ff::LabelRef trueLabel, Ff::LabelRef falseLabel) {
    lastKnownPos = stmt.source;
    auto &loopCtrl{std::get<std::optional<Pa::LoopControl>>(stmt.statement.t)};
    M::Value *condition{nullptr};
    if (loopCtrl.has_value()) {
      std::visit(Co::visitors{
                     [&](const parser::LoopControl::Bounds &) {
                       auto iter{doMap.find(&stmt.statement)};
                       assert(iter != doMap.end());
                       condition = iter->second.condition->getResult(0);
                     },
                     [&](const parser::ScalarLogicalExpr &logical) {
                       auto loc{toLocation(stmt.source)};
                       auto *exp{Se::GetExpr(logical)};
                       condition = createFIRExpr(loc, exp);
                     },
                     [&](const parser::LoopControl::Concurrent &concurrent) {
                       // FIXME: incorrectly lowering DO CONCURRENT
                       condition = getTrueConstant();
                     },
                 },
          loopCtrl->u);
    } else {
      condition = getTrueConstant();
    }
    assert(condition && "condition must be a Value");
    genCondBranch(condition, trueLabel, falseLabel);
  }

  // Action statements
  void genMLIR(const Pa::AllocateStmt &stmt);
  void genMLIR(const Pa::AssignmentStmt &stmt) {
    auto loc{toLocation()};
    auto *rhs{Se::GetExpr(std::get<Pa::Expr>(stmt.t))};
    auto *value{createFIRExpr(loc, rhs)};
    auto *lhs{Se::GetExpr(std::get<Pa::Variable>(stmt.t))};
    auto lhsOps{translateSomeAddrExpr(builder.get(), semanticsContext, lhs)};
    M::Value *toLoc{nullptr};
    auto defOps{getArgs(lhsOps)};
    if (defOps.size() == 1)
      if (auto *defOp{defOps[0]->getDefiningOp()})
        if (auto load{M::dyn_cast<LoadExpr>(defOp)}) {
          // Skip generating a locate_expr when the location is obvious
          toLoc = load.getOperand();
        }
    if (!toLoc) {
      LocateExpr address{builder->create<LocateExpr>(
          loc, lhs, getDict(lhsOps), getArgs(lhsOps), refExprType(lhs))};
      toLoc = address.getResult();
    }
    builder->create<StoreExpr>(loc, value, toLoc);
  }
  void genMLIR(const Pa::BackspaceStmt &stmt);
  void genMLIR(const Pa::CallStmt &stmt);
  void genMLIR(const Pa::CloseStmt &stmt);
  void genMLIR(const Pa::DeallocateStmt &stmt);
  void genMLIR(const Pa::EndfileStmt &stmt);
  void genMLIR(const Pa::EventPostStmt &stmt);
  void genMLIR(const Pa::EventWaitStmt &stmt);
  void genMLIR(const Pa::FlushStmt &stmt);
  void genMLIR(const Pa::FormTeamStmt &stmt);
  void genMLIR(const Pa::InquireStmt &stmt);
  void genMLIR(const Pa::LockStmt &stmt);
  void genMLIR(const Pa::NullifyStmt &stmt);
  void genMLIR(const Pa::OpenStmt &stmt);
  void genMLIR(const Pa::PointerAssignmentStmt &stmt);
  void genMLIR(const Pa::PrintStmt &stmt);
  void genMLIR(const Pa::ReadStmt &stmt);
  void genMLIR(const Pa::RewindStmt &stmt);
  void genMLIR(const Pa::SyncAllStmt &stmt);
  void genMLIR(const Pa::SyncImagesStmt &stmt);
  void genMLIR(const Pa::SyncMemoryStmt &stmt);
  void genMLIR(const Pa::SyncTeamStmt &stmt);
  void genMLIR(const Pa::UnlockStmt &stmt);
  void genMLIR(const Pa::WaitStmt &stmt);
  void genMLIR(const Pa::WhereStmt &stmt);
  void genMLIR(const Pa::WriteStmt &stmt);
  void genMLIR(const Pa::ForallStmt &stmt);
  void genMLIR(FIR::AnalysisData &ad, const Pa::AssignStmt &stmt);
  void genMLIR(const Pa::PauseStmt &stmt);

  template<typename A>
  void translateRoutine(const A &routine, const std::string &name,
      const Se::Symbol *funcSym, bool dumpFlatIR = false);

  void genCondBranch(
      M::Value *cond, Ff::LabelRef trueBlock, Ff::LabelRef falseBlock) {
    auto trueIter{blockMap.find(trueBlock)};
    auto falseIter{blockMap.find(falseBlock)};
    if (trueIter != blockMap.end() && falseIter != blockMap.end()) {
      llvm::SmallVector<M::Value *, 2> blanks;
      builder->create<M::CondBranchOp>(toLocation(), cond, trueIter->second,
          blanks, falseIter->second, blanks);
    } else {
      using namespace std::placeholders;
      edgeQ.emplace_back(std::bind(
          [](FIRBuilder *builder, M::Block *block, M::Value *cnd,
              Ff::LabelRef trueDest, Ff::LabelRef falseDest,
              M::Location location, const LabelMapType &map) {
            llvm::SmallVector<M::Value *, 2> blk;
            builder->setInsertionPointToEnd(block);
            auto tdp{map.find(trueDest)};
            auto fdp{map.find(falseDest)};
            assert(tdp != map.end() && fdp != map.end());
            builder->create<M::CondBranchOp>(
                location, cnd, tdp->second, blk, fdp->second, blk);
          },
          builder.get(), builder->getInsertionBlock(), cond, trueBlock,
          falseBlock, toLocation(), _1));
    }
  }

  template<typename A>
  void genSwitchBranch(const M::Location &loc, M::Value *selector,
      std::list<typename A::Conditions> &&conditions,
      const std::vector<Ff::LabelRef> &labels) {
    assert(conditions.size() == labels.size());
    bool haveAllLabels{true};
    std::size_t u{0};
    // do we already have all the targets?
    for (auto last{labels.size()}; u != last; ++u) {
      haveAllLabels = blockMap.find(labels[u]) != blockMap.end();
      if (!haveAllLabels) break;
    }
    if (haveAllLabels) {
      // yes, so generate the FIR operation now
      u = 0;
      std::vector<typename A::BranchTuple> x;
      for (auto &cond : conditions) {
        llvm::SmallVector<M::Value *, 2> blanks;
        x.emplace_back(cond, blockMap.find(labels[u++])->second, blanks);
      }
      builder->create<A>(loc, selector, x);
    } else {
      // no, so queue the FIR operation for later
      using namespace std::placeholders;
      edgeQ.emplace_back(std::bind(
          [](FIRBuilder *builder, M::Block *block, M::Value *sel,
              const std::list<typename A::Conditions> &conditions,
              const std::vector<Ff::LabelRef> &labels, M::Location location,
              const LabelMapType &map) {
            std::size_t u{0};
            std::vector<typename A::BranchTuple> x;
            llvm::SmallVector<M::Value *, 2> blanks;
            for (auto &cond : conditions) {
              auto iter{map.find(labels[u++])};
              assert(iter != map.end());
              x.emplace_back(cond, iter->second, blanks);
            }
            builder->setInsertionPointToEnd(block);
            builder->create<A>(location, sel, x);
          },
          builder.get(), builder->getInsertionBlock(), selector, conditions,
          labels, loc, _1));
    }
  }

  void finalizeQueued() {
    for (auto &edgeFunc : edgeQ) {
      edgeFunc(blockMap);
    }
  }

public:
  MLIRConverter(M::MLIRContext &mlirCtxt, Se::SemanticsContext &semCtxt)
    : mlirContext{mlirCtxt}, semanticsContext{semCtxt},
      module{llvm::make_unique<M::Module>(&mlirCtxt)} {}
  MLIRConverter() = delete;

  std::unique_ptr<M::Module> acquireModule() { return std::move(module); }

  template<typename A> constexpr bool Pre(const A &) { return true; }
  template<typename A> constexpr void Post(const A &) {}

  /// Translate the various routines from the parse tree
  void Post(const Pa::MainProgram &mainp) {
    std::string mainName{"_MAIN"s};
    if (auto &ps{
            std::get<std::optional<Pa::Statement<Pa::ProgramStmt>>>(mainp.t)}) {
      mainName = ps->statement.v.ToString();
      lastKnownPos = ps->source;
    }
    translateRoutine(mainp, mainName, nullptr);
  }
  void Post(const Pa::FunctionSubprogram &subp) {
    auto &stmt{std::get<Pa::Statement<Pa::FunctionStmt>>(subp.t)};
    lastKnownPos = stmt.source;
    auto &name{std::get<Pa::Name>(stmt.statement.t)};
    translateRoutine(subp, name.ToString(), name.symbol);
  }
  void Post(const Pa::SubroutineSubprogram &subp) {
    auto &stmt{std::get<Pa::Statement<Pa::SubroutineStmt>>(subp.t)};
    lastKnownPos = stmt.source;
    auto &name{std::get<Pa::Name>(stmt.statement.t)};
    translateRoutine(subp, name.ToString(), name.symbol);
  }
};

/// SELECT CASE
/// Build a switch-like structure for a SELECT CASE
void MLIRConverter::genMLIR(
    const Ff::SwitchOp &op, const Pa::CaseConstruct &stmt) {
  auto loc{toLocation(op.source)};
  auto &cstm{std::get<Pa::Statement<Pa::SelectCaseStmt>>(stmt.t)};
  auto *exp{Se::GetExpr(std::get<Pa::Scalar<Pa::Expr>>(cstm.statement.t))};
  auto *e1{createFIRExpr(loc, exp)};
  auto &cases{std::get<std::list<Pa::CaseConstruct::Case>>(stmt.t)};
  std::list<SelectCaseOp::Conditions> conds;
  // Per C1145, we know each `case-expr` must have type INTEGER, CHARACTER, or
  // LOGICAL
  for (auto &sel : cases) {
    auto &cs{std::get<Pa::Statement<Pa::CaseStmt>>(sel.t)};
    auto locs{toLocation(cs.source)};
    auto &csel{std::get<Pa::CaseSelector>(cs.statement.t)};
    std::visit(
        Co::visitors{
            [&](const std::list<Pa::CaseValueRange> &ranges) {
              for (auto &r : ranges) {
                std::visit(Co::visitors{
                               [&](const Pa::CaseValue &val) {
                                 auto *term{fromCaseValue(locs, val)};
                                 conds.emplace_back(genEQ(e1, term));
                               },
                               [&](const Pa::CaseValueRange::Range &rng) {
                                 SelectCaseOp::Conditions rangeComparison{
                                     nullptr};
                                 if (rng.lower.has_value()) {
                                   auto *term{fromCaseValue(locs, *rng.lower)};
                                   // rc = e1 >= lower.term
                                   rangeComparison = genGE(e1, term);
                                 }
                                 if (rng.upper.has_value()) {
                                   auto *term{fromCaseValue(locs, *rng.upper)};
                                   // c = e1 <= upper.term
                                   auto *comparison{genLE(e1, term)};
                                   // rc = if rc then (rc && c) else c
                                   if (rangeComparison) {
                                     rangeComparison =
                                         genAND(rangeComparison, comparison);
                                   } else {
                                     rangeComparison = comparison;
                                   }
                                 }
                                 conds.emplace_back(rangeComparison);
                               },
                           },
                    r.u);
              }
            },
            [&](const Pa::Default &) { conds.emplace_back(getTrueConstant()); },
        },
        csel.u);
  }
  genSwitchBranch<SelectCaseOp>(loc, e1, std::move(conds), op.refs);
}

/// SELECT RANK
/// Build a switch-like structure for a SELECT RANK
void MLIRConverter::genMLIR(
    const Ff::SwitchOp &op, const Pa::SelectRankConstruct &stmt) {
  auto loc{toLocation(op.source)};
  auto &rstm{std::get<Pa::Statement<Pa::SelectRankStmt>>(stmt.t)};
  auto *exp{std::visit([](auto &x) { return Se::GetExpr(x); },
      std::get<Pa::Selector>(rstm.statement.t).u)};
  auto *e1{createFIRExpr(loc, exp)};
  auto &ranks{std::get<std::list<Pa::SelectRankConstruct::RankCase>>(stmt.t)};
  std::list<SelectRankOp::Conditions> conds;
  for (auto &r : ranks) {
    auto &rs{std::get<Pa::Statement<Pa::SelectRankCaseStmt>>(r.t)};
    auto &rank{std::get<Pa::SelectRankCaseStmt::Rank>(rs.statement.t)};
    std::visit(
        Co::visitors{
            [&](const Pa::ScalarIntConstantExpr &ex) {
              auto *ie{createFIRExpr(loc, Se::GetExpr(ex))};
              conds.emplace_back(ie);
            },
            [&](const Pa::Star &) {
              // FIXME: using a bogon for now.  Special value per
              // whatever the runtime returns.
              auto attr{builder->getIntegerAttr(e1->getType(), -1)};
              conds.emplace_back(builder->create<M::ConstantOp>(loc, attr));
            },
            [&](const Pa::Default &) { conds.emplace_back(getTrueConstant()); },
        },
        rank.u);
  }
  // FIXME: fix the type of the function
  auto *callee{genRuntimeFunction(FIRT_GET_RANK, 0)};
  llvm::SmallVector<M::Value *, 1> operands{e1};
  auto e3{builder->create<M::CallOp>(loc, callee, operands)};
  genSwitchBranch<SelectRankOp>(
      loc, e3.getResult(0), std::move(conds), op.refs);
}

/// SELECT TYPE
/// Build a switch-like structure for a SELECT TYPE
void MLIRConverter::genMLIR(
    const Ff::SwitchOp &op, const Pa::SelectTypeConstruct &stmt) {
  auto loc{toLocation(op.source)};
  auto &tstm{std::get<Pa::Statement<Pa::SelectTypeStmt>>(stmt.t)};
  auto *exp{std::visit([](auto &x) { return Se::GetExpr(x); },
      std::get<Pa::Selector>(tstm.statement.t).u)};
  auto *e1{createFIRExpr(loc, exp)};
  auto &types{std::get<std::list<Pa::SelectTypeConstruct::TypeCase>>(stmt.t)};
  std::list<SelectTypeOp::Conditions> conds;
  for (auto &t : types) {
    auto &ts{std::get<Pa::Statement<Pa::TypeGuardStmt>>(t.t)};
    auto &ty{std::get<Pa::TypeGuardStmt::Guard>(ts.statement.t)};
    std::visit(
        Co::visitors{
            [&](const Pa::TypeSpec &) {
              // FIXME: add arguments
              auto *func{genRuntimeFunction(FIRT_ISA_TYPE, 0)};
              llvm::SmallVector<M::Value *, 2> operands;
              auto call{builder->create<M::CallOp>(loc, func, operands)};
              conds.emplace_back(call.getResult(0));
            },
            [&](const Pa::DerivedTypeSpec &) {
              // FIXME: add arguments
              auto *func{genRuntimeFunction(FIRT_ISA_SUBTYPE, 0)};
              llvm::SmallVector<M::Value *, 2> operands;
              auto call{builder->create<M::CallOp>(loc, func, operands)};
              conds.emplace_back(call.getResult(0));
            },
            [&](const Pa::Default &) { conds.emplace_back(getTrueConstant()); },
        },
        ty.u);
  }
  auto *callee{genRuntimeFunction(FIRT_GET_ELETYPE, 0)};
  llvm::SmallVector<M::Value *, 1> operands{e1};
  auto e3{builder->create<M::CallOp>(loc, callee, operands)};
  genSwitchBranch<SelectTypeOp>(
      loc, e3.getResult(0), std::move(conds), op.refs);
}

void MLIRConverter::genMLIR(const Ff::SwitchIOOp &op) {}

void MLIRConverter::genMLIR(const Pa::AllocateStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::BackspaceStmt &stmt) {
  // builder->create<IOCallOp>(stmt.v);
}
void MLIRConverter::genMLIR(const Pa::CallStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::CloseStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::DeallocateStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::EndfileStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::EventPostStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::EventWaitStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::FlushStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::FormTeamStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::InquireStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::LockStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::NullifyStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::OpenStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::PointerAssignmentStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::PrintStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::ReadStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::RewindStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::SyncAllStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::SyncImagesStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::SyncMemoryStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::SyncTeamStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::UnlockStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::WaitStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::WhereStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::WriteStmt &stmt) {}
void MLIRConverter::genMLIR(const Pa::ForallStmt &stmt) {}
void MLIRConverter::genMLIR(FIR::AnalysisData &ad, const Pa::AssignStmt &stmt) {
}
void MLIRConverter::genMLIR(const Pa::PauseStmt &stmt) {}

/// translate action statements
void MLIRConverter::genMLIR(FIR::AnalysisData &ad, const Ff::ActionOp &op) {
  lastKnownPos = op.v->source;
  std::visit(
      Co::visitors{
          [](const Pa::ContinueStmt &) { assert(false); },
          [](const Pa::FailImageStmt &) { assert(false); },
          [](const Co::Indirection<Pa::ArithmeticIfStmt> &) { assert(false); },
          [](const Co::Indirection<Pa::AssignedGotoStmt> &) { assert(false); },
          [](const Co::Indirection<Pa::ComputedGotoStmt> &) { assert(false); },
          [](const Co::Indirection<Pa::CycleStmt> &) { assert(false); },
          [](const Co::Indirection<Pa::ExitStmt> &) { assert(false); },
          [](const Co::Indirection<Pa::GotoStmt> &) { assert(false); },
          [](const Co::Indirection<Pa::IfStmt> &) { assert(false); },
          [](const Co::Indirection<Pa::ReturnStmt> &) { assert(false); },
          [](const Co::Indirection<Pa::StopStmt> &) { assert(false); },
          [&](const Co::Indirection<Pa::AssignStmt> &assign) {
            genMLIR(ad, assign.value());
          },
          [&](const auto &stmt) { genMLIR(stmt.value()); },
      },
      op.v->statement.u);
}

void MLIRConverter::genMLIR(
    FIR::AnalysisData &ad, const Ff::IndirectGotoOp &op) {
  // add or queue an igoto
}

void MLIRConverter::genMLIR(
    FIR::AnalysisData &ad, std::list<Ff::Op> &operations) {
  bool lastWasLabel{false};
  for (auto &op : operations) {
    std::visit(Co::visitors{
                   [&](const Ff::IndirectGotoOp &oper) {
                     genMLIR(ad, oper);
                     lastWasLabel = false;
                   },
                   [&](const Ff::ActionOp &oper) {
                     noInsPt = false;
                     genMLIR(ad, oper);
                     lastWasLabel = false;
                   },
                   [&](const Ff::LabelOp &oper) {
                     genMLIR(lastWasLabel, oper);
                     lastWasLabel = true;
                   },
                   [&](const Ff::BeginOp &oper) {
                     noInsPt = false;
                     genMLIR(oper);
                     lastWasLabel = true;
                   },
                   [&](const auto &oper) {
                     noInsPt = false;
                     genMLIR(oper);
                     lastWasLabel = false;
                   },
               },
        op.u);
  }
  if (builder->getInsertionBlock()) {
    // FIXME: assuming type of '() -> ()'
    builder->create<M::ReturnOp>(toLocation());
  }
}

/// Translate the routine to MLIR
template<typename A>
void MLIRConverter::translateRoutine(const A &routine, const std::string &name,
    const Se::Symbol *funcSym, bool dumpFlatIR) {
  M::Function *func{module->getNamedFunction(name)};
  if (!func) {
    // get arguments and return type if any, otherwise just use empty vectors
    llvm::SmallVector<M::Type, 8> args;
    llvm::SmallVector<M::Type, 2> results;
    if (funcSym) {
      if (auto *details{funcSym->detailsIf<Se::SubprogramDetails>()}) {
        for (auto a : details->dummyArgs()) {
          auto type{
              translateSymbolToFIRType(&mlirContext, semanticsContext, a)};
          args.push_back(FIRReferenceType::get(type));
        }
        if (details->isFunction()) {
          // FIXME: handle subroutines that return magic values
          auto *result{&details->result()};
          results.push_back(
              translateSymbolToFIRType(&mlirContext, semanticsContext, result));
        }
      } else {
        llvm::errs() << "Symbol: " << funcSym->name().ToString() << " @ "
                     << funcSym->details().index() << '\n';
        assert(false && "symbol misidentified by front-end");
      }
    }
    auto funcTy{M::FunctionType::get(args, results, &mlirContext)};
    func = createFunction(module.get(), name, funcTy);
  }
  func->addEntryBlock();
  builder = llvm::make_unique<FIRBuilder>(func);
  builder->setInsertionPointToStart(&func->front());
  if (funcSym) {
    auto &entryBlock{func->front()};
    if (auto *details{funcSym->detailsIf<Se::SubprogramDetails>()}) {
      for (const auto &v :
          llvm::zip(details->dummyArgs(), entryBlock.getArguments())) {
        builder->addSymbol(std::get<0>(v), std::get<1>(v));
      }
    } else {
      llvm::errs() << "Symbol: " << funcSym->name().ToString() << " @ "
                   << funcSym->details().index() << '\n';
      assert(false && "symbol misidentified by front-end");
    }
  }
  FIR::AnalysisData ad;
  std::list<Ff::Op> operations;
  CreateFlatIR(routine, operations, ad);
  if (dumpFlatIR) {
    FIR::dump(operations);
  }
  genMLIR(ad, operations);
  finalizeQueued();
}

}  // namespace

std::unique_ptr<M::Module> Br::MLIRViaduct(M::MLIRContext &mlirCtxt,
    const Pa::Program &prg, Se::SemanticsContext &semCtxt) {
  MLIRConverter converter{mlirCtxt, semCtxt};
  Walk(prg, converter);
  return converter.acquireModule();
}

std::unique_ptr<llvm::Module> Br::LLVMViaduct(M::Module &module) {
  return M::translateModuleToLLVMIR(module);
}

std::unique_ptr<M::MLIRContext> Br::getFortranMLIRContext() {
  M::registerDialect<FIROpsDialect>();
  return std::make_unique<M::MLIRContext>();
}
