//===-- Bridge.cc ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/lower/Bridge.h"
#include "NSAliases.h"
#include "flang/lower/ConvertExpr.h"
#include "flang/lower/ConvertType.h"
#include "flang/lower/IO.h"
#include "flang/lower/Intrinsics.h"
#include "flang/lower/Mangler.h"
#include "flang/lower/OpBuilder.h"
#include "flang/lower/PFTBuilder.h"
#include "flang/lower/Runtime.h"
#include "flang/optimizer/Dialect/FIRDialect.h"
#include "flang/optimizer/Dialect/FIROps.h"
#include "flang/optimizer/Dialect/FIRType.h"
#include "flang/optimizer/Support/InternalNames.h"
#include "flang/parser/parse-tree.h"
#include "flang/semantics/tools.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/Parser.h"
#include "mlir/Target/LLVMIR.h"
#include "llvm/Support/CommandLine.h"

using namespace Fortran;
using namespace Fortran::lower;

namespace {

L::cl::opt<bool> dumpBeforeFir("fdebug-dump-pre-fir", L::cl::init(false),
                               L::cl::desc("dump the IR tree prior to FIR"));

L::cl::opt<bool>
    disableToDoAssertions("disable-burnside-todo",
                          L::cl::desc("disable burnside bridge asserts"),
                          L::cl::init(false), L::cl::Hidden);

#undef TODO
#define TODO() assert(false && "not implemented yet")

using SelectCaseConstruct = Pa::CaseConstruct;
using SelectRankConstruct = Pa::SelectRankConstruct;
using SelectTypeConstruct = Pa::SelectTypeConstruct;

using CFGSinkListType = L::SmallVector<PFT::Evaluation *, 2>;
using CFGMapType = L::DenseMap<PFT::Evaluation *, CFGSinkListType *>;

constexpr static bool isStopStmt(const Pa::StopStmt &stm) {
  return std::get<Pa::StopStmt::Kind>(stm.t) == Pa::StopStmt::Kind::Stop;
}

// CfgBuilder implementation
#include "CFGBuilder.h"

#undef TODO
#define TODO()                                                                 \
  {                                                                            \
    if (disableToDoAssertions)                                                 \
      M::emitError(toLocation(), __FILE__)                                     \
          << ":" << __LINE__ << " not implemented";                            \
    else                                                                       \
      assert(false && "not yet implemented");                                  \
  }

/// Converter from PFT to FIR
///
/// After building the PFT and decorating it, the FirConverter processes that
/// representation and lowers it to the FIR executable representation.
class FirConverter : public AbstractConverter {
  using LabelMapType = std::map<PFT::Evaluation *, M::Block *>;
  using Closure = std::function<void(const LabelMapType &)>;

  //
  // Helper function members
  //

  M::Value createFIRAddr(M::Location loc, const Se::SomeExpr *expr) {
    return createSomeAddress(loc, *this, *expr, localSymbols, intrinsics);
  }

  M::Value createFIRExpr(M::Location loc, const Se::SomeExpr *expr) {
    return createSomeExpression(loc, *this, *expr, localSymbols, intrinsics);
  }
  M::Value createLogicalExprAsI1(M::Location loc, const Se::SomeExpr *expr) {
    return createI1LogicalExpression(loc, *this, *expr, localSymbols,
                                     intrinsics);
  }
  M::Value createTemporary(M::Location loc, const Se::Symbol &sym) {
    return Br::createTemporary(loc, *builder, localSymbols, genType(sym), &sym);
  }

  M::FuncOp genFunctionFIR(L::StringRef callee, M::FunctionType funcTy) {
    if (auto func{getNamedFunction(module, callee)}) {
      return func;
    }
    return createFunction(*this, callee, funcTy);
  }

  static bool inMainProgram(PFT::Evaluation *cstr) {
    return std::visit(
        Co::visitors{
            [](PFT::FunctionLikeUnit *c) { return c->isMainProgram(); },
            [&](PFT::Evaluation *c) { return inMainProgram(c); },
            [](auto *) { return false; },
        },
        cstr->parent.p);
  }
  static const Pa::SubroutineStmt *inSubroutine(PFT::Evaluation *cstr) {
    return std::visit(
        Co::visitors{
            [](PFT::FunctionLikeUnit *c) { return c->getSubroutine(); },
            [&](PFT::Evaluation *c) { return inSubroutine(c); },
            [](auto *) -> const Pa::SubroutineStmt * { return nullptr; },
        },
        cstr->parent.p);
  }
  static const Pa::FunctionStmt *inFunction(PFT::Evaluation *cstr) {
    return std::visit(
        Co::visitors{
            [](PFT::FunctionLikeUnit *c) { return c->getFunction(); },
            [&](PFT::Evaluation *c) { return inFunction(c); },
            [](auto *) -> const Pa::FunctionStmt * { return nullptr; },
        },
        cstr->parent.p);
  }
  static const Pa::MpSubprogramStmt *inMPSubp(PFT::Evaluation *cstr) {
    return std::visit(
        Co::visitors{
            [](PFT::FunctionLikeUnit *c) { return c->getMPSubp(); },
            [&](PFT::Evaluation *c) { return inMPSubp(c); },
            [](auto *) -> const Pa::MpSubprogramStmt * { return nullptr; },
        },
        cstr->parent.p);
  }

  template <typename A>
  static const Se::SomeExpr *getScalarExprOfTuple(const A &tuple) {
    return Se::GetExpr(std::get<Pa::ScalarLogicalExpr>(tuple));
  }
  template <typename A>
  static const Se::SomeExpr *getExprOfTuple(const A &tuple) {
    return Se::GetExpr(std::get<Pa::LogicalExpr>(tuple));
  }
  /// Get the condition expression for a CondGoto evaluation
  const Se::SomeExpr *getEvaluationCondition(PFT::Evaluation &eval) {
    return eval.visit(Co::visitors{
        [&](const Pa::IfStmt &stmt) { return getScalarExprOfTuple(stmt.t); },
        [&](const Pa::IfThenStmt &stmt) {
          return getScalarExprOfTuple(stmt.t);
        },
        [&](const Pa::ElseIfStmt &stmt) {
          return getScalarExprOfTuple(stmt.t);
        },
        [&](const Pa::WhereConstructStmt &stmt) {
          return getExprOfTuple(stmt.t);
        },
        [&](const Pa::MaskedElsewhereStmt &stmt) {
          return getExprOfTuple(stmt.t);
        },
        [&](auto &) -> const Se::SomeExpr * {
          M::emitError(toLocation(), "unexpected conditional branch case");
          return nullptr;
        },
    });
  }

  //
  // Function-like PFT entry and exit statements
  //

  void genFIR(const Pa::Statement<Pa::ProgramStmt> &stmt, std::string &name,
              const Se::Symbol *&) {
    setCurrentPosition(stmt.source);
    name = uniquer.doProgramEntry();
  }
  void genFIR(const Pa::Statement<Pa::EndProgramStmt> &stmt, std::string &,
              const Se::Symbol *&) {
    setCurrentPosition(stmt.source);
    genFIR(stmt.statement);
  }
  void genFIR(const Pa::Statement<Pa::FunctionStmt> &stmt, std::string &name,
              const Se::Symbol *&symbol) {
    setCurrentPosition(stmt.source);
    auto &n{std::get<Pa::Name>(stmt.statement.t)};
    symbol = n.symbol;
    assert(symbol && "Name resolution failure");
    name = mangleName(*symbol);
  }
  void genFIR(const Pa::Statement<Pa::EndFunctionStmt> &stmt, std::string &,
              const Se::Symbol *&symbol) {
    setCurrentPosition(stmt.source);
    assert(symbol);
    genFIRFunctionReturn(*symbol);
  }
  void genFIR(const Pa::Statement<Pa::SubroutineStmt> &stmt, std::string &name,
              const Se::Symbol *&symbol) {
    setCurrentPosition(stmt.source);
    auto &n{std::get<Pa::Name>(stmt.statement.t)};
    symbol = n.symbol;
    assert(symbol && "Name resolution failure");
    name = mangleName(*symbol);
  }
  void genFIR(const Pa::Statement<Pa::EndSubroutineStmt> &stmt, std::string &,
              const Se::Symbol *&) {
    setCurrentPosition(stmt.source);
    genFIR(stmt.statement);
  }
  void genFIR(const Pa::Statement<Pa::MpSubprogramStmt> &stmt,
              std::string &name, const Se::Symbol *&symbol) {
    setCurrentPosition(stmt.source);
    auto &n{stmt.statement.v};
    name = n.ToString();
    symbol = n.symbol;
  }
  void genFIR(const Pa::Statement<Pa::EndMpSubprogramStmt> &stmt, std::string &,
              const Se::Symbol *&) {
    setCurrentPosition(stmt.source);
    genFIR(stmt.statement);
  }

  //
  // Termination of symbolically referenced execution units
  //

  /// END of program
  ///
  /// Generate the cleanup block before the program exits
  void genFIRProgramExit() { builder->create<M::ReturnOp>(toLocation()); }
  void genFIR(const Pa::EndProgramStmt &) { genFIRProgramExit(); }

  /// END of procedure-like constructs
  ///
  /// Generate the cleanup block before the procedure exits
  void genFIRFunctionReturn(const Se::Symbol &functionSymbol) {
    const auto &details{functionSymbol.get<Se::SubprogramDetails>()};
    M::Value resultRef{localSymbols.lookupSymbol(details.result())};
    M::Value r{builder->create<fir::LoadOp>(toLocation(), resultRef)};
    builder->create<M::ReturnOp>(toLocation(), r);
  }
  template <typename A>
  void genFIRProcedureExit(const A *) {
    // FIXME: alt-returns
    builder->create<M::ReturnOp>(toLocation());
  }
  void genFIR(const Pa::EndSubroutineStmt &) {
    genFIRProcedureExit(static_cast<const Pa::SubroutineStmt *>(nullptr));
  }
  void genFIR(const Pa::EndMpSubprogramStmt &) {
    genFIRProcedureExit(static_cast<const Pa::MpSubprogramStmt *>(nullptr));
  }

  //
  // Statements that have control-flow semantics
  //

  // Conditional goto control-flow semantics
  void genFIREvalCondGoto(PFT::Evaluation &eval) {
    genFIR(eval);
    auto targets{findTargetsOf(eval)};
    auto *expr{getEvaluationCondition(eval)};
    assert(expr && "condition expression missing");
    auto cond{createLogicalExprAsI1(toLocation(), expr)};
    genFIRCondBranch(cond, targets[0], targets[1]);
  }

  void genFIRCondBranch(M::Value cond, PFT::Evaluation *trueDest,
                        PFT::Evaluation *falseDest) {
    using namespace std::placeholders;
    localEdgeQ.emplace_back(std::bind(
        [](M::OpBuilder *builder, M::Block *block, M::Value cnd,
           PFT::Evaluation *trueDest, PFT::Evaluation *falseDest,
           M::Location location, const LabelMapType &map) {
          L::SmallVector<M::Value, 2> blk;
          builder->setInsertionPointToEnd(block);
          auto tdp{map.find(trueDest)};
          auto fdp{map.find(falseDest)};
          assert(tdp != map.end() && fdp != map.end());
          builder->create<M::CondBranchOp>(location, cnd, tdp->second, blk,
                                           fdp->second, blk);
        },
        builder, builder->getInsertionBlock(), cond, trueDest, falseDest,
        toLocation(), _1));
  }

  // Goto control-flow semantics
  //
  // These are unconditional jumps. There is nothing to evaluate.
  void genFIREvalGoto(PFT::Evaluation &eval) {
    using namespace std::placeholders;
    localEdgeQ.emplace_back(std::bind(
        [](M::OpBuilder *builder, M::Block *block, PFT::Evaluation *dest,
           M::Location location, const LabelMapType &map) {
          builder->setInsertionPointToEnd(block);
          assert(map.find(dest) != map.end() && "no destination");
          builder->create<M::BranchOp>(location, map.find(dest)->second);
        },
        builder, builder->getInsertionBlock(), findSinkOf(eval), toLocation(),
        _1));
  }

  // Indirect goto control-flow semantics
  //
  // For assigned gotos, which is an obsolescent feature. Lower to a switch.
  void genFIREvalIndGoto(PFT::Evaluation &eval) {
    genFIR(eval);
    // FIXME
  }

  // IO statements that have control-flow semantics
  //
  // First lower the IO statement and then do the multiway switch op
  void genFIREvalIoSwitch(PFT::Evaluation &eval) {
    genFIR(eval);
    genFIRIOSwitch(eval);
  }
  void genFIRIOSwitch(PFT::Evaluation &) { TODO(); }

  // Iterative loop control-flow semantics
  void genFIREvalIterative(PFT::Evaluation &) { TODO(); }

  void switchInsertionPointToWhere(fir::WhereOp &where) {
    builder->setInsertionPointToStart(&where.whereRegion().front());
  }
  void switchInsertionPointToOtherwise(fir::WhereOp &where) {
    builder->setInsertionPointToStart(&where.otherRegion().front());
  }
  template <typename A>
  void genWhereCondition(fir::WhereOp &where, const A *stmt) {
    auto cond{createLogicalExprAsI1(
        toLocation(), Se::GetExpr(std::get<Pa::ScalarLogicalExpr>(stmt->t)))};
    where = builder->create<fir::WhereOp>(toLocation(), cond, true);
    switchInsertionPointToWhere(where);
  }

  M::Value genFIRLoopIndex(const Pa::ScalarExpr &x) {
    return builder->create<fir::ConvertOp>(toLocation(),
                                           M::IndexType::get(&mlirContext),
                                           genExprValue(*Se::GetExpr(x)));
  }

  /// Structured control op (`fir.loop`, `fir.where`)
  ///
  /// Convert a DoConstruct to a `fir.loop` op.
  /// Convert an IfConstruct to a `fir.where` op.
  ///
  void genFIREvalStructuredOp(PFT::Evaluation &eval) {
    // TODO: array expressions, FORALL, WHERE ...

    // process the list of Evaluations
    assert(eval.subs && "eval must have a body");
    auto *insPt = builder->getInsertionBlock();

    if (const auto *doConstruct{eval.getIf<Pa::DoConstruct>()}) {
      if (const auto &loopControl{doConstruct->GetLoopControl()}) {
        std::visit(Co::visitors{
                       [&](const Pa::LoopControl::Bounds &x) {
                         M::Value lo{genFIRLoopIndex(x.lower)};
                         M::Value hi{genFIRLoopIndex(x.upper)};
                         auto step{x.step.has_value()
                                       ? genExprValue(*Se::GetExpr(*x.step))
                                       : M::Value{}};
                         auto *sym{x.name.thing.symbol};
                         LoopBuilder{*builder, toLocation()}.createLoop(
                             lo, hi, step,
                             [&](OpBuilderWrapper &handler, M::Value index) {
                               // TODO: should push this cast down to the uses
                               auto cvt{handler.create<fir::ConvertOp>(
                                   genType(*sym), index)};
                               localSymbols.pushShadowSymbol(*sym, cvt);
                               for (auto &e : *eval.subs) {
                                 genFIR(e);
                               }
                               localSymbols.popShadowSymbol();
                             });
                       },
                       [&](const Pa::ScalarLogicalExpr &) {
                         // we should never reach here
                         M::emitError(toLocation(),
                                      "loop lacks iteration space");
                       },
                       [&](const Pa::LoopControl::Concurrent &x) {
                         // FIXME: can project a multi-dimensional space
                         LoopBuilder{*builder, toLocation()}.createLoop(
                             M::Value{}, M::Value{},
                             [&](OpBuilderWrapper &, M::Value) {
                               for (auto &e : *eval.subs) {
                                 genFIR(e);
                               }
                             });
                       },
                   },
                   loopControl->u);
      } else {
        // TODO: Infinite loop: 11.1.7.4.1 par 2
        TODO();
      }
    } else if (eval.isA<Pa::IfConstruct>()) {
      // Construct fir.where
      fir::WhereOp where;
      for (auto &e : *eval.subs) {
        if (auto *s{e.getIf<Pa::IfThenStmt>()}) {
          // fir.where op
          genWhereCondition(where, s);
        } else if (auto *s{e.getIf<Pa::ElseIfStmt>()}) {
          // otherwise block, then nested fir.where
          switchInsertionPointToOtherwise(where);
          genWhereCondition(where, s);
        } else if (e.isA<Pa::ElseStmt>()) {
          // otherwise block
          switchInsertionPointToOtherwise(where);
        } else if (e.isA<Pa::EndIfStmt>()) {
          // close all open fir.where ops
          builder->clearInsertionPoint();
        } else {
          genFIR(e);
        }
      }
    } else {
      assert(false && "not yet implemented");
    }
    builder->setInsertionPointToEnd(insPt);
  }

  // Return from subprogram control-flow semantics
  void genFIREvalReturn(PFT::Evaluation &eval) {
    // Handled case-by-case
    // FIXME: think about moving the case code here
  }

  // Multiway switch control-flow semantics
  void genFIREvalSwitch(PFT::Evaluation &eval) {
    genFIR(eval);
    // FIXME
  }

  // Terminate process control-flow semantics
  //
  // Call a runtime routine that does not return
  void genFIREvalTerminate(PFT::Evaluation &eval) {
    genFIR(eval);
    builder->create<fir::UnreachableOp>(toLocation());
  }

  // No control-flow
  void genFIREvalNone(PFT::Evaluation &eval) { genFIR(eval); }

  M::FuncOp getFunc(L::StringRef name, M::FunctionType ty) {
    if (auto func = getNamedFunction(module, name)) {
      assert(func.getType() == ty);
      return func;
    }
    return createFunction(*this, name, ty);
  }

  /// Lowering of CALL statement
  ///
  /// 1. Determine what function is being called/dispatched to
  /// 2. Build a tuple of arguments to be passed to that function
  /// 3. Emit fir.call/fir.dispatch on arguments
  void genFIR(const Pa::CallStmt &stmt) {
    L::SmallVector<M::Type, 8> argTy;
    L::SmallVector<M::Type, 2> resTy;
    L::StringRef funName;
    std::vector<Se::Symbol *> argsList;
    setCurrentPosition(stmt.v.source);
    std::visit(Co::visitors{
                   [&](const Pa::Name &name) {
                     auto *sym = name.symbol;
                     auto n{sym->name()};
                     funName = L::StringRef{n.begin(), n.size()};
                     auto &details = sym->get<Se::SubprogramDetails>();
                     // TODO ProcEntityDetails?
                     // TODO bindName()?
                     argsList = details.dummyArgs();
                   },
                   [&](const Pa::ProcComponentRef &) { TODO(); },
               },
               std::get<Pa::ProcedureDesignator>(stmt.v.t).u);
    for (auto *d : argsList) {
      Se::SymbolRef sr{*d};
      // FIXME:
      argTy.push_back(fir::ReferenceType::get(genType(sr)));
    }
    auto funTy{M::FunctionType::get(argTy, resTy, builder->getContext())};
    // FIXME: mangle name
    M::FuncOp func{getFunc(funName, funTy)};
    (void)func; // FIXME
    std::vector<M::Value> actuals;
    for (auto &aa : std::get<std::list<Pa::ActualArgSpec>>(stmt.v.t)) {
      auto &kw = std::get<std::optional<Pa::Keyword>>(aa.t);
      auto &arg = std::get<Pa::ActualArg>(aa.t);
      M::Value fe;
      std::visit(Co::visitors{
                     [&](const Co::Indirection<Pa::Expr> &e) {
                       // FIXME: needs to match argument, assumes trivial by-ref
                       fe = genExprAddr(*Se::GetExpr(e));
                     },
                     [&](const Pa::AltReturnSpec &) { TODO(); },
                     [&](const Pa::ActualArg::PercentRef &) { TODO(); },
                     [&](const Pa::ActualArg::PercentVal &) { TODO(); },
                 },
                 arg.u);
      if (kw.has_value()) {
        TODO();
        continue;
      }
      actuals.push_back(fe);
    }

    builder->create<fir::CallOp>(toLocation(), resTy,
                                 builder->getSymbolRefAttr(funName), actuals);
  }

  void genFIR(const Pa::IfStmt &) { TODO(); }
  void genFIR(const Pa::WaitStmt &) { TODO(); }
  void genFIR(const Pa::WhereStmt &) { TODO(); }
  void genFIR(const Pa::ComputedGotoStmt &stmt) {
    auto *exp{Se::GetExpr(std::get<Pa::ScalarIntExpr>(stmt.t))};
    auto e1{genExprValue(*exp)};
    (void)e1;
    TODO();
  }
  void genFIR(const Pa::ForallStmt &) { TODO(); }
  void genFIR(const Pa::ArithmeticIfStmt &stmt) {
    auto *exp{Se::GetExpr(std::get<Pa::Expr>(stmt.t))};
    auto e1{genExprValue(*exp)};
    (void)e1;
    TODO();
  }
  void genFIR(const Pa::AssignedGotoStmt &) { TODO(); }

  void genFIR(const Pa::AssociateConstruct &) { TODO(); }
  void genFIR(const Pa::BlockConstruct &) { TODO(); }
  void genFIR(const Pa::ChangeTeamConstruct &) { TODO(); }
  void genFIR(const Pa::CriticalConstruct &) { TODO(); }
  void genFIR(const Pa::DoConstruct &) { TODO(); }
  void genFIR(const Pa::IfConstruct &) { TODO(); }

  void genFIR(const SelectCaseConstruct &) { TODO(); }
  void genFIR(const SelectRankConstruct &) { TODO(); }
  void genFIR(const SelectTypeConstruct &) { TODO(); }

  void genFIR(const Pa::WhereConstruct &) { TODO(); }

  /// Lower FORALL construct (See 10.2.4)
  void genFIR(const Pa::ForallConstruct &forall) {
    auto &stmt{std::get<Pa::Statement<Pa::ForallConstructStmt>>(forall.t)};
    setCurrentPosition(stmt.source);
    auto &fas{stmt.statement};
    auto &ctrl{std::get<Co::Indirection<Pa::ConcurrentHeader>>(fas.t).value()};
    (void)ctrl;
    for (auto &s : std::get<std::list<Pa::ForallBodyConstruct>>(forall.t)) {
      std::visit(Co::visitors{
                     [&](const Pa::Statement<Pa::ForallAssignmentStmt> &b) {
                       setCurrentPosition(b.source);
                       genFIR(b.statement);
                     },
                     [&](const Pa::Statement<Pa::WhereStmt> &b) {
                       setCurrentPosition(b.source);
                       genFIR(b.statement);
                     },
                     [&](const Pa::WhereConstruct &b) { genFIR(b); },
                     [&](const Co::Indirection<Pa::ForallConstruct> &b) {
                       genFIR(b.value());
                     },
                     [&](const Pa::Statement<Pa::ForallStmt> &b) {
                       setCurrentPosition(b.source);
                       genFIR(b.statement);
                     },
                 },
                 s.u);
    }
    TODO();
  }
  void genFIR(const Pa::ForallAssignmentStmt &s) {
    std::visit([&](auto &b) { genFIR(b); }, s.u);
  }

  void genFIR(const Pa::CompilerDirective &) { TODO(); }
  void genFIR(const Pa::OpenMPConstruct &) { TODO(); }
  void genFIR(const Pa::OmpEndLoopDirective &) { TODO(); }

  void genFIR(const Pa::AssociateStmt &) { TODO(); }
  void genFIR(const Pa::EndAssociateStmt &) { TODO(); }
  void genFIR(const Pa::BlockStmt &) { TODO(); }
  void genFIR(const Pa::EndBlockStmt &) { TODO(); }
  void genFIR(const Pa::SelectCaseStmt &) { TODO(); }
  void genFIR(const Pa::CaseStmt &) { TODO(); }
  void genFIR(const Pa::EndSelectStmt &) { TODO(); }
  void genFIR(const Pa::ChangeTeamStmt &) { TODO(); }
  void genFIR(const Pa::EndChangeTeamStmt &) { TODO(); }
  void genFIR(const Pa::CriticalStmt &) { TODO(); }
  void genFIR(const Pa::EndCriticalStmt &) { TODO(); }

  // Do loop is handled by EvalIterative(), EvalStructuredOp()
  void genFIR(const Pa::NonLabelDoStmt &) {} // do nothing
  void genFIR(const Pa::EndDoStmt &) {}      // do nothing

  // If-Then-Else is handled by EvalCondGoto(), EvalStructuredOp()
  void genFIR(const Pa::IfThenStmt &) {} // do nothing
  void genFIR(const Pa::ElseIfStmt &) {} // do nothing
  void genFIR(const Pa::ElseStmt &) {}   // do nothing
  void genFIR(const Pa::EndIfStmt &) {}  // do nothing

  void genFIR(const Pa::SelectRankStmt &) { TODO(); }
  void genFIR(const Pa::SelectRankCaseStmt &) { TODO(); }
  void genFIR(const Pa::SelectTypeStmt &) { TODO(); }
  void genFIR(const Pa::TypeGuardStmt &) { TODO(); }

  void genFIR(const Pa::WhereConstructStmt &) { TODO(); }
  void genFIR(const Pa::MaskedElsewhereStmt &) { TODO(); }
  void genFIR(const Pa::ElsewhereStmt &) { TODO(); }
  void genFIR(const Pa::EndWhereStmt &) { TODO(); }
  void genFIR(const Pa::ForallConstructStmt &) { TODO(); }
  void genFIR(const Pa::EndForallStmt &) { TODO(); }

  //
  // Statements that do not have control-flow semantics
  //

  // IO statements (see io.h)

  void genFIR(const Pa::BackspaceStmt &stmt) {
    genBackspaceStatement(*this, stmt);
  }
  void genFIR(const Pa::CloseStmt &stmt) { genCloseStatement(*this, stmt); }
  void genFIR(const Pa::EndfileStmt &stmt) { genEndfileStatement(*this, stmt); }
  void genFIR(const Pa::FlushStmt &stmt) { genFlushStatement(*this, stmt); }
  void genFIR(const Pa::InquireStmt &stmt) { genInquireStatement(*this, stmt); }
  void genFIR(const Pa::OpenStmt &stmt) { genOpenStatement(*this, stmt); }
  void genFIR(const Pa::PrintStmt &stmt) { genPrintStatement(*this, stmt); }
  void genFIR(const Pa::ReadStmt &stmt) { genReadStatement(*this, stmt); }
  void genFIR(const Pa::RewindStmt &stmt) { genRewindStatement(*this, stmt); }
  void genFIR(const Pa::WriteStmt &stmt) { genWriteStatement(*this, stmt); }

  void genFIR(const Pa::AllocateStmt &) { TODO(); }

  void genCharacterAssignement(
      const Ev::Assignment::IntrinsicAssignment &assignment) {
    // Helper to get address and length from an Expr that is a character
    // variable designator
    auto getAddrAndLength{[&](const SomeExpr &charDesignatorExpr)
                              -> CharacterOpsBuilder::CharValue {
      M::Value addr = genExprAddr(charDesignatorExpr);
      const auto &charExpr{
          std::get<Ev::Expr<Ev::SomeCharacter>>(charDesignatorExpr.u)};
      std::optional<Ev::Expr<Ev::SubscriptInteger>> lenExpr{charExpr.LEN()};
      assert(lenExpr && "could not get expression to compute character length");
      M::Value len{genExprValue(Ev::AsGenericExpr(std::move(*lenExpr)))};
      return CharacterOpsBuilder::CharValue{addr, len};
    }};

    CharacterOpsBuilder charBuilder{*builder, toLocation()};

    // RHS evaluation.
    // FIXME:  Only works with rhs that are variable reference.
    // Other expression evaluation are not simple copies.
    auto rhs{getAddrAndLength(assignment.rhs)};
    // A temp is needed to evaluate rhs until proven it does not depend on lhs.
    auto tempToEvalRhs{charBuilder.createTemp(rhs.getCharacterType(), rhs.len)};
    charBuilder.createCopy(tempToEvalRhs, rhs, rhs.len);

    // Copy the minimum of the lhs and rhs lengths and pad the lhs remainder
    auto lhs{getAddrAndLength(assignment.lhs)};
    auto cmpLen{
        charBuilder.create<M::CmpIOp>(M::CmpIPredicate::slt, lhs.len, rhs.len)};
    auto copyCount{charBuilder.create<M::SelectOp>(cmpLen, lhs.len, rhs.len)};
    charBuilder.createCopy(lhs, tempToEvalRhs, copyCount);
    charBuilder.createPadding(lhs, copyCount, lhs.len);
  }

  void genFIR(const Pa::AssignmentStmt &stmt) {
    assert(stmt.typedAssignment && "assignment analysis failed");
    if (const auto *assignment{std::get_if<Ev::Assignment::IntrinsicAssignment>(
            &stmt.typedAssignment->v->u)}) {
      const Se::Symbol *sym{Ev::UnwrapWholeSymbolDataRef(assignment->lhs)};
      if (sym && Se::IsAllocatable(*sym)) {
        // Assignment of allocatable are more complex, the lhs
        // may need to be deallocated/reallocated.
        // See Fortran 2018 10.2.1.3 p3
        TODO();
      } else if (sym && Se::IsPointer(*sym)) {
        // Target of the pointer must be assigned.
        // See Fortran 2018 10.2.1.3 p2
        TODO();
      } else if (assignment->lhs.Rank() > 0) {
        // Array assignment
        // See Fortran 2018 10.2.1.3 p5, p6, and p7
        TODO();
      } else {
        // Scalar assignments
        std::optional<Ev::DynamicType> lhsType{assignment->lhs.GetType()};
        assert(lhsType && "lhs cannot be typeless");
        switch (lhsType->category()) {
        case IntegerCat:
        case RealCat:
        case ComplexCat:
        case LogicalCat:
          // Fortran 2018 10.2.1.3 p8 and p9
          // Conversions are already inserted by semantic
          // analysis.
          builder->create<fir::StoreOp>(toLocation(),
                                        genExprValue(assignment->rhs),
                                        genExprAddr(assignment->lhs));
          break;
        case CharacterCat:
          // Fortran 2018 10.2.1.3 p10 and p11
          genCharacterAssignement(*assignment);
          break;
        case DerivedCat:
          // Fortran 2018 10.2.1.3 p12 and p13
          TODO();
          break;
        }
      }
    } else {
      // Defined assignment: call ProcRef
      TODO();
    }
  }

  void genFIR(const Pa::ContinueStmt &) {} // do nothing
  void genFIR(const Pa::DeallocateStmt &) { TODO(); }
  void genFIR(const Pa::EventPostStmt &) {
    // call some runtime routine
    TODO();
  }
  void genFIR(const Pa::EventWaitStmt &) {
    // call some runtime routine
    TODO();
  }

  void genFIR(const Pa::FormTeamStmt &) { TODO(); }
  void genFIR(const Pa::LockStmt &) {
    // call some runtime routine
    TODO();
  }

  /// Nullify pointer object list
  ///
  /// For each pointer object, reset the pointer to a disassociated status.
  /// We do this by setting each pointer to null.
  void genFIR(const Pa::NullifyStmt &stmt) {
    for (auto &po : stmt.v) {
      std::visit(
          Co::visitors{
              [&](const Pa::Name &sym) {
                auto ty{genType(*sym.symbol)};
                auto load{builder->create<fir::LoadOp>(
                    toLocation(), localSymbols.lookupSymbol(*sym.symbol))};
                auto idxTy{M::IndexType::get(&mlirContext)};
                auto zero{builder->create<M::ConstantOp>(
                    toLocation(), idxTy, builder->getIntegerAttr(idxTy, 0))};
                auto cast{
                    builder->create<fir::ConvertOp>(toLocation(), ty, zero)};
                builder->create<fir::StoreOp>(toLocation(), cast, load);
              },
              [&](const Pa::StructureComponent &) { TODO(); },
          },
          po.u);
    }
  }
  void genFIR(const Pa::PointerAssignmentStmt &) { TODO(); }

  void genFIR(const Pa::SyncAllStmt &) {
    // call some runtime routine
    TODO();
  }
  void genFIR(const Pa::SyncImagesStmt &) {
    // call some runtime routine
    TODO();
  }
  void genFIR(const Pa::SyncMemoryStmt &) {
    // call some runtime routine
    TODO();
  }
  void genFIR(const Pa::SyncTeamStmt &) {
    // call some runtime routine
    TODO();
  }
  void genFIR(const Pa::UnlockStmt &) {
    // call some runtime routine
    TODO();
  }

  void genFIR(const Pa::AssignStmt &) { TODO(); }
  void genFIR(const Pa::FormatStmt &) { TODO(); }
  void genFIR(const Pa::EntryStmt &) { TODO(); }
  void genFIR(const Pa::PauseStmt &) {
    // call some runtime routine
    TODO();
  }
  void genFIR(const Pa::DataStmt &) { TODO(); }
  void genFIR(const Pa::NamelistStmt &) { TODO(); }

  // call FAIL IMAGE in runtime
  void genFIR(const Pa::FailImageStmt &stmt) {
    auto callee{
        genRuntimeFunction(RuntimeEntryCode::FailImageStatement, *builder)};
    L::SmallVector<M::Value, 1> operands; // FAIL IMAGE has no args
    builder->create<M::CallOp>(toLocation(), callee, operands);
  }

  // call STOP, ERROR STOP in runtime
  void genFIR(const Pa::StopStmt &stm) {
    auto callee{genRuntimeFunction(RuntimeEntryCode::StopStatement, *builder)};
    // TODO: 3 args: stop-code-opt, ierror, quiet-opt
    // auto isError{genFIRLo!isStopStmt(stmt)}
    L::SmallVector<M::Value, 8> operands;
    builder->create<M::CallOp>(toLocation(), callee, operands);
  }

  // gen expression, if any; share code with END of procedure
  void genFIR(const Pa::ReturnStmt &) {
    if (inMainProgram(currentEvaluation)) {
      builder->create<M::ReturnOp>(toLocation());
    } else if (auto *stmt = inSubroutine(currentEvaluation)) {
      genFIRProcedureExit(stmt);
    } else if (auto *stmt = inFunction(currentEvaluation)) {
      auto *symbol = std::get<Pa::Name>(stmt->t).symbol;
      assert(symbol);
      genFIRFunctionReturn(*symbol);
    } else if (auto *stmt = inMPSubp(currentEvaluation)) {
      genFIRProcedureExit(stmt);
    } else {
      M::emitError(toLocation(), "unknown subprogram type");
    }
  }

  // stubs for generic goto statements; see genFIREvalGoto()
  void genFIR(const Pa::CycleStmt &) {} // do nothing
  void genFIR(const Pa::ExitStmt &) {}  // do nothing
  void genFIR(const Pa::GotoStmt &) {}  // do nothing

  void genFIR(PFT::Evaluation &eval) {
    currentEvaluation = &eval;
    eval.visit(Co::visitors{
        [&](const auto &p) { genFIR(p); },
        [](const PFT::CGJump &) { /* do nothing */ },
    });
  }

  /// Lower an Evaluation
  ///
  /// If the Evaluation is annotated, we can attempt to lower it by the class of
  /// annotation. Otherwise, attempt to lower the Evaluation on a case-by-case
  /// basis.
  void lowerEval(PFT::Evaluation &eval) {
    setCurrentPosition(eval.pos);
    if (eval.isControlTarget()) {
      // start a new block
    }
    switch (eval.cfg) {
    case PFT::CFGAnnotation::None:
      genFIREvalNone(eval);
      break;
    case PFT::CFGAnnotation::Goto:
      genFIREvalGoto(eval);
      break;
    case PFT::CFGAnnotation::CondGoto:
      genFIREvalCondGoto(eval);
      break;
    case PFT::CFGAnnotation::IndGoto:
      genFIREvalIndGoto(eval);
      break;
    case PFT::CFGAnnotation::IoSwitch:
      genFIREvalIoSwitch(eval);
      break;
    case PFT::CFGAnnotation::Switch:
      genFIREvalSwitch(eval);
      break;
    case PFT::CFGAnnotation::Iterative:
      genFIREvalIterative(eval);
      break;
    case PFT::CFGAnnotation::FirStructuredOp:
      genFIREvalStructuredOp(eval);
      break;
    case PFT::CFGAnnotation::Return:
      genFIREvalReturn(eval);
      break;
    case PFT::CFGAnnotation::Terminate:
      genFIREvalTerminate(eval);
      break;
    }
  }

  M::FuncOp createNewFunction(L::StringRef name, const Se::Symbol *symbol) {
    // get arguments and return type if any, otherwise just use empty vectors
    L::SmallVector<M::Type, 8> args;
    L::SmallVector<M::Type, 2> results;
    auto funcTy{symbol ? genFunctionType(*symbol)
                       : M::FunctionType::get(args, results, &mlirContext)};
    return createFunction(*this, name, funcTy);
  }

  /// Prepare to translate a new function
  void startNewFunction(PFT::FunctionLikeUnit &funit, L::StringRef name,
                        const Se::Symbol *symbol) {
    M::FuncOp func{getNamedFunction(module, name)};
    if (!func) {
      func = createNewFunction(name, symbol);
    }
    func.addEntryBlock();
    assert(!builder && "expected nullptr");
    builder = new M::OpBuilder(func);
    assert(builder && "OpBuilder did not instantiate");
    builder->setInsertionPointToStart(&func.front());

    // plumb function's arguments
    if (symbol) {
      auto *entryBlock{&func.front()};
      auto *details{symbol->detailsIf<Se::SubprogramDetails>()};
      assert(details && "details for semantics symbol must be subprogram");
      for (const auto &v :
           L::zip(details->dummyArgs(), entryBlock->getArguments())) {
        if (std::get<0>(v)) {
          localSymbols.addSymbol(*std::get<0>(v), std::get<1>(v));
        } else {
          TODO(); // handle alternate return
        }
      }
      if (details->isFunction()) {
        createTemporary(toLocation(), details->result());
      }
    }
  }

  void finalizeQueuedEdges() {
    for (auto &edgeFunc : localEdgeQ) {
      edgeFunc(localBlockMap);
    }
    localEdgeQ.clear();
    localBlockMap.clear();
  }

  /// Cleanup after the function has been translated
  void endNewFunction() {
    finalizeQueuedEdges();
    delete builder;
    builder = nullptr;
    localSymbols.clear();
  }

  /// Lower a procedure-like construct
  void lowerFunc(PFT::FunctionLikeUnit &func, L::ArrayRef<L::StringRef> modules,
                 L::Optional<L::StringRef> host = {}) {
    std::string name;
    const Se::Symbol *symbol{nullptr};

    if (func.beginStmt) {
      currentEvaluation = nullptr;
      std::visit([&](auto *p) { genFIR(*p, name, symbol); }, *func.beginStmt);
    } else {
      name = uniquer.doProgramEntry();
    }

    startNewFunction(func, name, symbol);

    // lower this procedure
    for (auto &e : func.evals) {
      lowerEval(e);
    }
    currentEvaluation = nullptr;
    std::visit([&](auto *p) { genFIR(*p, name, symbol); }, func.endStmt);

    endNewFunction();

    // recursively lower internal procedures
    L::Optional<L::StringRef> optName{name};
    for (auto &f : func.funcs) {
      lowerFunc(f, modules, optName);
    }
  }

  void lowerMod(PFT::ModuleLikeUnit &mod) {
    // FIXME: build the vector of module names
    std::vector<L::StringRef> moduleName;

    // FIXME: do we need to visit the module statements?
    for (auto &f : mod.funcs) {
      lowerFunc(f, moduleName);
    }
  }

  //
  // Finalization of the CFG structure
  //

  /// Lookup the set of sinks for this source. There must be at least one.
  L::ArrayRef<PFT::Evaluation *> findTargetsOf(PFT::Evaluation &eval) {
    auto iter = cfgMap.find(&eval);
    assert(iter != cfgMap.end());
    return *iter->second;
  }

  /// Lookup the sink for this source. There must be exactly one.
  PFT::Evaluation *findSinkOf(PFT::Evaluation &eval) {
    auto iter = cfgMap.find(&eval);
    assert((iter != cfgMap.end()) && (iter->second->size() == 1));
    return iter->second->front();
  }

  /// prune the CFG for `f`
  void pruneFunc(PFT::FunctionLikeUnit &func) {
    // find and cache arcs, etc.
    if (!func.evals.empty()) {
      CfgBuilder{cfgMap, cfgEdgeSetPool}.run(func);
    }

    // do any internal procedures
    for (auto &f : func.funcs) {
      pruneFunc(f);
    }
  }

  void pruneMod(PFT::ModuleLikeUnit &mod) {
    for (auto &f : mod.funcs) {
      pruneFunc(f);
    }
  }

  void setCurrentPosition(const Pa::CharBlock &pos) {
    if (pos != Pa::CharBlock{}) {
      currentPosition = pos;
    }
  }

  //
  // Utility methods
  //

  /// Convert a parser CharBlock to a Location
  M::Location toLocation(const Pa::CharBlock &cb) { return genLocation(cb); }

  M::Location toLocation() { return toLocation(currentPosition); }

  // TODO: should these be moved to convert-expr?
  template <M::CmpIPredicate ICMPOPC>
  M::Value genCompare(M::Value lhs, M::Value rhs) {
    auto lty{lhs.getType()};
    assert(lty == rhs.getType());
    if (lty.isIntOrIndex())
      return builder->create<M::CmpIOp>(lhs.getLoc(), ICMPOPC, lhs, rhs);
    if (fir::LogicalType::kindof(lty.getKind()))
      return builder->create<M::CmpIOp>(lhs.getLoc(), ICMPOPC, lhs, rhs);
    if (fir::CharacterType::kindof(lty.getKind())) {
      // FIXME
      // return builder->create<M::CallOp>(lhs->getLoc(), );
    }
    M::emitError(toLocation(), "cannot generate operation on this type");
    return {};
  }

  M::Value genGE(M::Value lhs, M::Value rhs) {
    return genCompare<M::CmpIPredicate::sge>(lhs, rhs);
  }
  M::Value genLE(M::Value lhs, M::Value rhs) {
    return genCompare<M::CmpIPredicate::sle>(lhs, rhs);
  }
  M::Value genEQ(M::Value lhs, M::Value rhs) {
    return genCompare<M::CmpIPredicate::eq>(lhs, rhs);
  }
  M::Value genAND(M::Value lhs, M::Value rhs) {
    return builder->create<M::AndOp>(lhs.getLoc(), lhs, rhs);
  }

private:
  M::MLIRContext &mlirContext;
  const Pa::CookedSource *cooked;
  M::ModuleOp &module;
  const Co::IntrinsicTypeDefaultKinds &defaults;
  IntrinsicLibrary intrinsics;
  M::OpBuilder *builder{nullptr};
  fir::NameUniquer &uniquer;
  SymMap localSymbols;
  std::list<Closure> localEdgeQ;
  LabelMapType localBlockMap;
  Pa::CharBlock currentPosition;
  CFGMapType cfgMap;
  std::list<CFGSinkListType> cfgEdgeSetPool;
  PFT::Evaluation *currentEvaluation{nullptr}; // FIXME: this is a hack

public:
  FirConverter() = delete;
  FirConverter(const FirConverter &) = delete;
  FirConverter &operator=(const FirConverter &) = delete;
  virtual ~FirConverter() = default;

  explicit FirConverter(BurnsideBridge &bridge, fir::NameUniquer &uniquer)
      : mlirContext{bridge.getMLIRContext()}, cooked{bridge.getCookedSource()},
        module{bridge.getModule()}, defaults{bridge.getDefaultKinds()},
        intrinsics{IntrinsicLibrary(IntrinsicLibrary::Version::LLVM,
                                    bridge.getMLIRContext())},
        uniquer{uniquer} {}

  /// Convert the PFT to FIR
  void run(PFT::Program &pft) {
    // build pruned control
    for (auto &u : pft.getUnits()) {
      std::visit(Co::visitors{
                     [&](PFT::FunctionLikeUnit &f) { pruneFunc(f); },
                     [&](PFT::ModuleLikeUnit &m) { pruneMod(m); },
                     [](PFT::BlockDataUnit &) { /* do nothing */ },
                 },
                 u);
    }

    // do translation
    for (auto &u : pft.getUnits()) {
      std::visit(Co::visitors{
                     [&](PFT::FunctionLikeUnit &f) { lowerFunc(f, {}); },
                     [&](PFT::ModuleLikeUnit &m) { lowerMod(m); },
                     [&](PFT::BlockDataUnit &) { TODO(); },
                 },
                 u);
    }
  }

  M::FunctionType genFunctionType(SymbolRef sym) {
    return translateSymbolToFIRFunctionType(&mlirContext, defaults, sym);
  }

  //
  // AbstractConverter overrides

  M::Value genExprAddr(const SomeExpr &expr,
                       M::Location *loc = nullptr) override final {
    return createFIRAddr(loc ? *loc : toLocation(), &expr);
  }
  M::Value genExprValue(const SomeExpr &expr,
                        M::Location *loc = nullptr) override final {
    return createFIRExpr(loc ? *loc : toLocation(), &expr);
  }

  M::Type genType(const Ev::DataRef &data) override final {
    return translateDataRefToFIRType(&mlirContext, defaults, data);
  }
  M::Type genType(const SomeExpr &expr) override final {
    return translateSomeExprToFIRType(&mlirContext, defaults, &expr);
  }
  M::Type genType(SymbolRef sym) override final {
    return translateSymbolToFIRType(&mlirContext, defaults, sym);
  }
  M::Type genType(Co::TypeCategory tc, int kind) override final {
    return getFIRType(&mlirContext, defaults, tc, kind);
  }
  M::Type genType(Co::TypeCategory tc) override final {
    return getFIRType(&mlirContext, defaults, tc);
  }

  M::Location getCurrentLocation() override final { return toLocation(); }
  M::Location genLocation() override final {
    return M::UnknownLoc::get(&mlirContext);
  }
  M::Location genLocation(const Pa::CharBlock &block) override final {
    if (cooked) {
      auto loc{cooked->GetSourcePositionRange(block)};
      if (loc.has_value()) {
        // loc is a pair (begin, end); use the beginning position
        auto &filePos{loc->first};
        return M::FileLineColLoc::get(filePos.file.path(), filePos.line,
                                      filePos.column, &mlirContext);
      }
    }
    return genLocation();
  }

  M::OpBuilder &getOpBuilder() override final { return *builder; }
  M::ModuleOp &getModuleOp() override final { return module; }

  std::string mangleName(SymbolRef symbol) override final {
    return mangle::mangleName(uniquer, symbol);
  }
};

} // namespace

void Br::BurnsideBridge::lower(const Pa::Program &prg,
                               fir::NameUniquer &uniquer) {
  std::unique_ptr<PFT::Program> pft{Br::createPFT(prg)};
  Br::annotateControl(*pft);
  if (dumpBeforeFir) {
    Br::dumpPFT(L::errs(), *pft);
  }
  FirConverter converter{*this, uniquer};
  converter.run(*pft);
}

void Br::BurnsideBridge::parseSourceFile(L::SourceMgr &srcMgr) {
  auto owningRef = M::parseSourceFile(srcMgr, context.get());
  module.reset(new M::ModuleOp(owningRef.get().getOperation()));
  owningRef.release();
}

Br::BurnsideBridge::BurnsideBridge(
    const Co::IntrinsicTypeDefaultKinds &defaultKinds,
    const Pa::CookedSource *cooked)
    : defaultKinds{defaultKinds}, cooked{cooked} {
  context = std::make_unique<M::MLIRContext>();
  module = std::make_unique<M::ModuleOp>(
      M::ModuleOp::create(M::UnknownLoc::get(context.get())));
}
