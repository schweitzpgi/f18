//===-- Bridge.cc -- bridge to lower to MLIR ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/lower/Bridge.h"
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
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/Parser.h"
#include "mlir/Target/LLVMIR.h"
#include "llvm/Support/CommandLine.h"

namespace {

llvm::cl::opt<bool>
    dumpBeforeFir("fdebug-dump-pre-fir", llvm::cl::init(false),
                  llvm::cl::desc("dump the IR tree prior to FIR"));

llvm::cl::opt<bool>
    disableToDoAssertions("disable-burnside-todo",
                          llvm::cl::desc("disable burnside bridge asserts"),
                          llvm::cl::init(false), llvm::cl::Hidden);

#undef TODO
#define TODO() assert(false && "not implemented yet")

using SelectCaseConstruct = Fortran::parser::CaseConstruct;
using SelectRankConstruct = Fortran::parser::SelectRankConstruct;
using SelectTypeConstruct = Fortran::parser::SelectTypeConstruct;

using CFGSinkListType = llvm::SmallVector<Fortran::lower::pft::Evaluation *, 2>;
using CFGMapType =
    llvm::DenseMap<Fortran::lower::pft::Evaluation *, CFGSinkListType *>;

constexpr static bool isStopStmt(const Fortran::parser::StopStmt &stm) {
  return std::get<Fortran::parser::StopStmt::Kind>(stm.t) ==
         Fortran::parser::StopStmt::Kind::Stop;
}

// CfgBuilder implementation
#include "CFGBuilder.h"

#undef TODO
#define TODO()                                                                 \
  {                                                                            \
    if (disableToDoAssertions)                                                 \
      mlir::emitError(toLocation(), __FILE__)                                  \
          << ":" << __LINE__ << " not implemented";                            \
    else                                                                       \
      assert(false && "not yet implemented");                                  \
  }

/// Converter from PFT to FIR
///
/// After building the PFT and decorating it, the FirConverter processes that
/// representation and lowers it to the FIR executable representation.
class FirConverter : public Fortran::lower::AbstractConverter {
  using LabelMapType =
      std::map<Fortran::lower::pft::Evaluation *, mlir::Block *>;
  using Closure = std::function<void(const LabelMapType &)>;

  //
  // Helper function members
  //

  mlir::Value createFIRAddr(mlir::Location loc,
                            const Fortran::semantics::SomeExpr *expr) {
    return createSomeAddress(loc, *this, *expr, localSymbols, intrinsics);
  }

  mlir::Value createFIRExpr(mlir::Location loc,
                            const Fortran::semantics::SomeExpr *expr) {
    return createSomeExpression(loc, *this, *expr, localSymbols, intrinsics);
  }
  mlir::Value createLogicalExprAsI1(mlir::Location loc,
                                    const Fortran::semantics::SomeExpr *expr) {
    return createI1LogicalExpression(loc, *this, *expr, localSymbols,
                                     intrinsics);
  }
  mlir::Value createTemporary(mlir::Location loc,
                              const Fortran::semantics::Symbol &sym) {
    return Fortran::lower::createTemporary(loc, *builder, localSymbols,
                                           genType(sym), &sym);
  }

  mlir::FuncOp genFunctionFIR(llvm::StringRef callee,
                              mlir::FunctionType funcTy) {
    if (auto func = Fortran::lower::getNamedFunction(module, callee))
      return func;
    return createFunction(*this, callee, funcTy);
  }

  static bool inMainProgram(Fortran::lower::pft::Evaluation *cstr) {
    return std::visit(Fortran::common::visitors{
                          [](Fortran::lower::pft::FunctionLikeUnit *c) {
                            return c->isMainProgram();
                          },
                          [&](Fortran::lower::pft::Evaluation *c) {
                            return inMainProgram(c);
                          },
                          [](auto *) { return false; },
                      },
                      cstr->parent.p);
  }

  static const Fortran::parser::SubroutineStmt *
  inSubroutine(Fortran::lower::pft::Evaluation *cstr) {
    return std::visit(
        Fortran::common::visitors{
            [](Fortran::lower::pft::FunctionLikeUnit *c) {
              return c->getSubroutine();
            },
            [&](Fortran::lower::pft::Evaluation *c) { return inSubroutine(c); },
            [](auto *) -> const Fortran::parser::SubroutineStmt * {
              return nullptr;
            },
        },
        cstr->parent.p);
  }
  static const Fortran::parser::FunctionStmt *
  inFunction(Fortran::lower::pft::Evaluation *cstr) {
    return std::visit(
        Fortran::common::visitors{
            [](Fortran::lower::pft::FunctionLikeUnit *c) {
              return c->getFunction();
            },
            [&](Fortran::lower::pft::Evaluation *c) { return inFunction(c); },
            [](auto *) -> const Fortran::parser::FunctionStmt * {
              return nullptr;
            },
        },
        cstr->parent.p);
  }
  static const Fortran::parser::MpSubprogramStmt *
  inMPSubp(Fortran::lower::pft::Evaluation *cstr) {
    return std::visit(
        Fortran::common::visitors{
            [](Fortran::lower::pft::FunctionLikeUnit *c) {
              return c->getMPSubp();
            },
            [&](Fortran::lower::pft::Evaluation *c) { return inMPSubp(c); },
            [](auto *) -> const Fortran::parser::MpSubprogramStmt * {
              return nullptr;
            },
        },
        cstr->parent.p);
  }

  template <typename A>
  static const Fortran::semantics::SomeExpr *
  getScalarExprOfTuple(const A &tuple) {
    return Fortran::semantics::GetExpr(
        std::get<Fortran::parser::ScalarLogicalExpr>(tuple));
  }
  template <typename A>
  static const Fortran::semantics::SomeExpr *getExprOfTuple(const A &tuple) {
    return Fortran::semantics::GetExpr(
        std::get<Fortran::parser::LogicalExpr>(tuple));
  }
  /// Get the condition expression for a CondGoto evaluation
  const Fortran::semantics::SomeExpr *
  getEvaluationCondition(Fortran::lower::pft::Evaluation &eval) {
    return eval.visit(Fortran::common::visitors{
        [&](const Fortran::parser::IfStmt &stmt) {
          return getScalarExprOfTuple(stmt.t);
        },
        [&](const Fortran::parser::IfThenStmt &stmt) {
          return getScalarExprOfTuple(stmt.t);
        },
        [&](const Fortran::parser::ElseIfStmt &stmt) {
          return getScalarExprOfTuple(stmt.t);
        },
        [&](const Fortran::parser::WhereConstructStmt &stmt) {
          return getExprOfTuple(stmt.t);
        },
        [&](const Fortran::parser::MaskedElsewhereStmt &stmt) {
          return getExprOfTuple(stmt.t);
        },
        [&](auto &) -> const Fortran::semantics::SomeExpr * {
          mlir::emitError(toLocation(), "unexpected conditional branch case");
          return nullptr;
        },
    });
  }

  //
  // Function-like PFT entry and exit statements
  //

  void
  genFIR(const Fortran::parser::Statement<Fortran::parser::ProgramStmt> &stmt,
         std::string &name, const Fortran::semantics::Symbol *&) {
    setCurrentPosition(stmt.source);
    name = uniquer.doProgramEntry();
  }
  void genFIR(
      const Fortran::parser::Statement<Fortran::parser::EndProgramStmt> &stmt,
      std::string &, const Fortran::semantics::Symbol *&) {
    setCurrentPosition(stmt.source);
    genFIR(stmt.statement);
  }
  void
  genFIR(const Fortran::parser::Statement<Fortran::parser::FunctionStmt> &stmt,
         std::string &name, const Fortran::semantics::Symbol *&symbol) {
    setCurrentPosition(stmt.source);
    auto &n = std::get<Fortran::parser::Name>(stmt.statement.t);
    symbol = n.symbol;
    assert(symbol && "Name resolution failure");
    name = mangleName(*symbol);
  }
  void genFIR(
      const Fortran::parser::Statement<Fortran::parser::EndFunctionStmt> &stmt,
      std::string &, const Fortran::semantics::Symbol *&symbol) {
    setCurrentPosition(stmt.source);
    assert(symbol);
    genFIRFunctionReturn(*symbol);
  }
  void genFIR(
      const Fortran::parser::Statement<Fortran::parser::SubroutineStmt> &stmt,
      std::string &name, const Fortran::semantics::Symbol *&symbol) {
    setCurrentPosition(stmt.source);
    auto &n = std::get<Fortran::parser::Name>(stmt.statement.t);
    symbol = n.symbol;
    assert(symbol && "Name resolution failure");
    name = mangleName(*symbol);
  }
  void
  genFIR(const Fortran::parser::Statement<Fortran::parser::EndSubroutineStmt>
             &stmt,
         std::string &, const Fortran::semantics::Symbol *&) {
    setCurrentPosition(stmt.source);
    genFIR(stmt.statement);
  }
  void genFIR(
      const Fortran::parser::Statement<Fortran::parser::MpSubprogramStmt> &stmt,
      std::string &name, const Fortran::semantics::Symbol *&symbol) {
    setCurrentPosition(stmt.source);
    auto &n = stmt.statement.v;
    name = n.ToString();
    symbol = n.symbol;
  }
  void
  genFIR(const Fortran::parser::Statement<Fortran::parser::EndMpSubprogramStmt>
             &stmt,
         std::string &, const Fortran::semantics::Symbol *&) {
    setCurrentPosition(stmt.source);
    genFIR(stmt.statement);
  }

  //
  // Termination of symbolically referenced execution units
  //

  /// END of program
  ///
  /// Generate the cleanup block before the program exits
  void genFIRProgramExit() { builder->create<mlir::ReturnOp>(toLocation()); }
  void genFIR(const Fortran::parser::EndProgramStmt &) { genFIRProgramExit(); }

  /// END of procedure-like constructs
  ///
  /// Generate the cleanup block before the procedure exits
  void genFIRFunctionReturn(const Fortran::semantics::Symbol &functionSymbol) {
    const auto &details =
        functionSymbol.get<Fortran::semantics::SubprogramDetails>();
    mlir::Value resultRef = localSymbols.lookupSymbol(details.result());
    mlir::Value r = builder->create<fir::LoadOp>(toLocation(), resultRef);
    builder->create<mlir::ReturnOp>(toLocation(), r);
  }
  template <typename A>
  void genFIRProcedureExit(const A *) {
    // FIXME: alt-returns
    builder->create<mlir::ReturnOp>(toLocation());
  }
  void genFIR(const Fortran::parser::EndSubroutineStmt &) {
    genFIRProcedureExit(
        static_cast<const Fortran::parser::SubroutineStmt *>(nullptr));
  }
  void genFIR(const Fortran::parser::EndMpSubprogramStmt &) {
    genFIRProcedureExit(
        static_cast<const Fortran::parser::MpSubprogramStmt *>(nullptr));
  }

  //
  // Statements that have control-flow semantics
  //

  // Conditional goto control-flow semantics
  void genFIREvalCondGoto(Fortran::lower::pft::Evaluation &eval) {
    genFIR(eval);
    auto targets = findTargetsOf(eval);
    auto *expr = getEvaluationCondition(eval);
    assert(expr && "condition expression missing");
    auto cond = createLogicalExprAsI1(toLocation(), expr);
    genFIRCondBranch(cond, targets[0], targets[1]);
  }

  void genFIRCondBranch(mlir::Value cond,
                        Fortran::lower::pft::Evaluation *trueDest,
                        Fortran::lower::pft::Evaluation *falseDest) {
    using namespace std::placeholders;
    localEdgeQ.emplace_back(std::bind(
        [](mlir::OpBuilder *builder, mlir::Block *block, mlir::Value cnd,
           Fortran::lower::pft::Evaluation *trueDest,
           Fortran::lower::pft::Evaluation *falseDest, mlir::Location location,
           const LabelMapType &map) {
          llvm::SmallVector<mlir::Value, 2> blk;
          builder->setInsertionPointToEnd(block);
          auto tdp = map.find(trueDest);
          auto fdp = map.find(falseDest);
          assert(tdp != map.end() && fdp != map.end());
          builder->create<mlir::CondBranchOp>(location, cnd, tdp->second, blk,
                                              fdp->second, blk);
        },
        builder, builder->getInsertionBlock(), cond, trueDest, falseDest,
        toLocation(), _1));
  }

  // Goto control-flow semantics
  //
  // These are unconditional jumps. There is nothing to evaluate.
  void genFIREvalGoto(Fortran::lower::pft::Evaluation &eval) {
    using namespace std::placeholders;
    localEdgeQ.emplace_back(std::bind(
        [](mlir::OpBuilder *builder, mlir::Block *block,
           Fortran::lower::pft::Evaluation *dest, mlir::Location location,
           const LabelMapType &map) {
          builder->setInsertionPointToEnd(block);
          assert(map.find(dest) != map.end() && "no destination");
          builder->create<mlir::BranchOp>(location, map.find(dest)->second);
        },
        builder, builder->getInsertionBlock(), findSinkOf(eval), toLocation(),
        _1));
  }

  // Indirect goto control-flow semantics
  //
  // For assigned gotos, which is an obsolescent feature. Lower to a switch.
  void genFIREvalIndGoto(Fortran::lower::pft::Evaluation &eval) {
    genFIR(eval);
    // FIXME
  }

  // IO statements that have control-flow semantics
  //
  // First lower the IO statement and then do the multiway switch op
  void genFIREvalIoSwitch(Fortran::lower::pft::Evaluation &eval) {
    genFIR(eval);
    genFIRIOSwitch(eval);
  }
  void genFIRIOSwitch(Fortran::lower::pft::Evaluation &) { TODO(); }

  // Iterative loop control-flow semantics
  void genFIREvalIterative(Fortran::lower::pft::Evaluation &) { TODO(); }

  void switchInsertionPointToWhere(fir::WhereOp &where) {
    builder->setInsertionPointToStart(&where.whereRegion().front());
  }
  void switchInsertionPointToOtherwise(fir::WhereOp &where) {
    builder->setInsertionPointToStart(&where.otherRegion().front());
  }
  template <typename A>
  void genWhereCondition(fir::WhereOp &where, const A *stmt) {
    auto cond = createLogicalExprAsI1(
        toLocation(),
        Fortran::semantics::GetExpr(
            std::get<Fortran::parser::ScalarLogicalExpr>(stmt->t)));
    where = builder->create<fir::WhereOp>(toLocation(), cond, true);
    switchInsertionPointToWhere(where);
  }

  mlir::Value genFIRLoopIndex(const Fortran::parser::ScalarExpr &x) {
    return builder->create<fir::ConvertOp>(
        toLocation(), mlir::IndexType::get(&mlirContext),
        genExprValue(*Fortran::semantics::GetExpr(x)));
  }

  /// Structured control op (`fir.loop`, `fir.where`)
  ///
  /// Convert a DoConstruct to a `fir.loop` op.
  /// Convert an IfConstruct to a `fir.where` op.
  ///
  void genFIREvalStructuredOp(Fortran::lower::pft::Evaluation &eval) {
    // TODO: array expressions, FORALL, WHERE ...

    // process the list of Evaluations
    assert(eval.subs && "eval must have a body");
    auto *insPt = builder->getInsertionBlock();

    if (const auto *doConstruct{eval.getIf<Fortran::parser::DoConstruct>()}) {
      if (const auto &loopControl{doConstruct->GetLoopControl()}) {
        std::visit(
            Fortran::common::visitors{
                [&](const Fortran::parser::LoopControl::Bounds &x) {
                  mlir::Value lo = genFIRLoopIndex(x.lower);
                  mlir::Value hi = genFIRLoopIndex(x.upper);
                  auto step =
                      x.step.has_value()
                          ? genExprValue(*Fortran::semantics::GetExpr(*x.step))
                          : mlir::Value{};
                  auto *sym = x.name.thing.symbol;
                  Fortran::lower::LoopBuilder{*builder, toLocation()}
                      .createLoop(lo, hi, step,
                                  [&](Fortran::lower::OpBuilderWrapper &handler,
                                      mlir::Value index) {
                                    // TODO: should push this cast down to the
                                    // uses
                                    auto cvt = handler.create<fir::ConvertOp>(
                                        genType(*sym), index);
                                    localSymbols.pushShadowSymbol(*sym, cvt);
                                    for (auto &e : *eval.subs) {
                                      genFIR(e);
                                    }
                                    localSymbols.popShadowSymbol();
                                  });
                },
                [&](const Fortran::parser::ScalarLogicalExpr &) {
                  // we should never reach here
                  mlir::emitError(toLocation(), "loop lacks iteration space");
                },
                [&](const Fortran::parser::LoopControl::Concurrent &x) {
                  // FIXME: can project a multi-dimensional space
                  Fortran::lower::LoopBuilder{*builder, toLocation()}
                      .createLoop(
                          mlir::Value{}, mlir::Value{},
                          [&](Fortran::lower::OpBuilderWrapper &, mlir::Value) {
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
    } else if (eval.isA<Fortran::parser::IfConstruct>()) {
      // Construct fir.where
      fir::WhereOp where;
      for (auto &e : *eval.subs) {
        if (auto *s{e.getIf<Fortran::parser::IfThenStmt>()}) {
          // fir.where op
          genWhereCondition(where, s);
        } else if (auto *s{e.getIf<Fortran::parser::ElseIfStmt>()}) {
          // otherwise block, then nested fir.where
          switchInsertionPointToOtherwise(where);
          genWhereCondition(where, s);
        } else if (e.isA<Fortran::parser::ElseStmt>()) {
          // otherwise block
          switchInsertionPointToOtherwise(where);
        } else if (e.isA<Fortran::parser::EndIfStmt>()) {
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
  void genFIREvalReturn(Fortran::lower::pft::Evaluation &eval) {
    // Handled case-by-case
    // FIXME: think about moving the case code here
  }

  // Multiway switch control-flow semantics
  void genFIREvalSwitch(Fortran::lower::pft::Evaluation &eval) {
    genFIR(eval);
    // FIXME
  }

  // Terminate process control-flow semantics
  //
  // Call a runtime routine that does not return
  void genFIREvalTerminate(Fortran::lower::pft::Evaluation &eval) {
    genFIR(eval);
    builder->create<fir::UnreachableOp>(toLocation());
  }

  // No control-flow
  void genFIREvalNone(Fortran::lower::pft::Evaluation &eval) { genFIR(eval); }

  mlir::FuncOp getFunc(llvm::StringRef name, mlir::FunctionType ty) {
    if (auto func = Fortran::lower::getNamedFunction(module, name)) {
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
  void genFIR(const Fortran::parser::CallStmt &stmt) {
    llvm::SmallVector<mlir::Type, 8> argTy;
    llvm::SmallVector<mlir::Type, 2> resTy;
    llvm::StringRef funName;
    std::vector<Fortran::semantics::Symbol *> argsList;
    setCurrentPosition(stmt.v.source);
    std::visit(Fortran::common::visitors{
                   [&](const Fortran::parser::Name &name) {
                     auto *sym = name.symbol;
                     auto n = sym->name();
                     funName = llvm::StringRef{n.begin(), n.size()};
                     auto &details =
                         sym->get<Fortran::semantics::SubprogramDetails>();
                     // TODO ProcEntityDetails?
                     // TODO bindName()?
                     argsList = details.dummyArgs();
                   },
                   [&](const Fortran::parser::ProcComponentRef &) { TODO(); },
               },
               std::get<Fortran::parser::ProcedureDesignator>(stmt.v.t).u);
    for (auto *d : argsList) {
      Fortran::semantics::SymbolRef sr = *d;
      // FIXME:
      argTy.push_back(fir::ReferenceType::get(genType(sr)));
    }
    auto funTy = mlir::FunctionType::get(argTy, resTy, builder->getContext());
    // FIXME: mangle name
    mlir::FuncOp func = getFunc(funName, funTy);
    (void)func; // FIXME
    std::vector<mlir::Value> actuals;
    for (auto &aa :
         std::get<std::list<Fortran::parser::ActualArgSpec>>(stmt.v.t)) {
      auto &kw = std::get<std::optional<Fortran::parser::Keyword>>(aa.t);
      auto &arg = std::get<Fortran::parser::ActualArg>(aa.t);
      mlir::Value fe;
      std::visit(
          Fortran::common::visitors{
              [&](const Fortran::common::Indirection<Fortran::parser::Expr>
                      &e) {
                // FIXME: needs to match argument, assumes trivial by-ref
                fe = genExprAddr(*Fortran::semantics::GetExpr(e));
              },
              [&](const Fortran::parser::AltReturnSpec &) { TODO(); },
              [&](const Fortran::parser::ActualArg::PercentRef &) { TODO(); },
              [&](const Fortran::parser::ActualArg::PercentVal &) { TODO(); },
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

  void genFIR(const Fortran::parser::IfStmt &) { TODO(); }
  void genFIR(const Fortran::parser::WaitStmt &) { TODO(); }
  void genFIR(const Fortran::parser::WhereStmt &) { TODO(); }
  void genFIR(const Fortran::parser::ComputedGotoStmt &stmt) {
    auto *exp = Fortran::semantics::GetExpr(
        std::get<Fortran::parser::ScalarIntExpr>(stmt.t));
    auto e1 = genExprValue(*exp);
    (void)e1;
    TODO();
  }
  void genFIR(const Fortran::parser::ForallStmt &) { TODO(); }
  void genFIR(const Fortran::parser::ArithmeticIfStmt &stmt) {
    auto *exp =
        Fortran::semantics::GetExpr(std::get<Fortran::parser::Expr>(stmt.t));
    auto e1 = genExprValue(*exp);
    (void)e1;
    TODO();
  }
  void genFIR(const Fortran::parser::AssignedGotoStmt &) { TODO(); }

  void genFIR(const Fortran::parser::AssociateConstruct &) { TODO(); }
  void genFIR(const Fortran::parser::BlockConstruct &) { TODO(); }
  void genFIR(const Fortran::parser::ChangeTeamConstruct &) { TODO(); }
  void genFIR(const Fortran::parser::CriticalConstruct &) { TODO(); }
  void genFIR(const Fortran::parser::DoConstruct &) { TODO(); }
  void genFIR(const Fortran::parser::IfConstruct &) { TODO(); }

  void genFIR(const SelectCaseConstruct &) { TODO(); }
  void genFIR(const SelectRankConstruct &) { TODO(); }
  void genFIR(const SelectTypeConstruct &) { TODO(); }

  void genFIR(const Fortran::parser::WhereConstruct &) { TODO(); }

  /// Lower FORALL construct (See 10.2.4)
  void genFIR(const Fortran::parser::ForallConstruct &forall) {
    auto &stmt = std::get<
        Fortran::parser::Statement<Fortran::parser::ForallConstructStmt>>(
        forall.t);
    setCurrentPosition(stmt.source);
    auto &fas = stmt.statement;
    auto &ctrl =
        std::get<
            Fortran::common::Indirection<Fortran::parser::ConcurrentHeader>>(
            fas.t)
            .value();
    (void)ctrl;
    for (auto &s :
         std::get<std::list<Fortran::parser::ForallBodyConstruct>>(forall.t)) {
      std::visit(
          Fortran::common::visitors{
              [&](const Fortran::parser::Statement<
                  Fortran::parser::ForallAssignmentStmt> &b) {
                setCurrentPosition(b.source);
                genFIR(b.statement);
              },
              [&](const Fortran::parser::Statement<Fortran::parser::WhereStmt>
                      &b) {
                setCurrentPosition(b.source);
                genFIR(b.statement);
              },
              [&](const Fortran::parser::WhereConstruct &b) { genFIR(b); },
              [&](const Fortran::common::Indirection<
                  Fortran::parser::ForallConstruct> &b) { genFIR(b.value()); },
              [&](const Fortran::parser::Statement<Fortran::parser::ForallStmt>
                      &b) {
                setCurrentPosition(b.source);
                genFIR(b.statement);
              },
          },
          s.u);
    }
    TODO();
  }
  void genFIR(const Fortran::parser::ForallAssignmentStmt &s) {
    std::visit([&](auto &b) { genFIR(b); }, s.u);
  }

  void genFIR(const Fortran::parser::CompilerDirective &) { TODO(); }
  void genFIR(const Fortran::parser::OpenMPConstruct &) { TODO(); }
  void genFIR(const Fortran::parser::OmpEndLoopDirective &) { TODO(); }

  void genFIR(const Fortran::parser::AssociateStmt &) { TODO(); }
  void genFIR(const Fortran::parser::EndAssociateStmt &) { TODO(); }
  void genFIR(const Fortran::parser::BlockStmt &) { TODO(); }
  void genFIR(const Fortran::parser::EndBlockStmt &) { TODO(); }
  void genFIR(const Fortran::parser::SelectCaseStmt &) { TODO(); }
  void genFIR(const Fortran::parser::CaseStmt &) { TODO(); }
  void genFIR(const Fortran::parser::EndSelectStmt &) { TODO(); }
  void genFIR(const Fortran::parser::ChangeTeamStmt &) { TODO(); }
  void genFIR(const Fortran::parser::EndChangeTeamStmt &) { TODO(); }
  void genFIR(const Fortran::parser::CriticalStmt &) { TODO(); }
  void genFIR(const Fortran::parser::EndCriticalStmt &) { TODO(); }

  // Do loop is handled by EvalIterative(), EvalStructuredOp()
  void genFIR(const Fortran::parser::NonLabelDoStmt &) {} // do nothing
  void genFIR(const Fortran::parser::EndDoStmt &) {}      // do nothing

  // If-Then-Else is handled by EvalCondGoto(), EvalStructuredOp()
  void genFIR(const Fortran::parser::IfThenStmt &) {} // do nothing
  void genFIR(const Fortran::parser::ElseIfStmt &) {} // do nothing
  void genFIR(const Fortran::parser::ElseStmt &) {}   // do nothing
  void genFIR(const Fortran::parser::EndIfStmt &) {}  // do nothing

  void genFIR(const Fortran::parser::SelectRankStmt &) { TODO(); }
  void genFIR(const Fortran::parser::SelectRankCaseStmt &) { TODO(); }
  void genFIR(const Fortran::parser::SelectTypeStmt &) { TODO(); }
  void genFIR(const Fortran::parser::TypeGuardStmt &) { TODO(); }

  void genFIR(const Fortran::parser::WhereConstructStmt &) { TODO(); }
  void genFIR(const Fortran::parser::MaskedElsewhereStmt &) { TODO(); }
  void genFIR(const Fortran::parser::ElsewhereStmt &) { TODO(); }
  void genFIR(const Fortran::parser::EndWhereStmt &) { TODO(); }
  void genFIR(const Fortran::parser::ForallConstructStmt &) { TODO(); }
  void genFIR(const Fortran::parser::EndForallStmt &) { TODO(); }

  //
  // Statements that do not have control-flow semantics
  //

  // IO statements (see io.h)

  void genFIR(const Fortran::parser::BackspaceStmt &stmt) {
    genBackspaceStatement(*this, stmt);
  }
  void genFIR(const Fortran::parser::CloseStmt &stmt) {
    genCloseStatement(*this, stmt);
  }
  void genFIR(const Fortran::parser::EndfileStmt &stmt) {
    genEndfileStatement(*this, stmt);
  }
  void genFIR(const Fortran::parser::FlushStmt &stmt) {
    genFlushStatement(*this, stmt);
  }
  void genFIR(const Fortran::parser::InquireStmt &stmt) {
    genInquireStatement(*this, stmt);
  }
  void genFIR(const Fortran::parser::OpenStmt &stmt) {
    genOpenStatement(*this, stmt);
  }
  void genFIR(const Fortran::parser::PrintStmt &stmt) {
    genPrintStatement(*this, stmt);
  }
  void genFIR(const Fortran::parser::ReadStmt &stmt) {
    genReadStatement(*this, stmt);
  }
  void genFIR(const Fortran::parser::RewindStmt &stmt) {
    genRewindStatement(*this, stmt);
  }
  void genFIR(const Fortran::parser::WriteStmt &stmt) {
    genWriteStatement(*this, stmt);
  }

  void genFIR(const Fortran::parser::AllocateStmt &) { TODO(); }

  void
  genCharacterAssignement(const Fortran::evaluate::Assignment &assignment) {
    // Helper to get address and length from an Expr that is a character
    // variable designator
    auto getAddrAndLength =
        [&](const Fortran::lower::SomeExpr &charDesignatorExpr)
        -> Fortran::lower::CharacterOpsBuilder::CharValue {
      mlir::Value addr = genExprAddr(charDesignatorExpr);
      const auto &charExpr =
          std::get<Fortran::evaluate::Expr<Fortran::evaluate::SomeCharacter>>(
              charDesignatorExpr.u);
      auto lenExpr = charExpr.LEN();
      assert(lenExpr && "could not get expression to compute character length");
      mlir::Value len =
          genExprValue(Fortran::evaluate::AsGenericExpr(std::move(*lenExpr)));
      return Fortran::lower::CharacterOpsBuilder::CharValue{addr, len};
    };

    Fortran::lower::CharacterOpsBuilder charBuilder{*builder, toLocation()};

    // RHS evaluation.
    // FIXME:  Only works with rhs that are variable reference.
    // Other expression evaluation are not simple copies.
    auto rhs = getAddrAndLength(assignment.rhs);
    // A temp is needed to evaluate rhs until proven it does not depend on lhs.
    auto tempToEvalRhs =
        charBuilder.createTemp(rhs.getCharacterType(), rhs.len);
    charBuilder.createCopy(tempToEvalRhs, rhs, rhs.len);

    // Copy the minimum of the lhs and rhs lengths and pad the lhs remainder
    auto lhs = getAddrAndLength(assignment.lhs);
    auto cmpLen = charBuilder.create<mlir::CmpIOp>(mlir::CmpIPredicate::slt,
                                                   lhs.len, rhs.len);
    auto copyCount =
        charBuilder.create<mlir::SelectOp>(cmpLen, lhs.len, rhs.len);
    charBuilder.createCopy(lhs, tempToEvalRhs, copyCount);
    charBuilder.createPadding(lhs, copyCount, lhs.len);
  }

  void genFIR(const Fortran::parser::AssignmentStmt &stmt) {
    assert(stmt.typedAssignment && stmt.typedAssignment->v &&
           "assignment analysis failed");
    const auto &assignment = *stmt.typedAssignment->v;
    std::visit( // better formatting
        Fortran::common::visitors{
            [&](const Fortran::evaluate::Assignment::Intrinsic &) {
              const auto *sym =
                  Fortran::evaluate::UnwrapWholeSymbolDataRef(assignment.lhs);
              if (sym && Fortran::semantics::IsAllocatable(*sym)) {
                // Assignment of allocatable are more complex, the lhs
                // may need to be deallocated/reallocated.
                // See Fortran 2018 10.2.1.3 p3
                TODO();
              } else if (sym && Fortran::semantics::IsPointer(*sym)) {
                // Target of the pointer must be assigned.
                // See Fortran 2018 10.2.1.3 p2
                TODO();
              } else if (assignment.lhs.Rank() > 0) {
                // Array assignment
                // See Fortran 2018 10.2.1.3 p5, p6, and p7
                TODO();
              } else {
                // Scalar assignments
                auto lhsType = assignment.lhs.GetType();
                assert(lhsType && "lhs cannot be typeless");
                switch (lhsType->category()) {
                case Fortran::lower::IntegerCat:
                case Fortran::lower::RealCat:
                case Fortran::lower::ComplexCat:
                case Fortran::lower::LogicalCat:
                  // Fortran 2018 10.2.1.3 p8 and p9
                  // Conversions are already inserted by semantic
                  // analysis.
                  builder->create<fir::StoreOp>(toLocation(),
                                                genExprValue(assignment.rhs),
                                                genExprAddr(assignment.lhs));
                  break;
                case Fortran::lower::CharacterCat:
                  // Fortran 2018 10.2.1.3 p10 and p11
                  genCharacterAssignement(assignment);
                  break;
                case Fortran::lower::DerivedCat:
                  // Fortran 2018 10.2.1.3 p12 and p13
                  TODO();
                  break;
                }
              }
            },
            [&](const Fortran::evaluate::ProcedureRef &) {
              // Defined assignment: call ProcRef
              TODO();
            },
            [&](const Fortran::evaluate::Assignment::BoundsSpec &) {
              // Pointer assignment with possibly empty bounds-spec
              TODO();
            },
            [&](const Fortran::evaluate::Assignment::BoundsRemapping &) {
              // Pointer assignment with bounds-remapping
              TODO();
            },
        },
        assignment.u);
  }

  void genFIR(const Fortran::parser::ContinueStmt &) {} // do nothing
  void genFIR(const Fortran::parser::DeallocateStmt &) { TODO(); }
  void genFIR(const Fortran::parser::EventPostStmt &) {
    // call some runtime routine
    TODO();
  }
  void genFIR(const Fortran::parser::EventWaitStmt &) {
    // call some runtime routine
    TODO();
  }

  void genFIR(const Fortran::parser::FormTeamStmt &) { TODO(); }
  void genFIR(const Fortran::parser::LockStmt &) {
    // call some runtime routine
    TODO();
  }

  /// Nullify pointer object list
  ///
  /// For each pointer object, reset the pointer to a disassociated status.
  /// We do this by setting each pointer to null.
  void genFIR(const Fortran::parser::NullifyStmt &stmt) {
    for (auto &po : stmt.v) {
      std::visit(
          Fortran::common::visitors{
              [&](const Fortran::parser::Name &sym) {
                auto ty = genType(*sym.symbol);
                auto load = builder->create<fir::LoadOp>(
                    toLocation(), localSymbols.lookupSymbol(*sym.symbol));
                auto idxTy = mlir::IndexType::get(&mlirContext);
                auto zero = builder->create<mlir::ConstantOp>(
                    toLocation(), idxTy, builder->getIntegerAttr(idxTy, 0));
                auto cast =
                    builder->create<fir::ConvertOp>(toLocation(), ty, zero);
                builder->create<fir::StoreOp>(toLocation(), cast, load);
              },
              [&](const Fortran::parser::StructureComponent &) { TODO(); },
          },
          po.u);
    }
  }
  void genFIR(const Fortran::parser::PointerAssignmentStmt &) { TODO(); }

  void genFIR(const Fortran::parser::SyncAllStmt &) {
    // call some runtime routine
    TODO();
  }
  void genFIR(const Fortran::parser::SyncImagesStmt &) {
    // call some runtime routine
    TODO();
  }
  void genFIR(const Fortran::parser::SyncMemoryStmt &) {
    // call some runtime routine
    TODO();
  }
  void genFIR(const Fortran::parser::SyncTeamStmt &) {
    // call some runtime routine
    TODO();
  }
  void genFIR(const Fortran::parser::UnlockStmt &) {
    // call some runtime routine
    TODO();
  }

  void genFIR(const Fortran::parser::AssignStmt &) { TODO(); }
  void genFIR(const Fortran::parser::FormatStmt &) { TODO(); }
  void genFIR(const Fortran::parser::EntryStmt &) { TODO(); }
  void genFIR(const Fortran::parser::PauseStmt &) {
    // call some runtime routine
    TODO();
  }
  void genFIR(const Fortran::parser::DataStmt &) { TODO(); }
  void genFIR(const Fortran::parser::NamelistStmt &) { TODO(); }

  // call FAIL IMAGE in runtime
  void genFIR(const Fortran::parser::FailImageStmt &stmt) {
    auto callee = genRuntimeFunction(
        Fortran::lower::RuntimeEntryCode::FailImageStatement, *builder);
    llvm::SmallVector<mlir::Value, 1> operands; // FAIL IMAGE has no args
    builder->create<mlir::CallOp>(toLocation(), callee, operands);
  }

  // call STOP, ERROR STOP in runtime
  void genFIR(const Fortran::parser::StopStmt &stm) {
    auto callee = genRuntimeFunction(
        Fortran::lower::RuntimeEntryCode::StopStatement, *builder);
    // TODO: 3 args: stop-code-opt, ierror, quiet-opt
    // auto isError{genFIRLo!isStopStmt(stmt)}
    builder->create<mlir::CallOp>(toLocation(), callee, llvm::None);
  }

  // gen expression, if any; share code with END of procedure
  void genFIR(const Fortran::parser::ReturnStmt &) {
    if (inMainProgram(currentEvaluation)) {
      builder->create<mlir::ReturnOp>(toLocation());
    } else if (auto *stmt = inSubroutine(currentEvaluation)) {
      genFIRProcedureExit(stmt);
    } else if (auto *stmt = inFunction(currentEvaluation)) {
      auto *symbol = std::get<Fortran::parser::Name>(stmt->t).symbol;
      assert(symbol);
      genFIRFunctionReturn(*symbol);
    } else if (auto *stmt = inMPSubp(currentEvaluation)) {
      genFIRProcedureExit(stmt);
    } else {
      mlir::emitError(toLocation(), "unknown subprogram type");
    }
  }

  // stubs for generic goto statements; see genFIREvalGoto()
  void genFIR(const Fortran::parser::CycleStmt &) {} // do nothing
  void genFIR(const Fortran::parser::ExitStmt &) {}  // do nothing
  void genFIR(const Fortran::parser::GotoStmt &) {}  // do nothing

  void genFIR(Fortran::lower::pft::Evaluation &eval) {
    currentEvaluation = &eval;
    eval.visit(Fortran::common::visitors{
        [&](const auto &p) { genFIR(p); },
        [](const Fortran::lower::pft::CGJump &) { /* do nothing */ },
    });
  }

  /// Lower an Evaluation
  ///
  /// If the Evaluation is annotated, we can attempt to lower it by the class of
  /// annotation. Otherwise, attempt to lower the Evaluation on a case-by-case
  /// basis.
  void lowerEval(Fortran::lower::pft::Evaluation &eval) {
    setCurrentPosition(eval.pos);
    if (eval.isControlTarget()) {
      // start a new block
    }
    switch (eval.cfg) {
    case Fortran::lower::pft::CFGAnnotation::None:
      genFIREvalNone(eval);
      break;
    case Fortran::lower::pft::CFGAnnotation::Goto:
      genFIREvalGoto(eval);
      break;
    case Fortran::lower::pft::CFGAnnotation::CondGoto:
      genFIREvalCondGoto(eval);
      break;
    case Fortran::lower::pft::CFGAnnotation::IndGoto:
      genFIREvalIndGoto(eval);
      break;
    case Fortran::lower::pft::CFGAnnotation::IoSwitch:
      genFIREvalIoSwitch(eval);
      break;
    case Fortran::lower::pft::CFGAnnotation::Switch:
      genFIREvalSwitch(eval);
      break;
    case Fortran::lower::pft::CFGAnnotation::Iterative:
      genFIREvalIterative(eval);
      break;
    case Fortran::lower::pft::CFGAnnotation::FirStructuredOp:
      genFIREvalStructuredOp(eval);
      break;
    case Fortran::lower::pft::CFGAnnotation::Return:
      genFIREvalReturn(eval);
      break;
    case Fortran::lower::pft::CFGAnnotation::Terminate:
      genFIREvalTerminate(eval);
      break;
    }
  }

  mlir::FuncOp createNewFunction(llvm::StringRef name,
                                 const Fortran::semantics::Symbol *symbol) {
    // get arguments and return type if any, otherwise just use empty vectors
    llvm::SmallVector<mlir::Type, 8> args;
    llvm::SmallVector<mlir::Type, 2> results;
    auto funcTy = symbol ? genFunctionType(*symbol)
                         : mlir::FunctionType::get(args, results, &mlirContext);
    return createFunction(*this, name, funcTy);
  }

  /// Prepare to translate a new function
  void startNewFunction(Fortran::lower::pft::FunctionLikeUnit &funit,
                        llvm::StringRef name,
                        const Fortran::semantics::Symbol *symbol) {
    mlir::FuncOp func = Fortran::lower::getNamedFunction(module, name);
    if (!func)
      func = createNewFunction(name, symbol);
    func.addEntryBlock();
    assert(!builder && "expected nullptr");
    builder = new mlir::OpBuilder(func);
    assert(builder && "OpBuilder did not instantiate");
    builder->setInsertionPointToStart(&func.front());

    // plumb function's arguments
    if (symbol) {
      auto *entryBlock = &func.front();
      auto *details =
          symbol->detailsIf<Fortran::semantics::SubprogramDetails>();
      assert(details && "details for semantics symbol must be subprogram");
      for (const auto &v :
           llvm::zip(details->dummyArgs(), entryBlock->getArguments())) {
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
    for (auto &edgeFunc : localEdgeQ)
      edgeFunc(localBlockMap);
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
  void lowerFunc(Fortran::lower::pft::FunctionLikeUnit &func,
                 llvm::ArrayRef<llvm::StringRef> modules,
                 llvm::Optional<llvm::StringRef> host = {}) {
    std::string name;
    const Fortran::semantics::Symbol *symbol = nullptr;

    if (func.beginStmt) {
      currentEvaluation = nullptr;
      std::visit([&](auto *p) { genFIR(*p, name, symbol); }, *func.beginStmt);
    } else {
      name = uniquer.doProgramEntry();
    }

    startNewFunction(func, name, symbol);

    // lower this procedure
    for (auto &e : func.evals)
      lowerEval(e);
    currentEvaluation = nullptr;
    std::visit([&](auto *p) { genFIR(*p, name, symbol); }, func.endStmt);

    endNewFunction();

    // recursively lower internal procedures
    llvm::Optional<llvm::StringRef> optName{name};
    for (auto &f : func.funcs)
      lowerFunc(f, modules, optName);
  }

  void lowerMod(Fortran::lower::pft::ModuleLikeUnit &mod) {
    // FIXME: build the vector of module names
    std::vector<llvm::StringRef> moduleName;

    // FIXME: do we need to visit the module statements?
    for (auto &f : mod.funcs)
      lowerFunc(f, moduleName);
  }

  //
  // Finalization of the CFG structure
  //

  /// Lookup the set of sinks for this source. There must be at least one.
  llvm::ArrayRef<Fortran::lower::pft::Evaluation *>
  findTargetsOf(Fortran::lower::pft::Evaluation &eval) {
    auto iter = cfgMap.find(&eval);
    assert(iter != cfgMap.end());
    return *iter->second;
  }

  /// Lookup the sink for this source. There must be exactly one.
  Fortran::lower::pft::Evaluation *
  findSinkOf(Fortran::lower::pft::Evaluation &eval) {
    auto iter = cfgMap.find(&eval);
    assert((iter != cfgMap.end()) && (iter->second->size() == 1));
    return iter->second->front();
  }

  /// prune the CFG for `f`
  void pruneFunc(Fortran::lower::pft::FunctionLikeUnit &func) {
    // find and cache arcs, etc.
    if (!func.evals.empty())
      CfgBuilder{cfgMap, cfgEdgeSetPool}.run(func);

    // do any internal procedures
    for (auto &f : func.funcs)
      pruneFunc(f);
  }

  void pruneMod(Fortran::lower::pft::ModuleLikeUnit &mod) {
    for (auto &f : mod.funcs)
      pruneFunc(f);
  }

  void setCurrentPosition(const Fortran::parser::CharBlock &pos) {
    if (pos != Fortran::parser::CharBlock{})
      currentPosition = pos;
  }

  //
  // Utility methods
  //

  /// Convert a parser CharBlock to a Location
  mlir::Location toLocation(const Fortran::parser::CharBlock &cb) {
    return genLocation(cb);
  }

  mlir::Location toLocation() { return toLocation(currentPosition); }

  // TODO: should these be moved to convert-expr?
  template <mlir::CmpIPredicate ICMPOPC>
  mlir::Value genCompare(mlir::Value lhs, mlir::Value rhs) {
    auto lty = lhs.getType();
    assert(lty == rhs.getType());
    if (lty.isSignlessIntOrIndex())
      return builder->create<mlir::CmpIOp>(lhs.getLoc(), ICMPOPC, lhs, rhs);
    if (fir::LogicalType::kindof(lty.getKind()))
      return builder->create<mlir::CmpIOp>(lhs.getLoc(), ICMPOPC, lhs, rhs);
    if (fir::CharacterType::kindof(lty.getKind())) {
      // FIXME
      // return builder->create<mlir::CallOp>(lhs->getLoc(), );
    }
    mlir::emitError(toLocation(), "cannot generate operation on this type");
    return {};
  }

  mlir::Value genGE(mlir::Value lhs, mlir::Value rhs) {
    return genCompare<mlir::CmpIPredicate::sge>(lhs, rhs);
  }
  mlir::Value genLE(mlir::Value lhs, mlir::Value rhs) {
    return genCompare<mlir::CmpIPredicate::sle>(lhs, rhs);
  }
  mlir::Value genEQ(mlir::Value lhs, mlir::Value rhs) {
    return genCompare<mlir::CmpIPredicate::eq>(lhs, rhs);
  }
  mlir::Value genAND(mlir::Value lhs, mlir::Value rhs) {
    return builder->create<mlir::AndOp>(lhs.getLoc(), lhs, rhs);
  }

private:
  mlir::MLIRContext &mlirContext;
  const Fortran::parser::CookedSource *cooked;
  mlir::ModuleOp &module;
  const Fortran::common::IntrinsicTypeDefaultKinds &defaults;
  Fortran::lower::IntrinsicLibrary intrinsics;
  mlir::OpBuilder *builder = nullptr;
  fir::NameUniquer &uniquer;
  Fortran::lower::SymMap localSymbols;
  std::list<Closure> localEdgeQ;
  LabelMapType localBlockMap;
  Fortran::parser::CharBlock currentPosition;
  CFGMapType cfgMap;
  std::list<CFGSinkListType> cfgEdgeSetPool;
  Fortran::lower::pft::Evaluation *currentEvaluation =
      nullptr; // FIXME: this is a hack

public:
  FirConverter() = delete;
  FirConverter(const FirConverter &) = delete;
  FirConverter &operator=(const FirConverter &) = delete;
  virtual ~FirConverter() = default;

  explicit FirConverter(Fortran::lower::LoweringBridge &bridge,
                        fir::NameUniquer &uniquer)
      : mlirContext{bridge.getMLIRContext()}, cooked{bridge.getCookedSource()},
        module{bridge.getModule()}, defaults{bridge.getDefaultKinds()},
        intrinsics{Fortran::lower::IntrinsicLibrary(
            Fortran::lower::IntrinsicLibrary::Version::LLVM,
            bridge.getMLIRContext())},
        uniquer{uniquer} {}

  /// Convert the PFT to FIR
  void run(Fortran::lower::pft::Program &pft) {
    // build pruned control
    for (auto &u : pft.getUnits()) {
      std::visit(
          Fortran::common::visitors{
              [&](Fortran::lower::pft::FunctionLikeUnit &f) { pruneFunc(f); },
              [&](Fortran::lower::pft::ModuleLikeUnit &m) { pruneMod(m); },
              [](Fortran::lower::pft::BlockDataUnit &) { /* do nothing */ },
          },
          u);
    }

    // do translation
    for (auto &u : pft.getUnits()) {
      std::visit(
          Fortran::common::visitors{
              [&](Fortran::lower::pft::FunctionLikeUnit &f) {
                lowerFunc(f, {});
              },
              [&](Fortran::lower::pft::ModuleLikeUnit &m) { lowerMod(m); },
              [&](Fortran::lower::pft::BlockDataUnit &) { TODO(); },
          },
          u);
    }
  }

  mlir::FunctionType genFunctionType(Fortran::lower::SymbolRef sym) {
    return Fortran::lower::translateSymbolToFIRFunctionType(&mlirContext,
                                                            defaults, sym);
  }

  //
  // AbstractConverter overrides

  mlir::Value genExprAddr(const Fortran::lower::SomeExpr &expr,
                          mlir::Location *loc = nullptr) override final {
    return createFIRAddr(loc ? *loc : toLocation(), &expr);
  }
  mlir::Value genExprValue(const Fortran::lower::SomeExpr &expr,
                           mlir::Location *loc = nullptr) override final {
    return createFIRExpr(loc ? *loc : toLocation(), &expr);
  }

  mlir::Type genType(const Fortran::evaluate::DataRef &data) override final {
    return Fortran::lower::translateDataRefToFIRType(&mlirContext, defaults,
                                                     data);
  }
  mlir::Type genType(const Fortran::lower::SomeExpr &expr) override final {
    return Fortran::lower::translateSomeExprToFIRType(&mlirContext, defaults,
                                                      &expr);
  }
  mlir::Type genType(Fortran::lower::SymbolRef sym) override final {
    return Fortran::lower::translateSymbolToFIRType(&mlirContext, defaults,
                                                    sym);
  }
  mlir::Type genType(Fortran::common::TypeCategory tc,
                     int kind) override final {
    return Fortran::lower::getFIRType(&mlirContext, defaults, tc, kind);
  }
  mlir::Type genType(Fortran::common::TypeCategory tc) override final {
    return Fortran::lower::getFIRType(&mlirContext, defaults, tc);
  }

  mlir::Location getCurrentLocation() override final { return toLocation(); }

  mlir::Location genLocation() override final {
    return mlir::UnknownLoc::get(&mlirContext);
  }

  mlir::Location
  genLocation(const Fortran::parser::CharBlock &block) override final {
    if (cooked) {
      auto loc = cooked->GetSourcePositionRange(block);
      if (loc.has_value()) {
        // loc is a pair (begin, end); use the beginning position
        auto &filePos = loc->first;
        return mlir::FileLineColLoc::get(filePos.file.path(), filePos.line,
                                         filePos.column, &mlirContext);
      }
    }
    return genLocation();
  }

  mlir::OpBuilder &getOpBuilder() override final { return *builder; }

  mlir::ModuleOp &getModuleOp() override final { return module; }

  std::string mangleName(Fortran::lower::SymbolRef symbol) override final {
    return Fortran::lower::mangle::mangleName(uniquer, symbol);
  }
};

} // namespace

void Fortran::lower::LoweringBridge::lower(const Fortran::parser::Program &prg,
                                           fir::NameUniquer &uniquer) {
  auto pft = Fortran::lower::createPFT(prg);
  Fortran::lower::annotateControl(*pft);
  if (dumpBeforeFir)
    Fortran::lower::dumpPFT(llvm::errs(), *pft);
  FirConverter converter{*this, uniquer};
  converter.run(*pft);
}

void Fortran::lower::LoweringBridge::parseSourceFile(llvm::SourceMgr &srcMgr) {
  auto owningRef = mlir::parseSourceFile(srcMgr, context.get());
  module.reset(new mlir::ModuleOp(owningRef.get().getOperation()));
  owningRef.release();
}

Fortran::lower::LoweringBridge::LoweringBridge(
    const Fortran::common::IntrinsicTypeDefaultKinds &defaultKinds,
    const Fortran::parser::CookedSource *cooked)
    : defaultKinds{defaultKinds}, cooked{cooked} {
  context = std::make_unique<mlir::MLIRContext>();
  module = std::make_unique<mlir::ModuleOp>(
      mlir::ModuleOp::create(mlir::UnknownLoc::get(context.get())));
}
