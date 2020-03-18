//===-- Bridge.cc -- bridge to lower to MLIR ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/Bridge.h"
#include "flang/Lower/ConvertExpr.h"
#include "flang/Lower/ConvertType.h"
#include "flang/Lower/FIRBuilder.h"
#include "flang/Lower/IO.h"
#include "flang/Lower/Intrinsics.h"
#include "flang/Lower/Mangler.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Lower/Runtime.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/tools.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/Parser.h"
#include "mlir/Target/LLVMIR.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"

#undef TODO
#define TODO()                                                                 \
  {                                                                            \
    if (disableToDoAssertions)                                                 \
      mlir::emitError(toLocation(), __FILE__)                                  \
          << ":" << __LINE__ << " not implemented";                            \
    else                                                                       \
      llvm_unreachable("not yet implemented");                                 \
  }

static llvm::cl::opt<bool>
    dumpBeforeFir("fdebug-dump-pre-fir", llvm::cl::init(false),
                  llvm::cl::desc("dump the IR tree prior to FIR"));

static llvm::cl::opt<bool>
    disableToDoAssertions("disable-burnside-todo",
                          llvm::cl::desc("disable burnside bridge asserts"),
                          llvm::cl::init(false), llvm::cl::Hidden);

namespace {
/// Information for generating a structured or unstructured increment loop.
struct IncrementLoopInfo {
  explicit IncrementLoopInfo(
      Fortran::semantics::Symbol *sym,
      const Fortran::parser::ScalarExpr &lowerExpr,
      const Fortran::parser::ScalarExpr &upperExpr,
      const std::optional<Fortran::parser::ScalarExpr> &stepExpr,
      mlir::Type type)
      : loopVariableSym{sym}, lowerExpr{lowerExpr}, upperExpr{upperExpr},
        stepExpr{stepExpr}, loopVariableType{type} {}

  bool isStructured() const { return headerBlock == nullptr; }

  // Data members for both structured and unstructured loops.
  Fortran::semantics::Symbol *loopVariableSym;
  const Fortran::parser::ScalarExpr &lowerExpr;
  const Fortran::parser::ScalarExpr &upperExpr;
  const std::optional<Fortran::parser::ScalarExpr> &stepExpr;
  mlir::Type loopVariableType;
  mlir::Value loopVariable{};
  mlir::Value stepValue{}; // possible uses in multiple blocks

  // Data members for structured loops.
  fir::LoopOp doLoop{};
  mlir::OpBuilder::InsertPoint insertionPoint{};

  // Data members for unstructured loops.
  mlir::Value tripVariable{};
  mlir::Block *headerBlock{nullptr};    // loop entry and test block
  mlir::Block *bodyBlock{nullptr};      // first loop body block
  mlir::Block *successorBlock{nullptr}; // loop exit target block
};
} // namespace

//===----------------------------------------------------------------------===//
// FirConverter
//===----------------------------------------------------------------------===//

namespace {
/// Walk over the pre-FIR tree (PFT) and lower it to the FIR dialect of MLIR.
///
/// After building the PFT, the FirConverter processes that representation
/// and lowers it to the FIR executable representation.
class FirConverter : public Fortran::lower::AbstractConverter {
public:
  explicit FirConverter(Fortran::lower::LoweringBridge &bridge,
                        fir::NameUniquer &uniquer)
      : mlirContext{bridge.getMLIRContext()}, cooked{bridge.getCookedSource()},
        module{bridge.getModule()}, defaults{bridge.getDefaultKinds()},
        intrinsics{Fortran::lower::IntrinsicLibrary(
            Fortran::lower::IntrinsicLibrary::Version::LLVM,
            bridge.getMLIRContext())},
        uniquer{uniquer} {}
  virtual ~FirConverter() = default;

  /// Convert the PFT to FIR
  void run(Fortran::lower::pft::Program &pft) {
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

  //===--------------------------------------------------------------------===//
  // AbstractConverter overrides
  //===--------------------------------------------------------------------===//

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
    return builder->getUnknownLoc();
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

  Fortran::lower::FirOpBuilder &getFirOpBuilder() override final {
    return *builder;
  }
  mlir::ModuleOp &getModuleOp() override final { return module; }

  std::string mangleName(Fortran::lower::SymbolRef symbol) override final {
    return Fortran::lower::mangle::mangleName(uniquer, symbol);
  }

private:
  FirConverter() = delete;
  FirConverter(const FirConverter &) = delete;
  FirConverter &operator=(const FirConverter &) = delete;

  //===--------------------------------------------------------------------===//
  // Helper member functions
  //===--------------------------------------------------------------------===//

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
    return builder->createTemporary(loc, localSymbols, genType(sym), llvm::None,
                                    &sym);
  }

  mlir::FuncOp genFunctionFIR(llvm::StringRef callee,
                              mlir::FunctionType funcTy) {
    if (auto func = builder->getNamedFunction(callee))
      return func;
    return builder->createFunction(callee, funcTy);
  }

  bool isNumericScalarCategory(Fortran::common::TypeCategory cat) {
    return cat == Fortran::lower::IntegerCat ||
           cat == Fortran::lower::RealCat ||
           cat == Fortran::lower::ComplexCat ||
           cat == Fortran::lower::LogicalCat;
  }

  bool isCharacterCategory(Fortran::common::TypeCategory cat) {
    return cat == Fortran::lower::CharacterCat;
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
                      cstr->parentVariant.p);
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
        cstr->parentVariant.p);
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
        cstr->parentVariant.p);
  }
  
  static const Fortran::parser::MpSubprogramStmt *
  inMpSubprogram(Fortran::lower::pft::Evaluation *cstr) {
    return std::visit(
        Fortran::common::visitors{
            [](Fortran::lower::pft::FunctionLikeUnit *c) {
              return c->getMpSubprogram();
            },
            [&](Fortran::lower::pft::Evaluation *c) {
              return inMpSubprogram(c);
            },
            [](auto *) -> const Fortran::parser::MpSubprogramStmt * {
              return nullptr;
            },
        },
        cstr->parentVariant.p);
  }

  void genFIRUnconditionalBranch(mlir::Block *targetBlock) {
    assert(targetBlock && "missing unconditional target block");
    builder->create<mlir::BranchOp>(toLocation(), targetBlock);
  }

  void
  genFIRUnconditionalBranch(Fortran::lower::pft::Evaluation *targetEvaluation) {
    builder->create<mlir::BranchOp>(toLocation(), targetEvaluation->block);
  }

  void genFIRConditionalBranch(mlir::Value &cond, mlir::Block *trueTarget,
                               mlir::Block *falseTarget) {
    builder->create<mlir::CondBranchOp>(toLocation(), cond, trueTarget,
                                        llvm::None, falseTarget, llvm::None);
  }

  void genFIRConditionalBranch(const Fortran::parser::ScalarLogicalExpr &expr,
                               Fortran::lower::pft::Evaluation *trueTarget,
                               Fortran::lower::pft::Evaluation *falseTarget) {
    assert(trueTarget && "missing conditional branch true block");
    assert(falseTarget && "missing conditional branch true block");
    mlir::Value cond =
        createLogicalExprAsI1(toLocation(), Fortran::semantics::GetExpr(expr));
    genFIRConditionalBranch(cond, trueTarget->block, falseTarget->block);
  }

  //===--------------------------------------------------------------------===//
  // Function-like PFT entry and exit statements
  //===--------------------------------------------------------------------===//

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
    genFIRProgramExit();
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
    genFIRProcedureExit(
        static_cast<const Fortran::parser::SubroutineStmt *>(nullptr));
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
    genFIRProcedureExit(
        static_cast<const Fortran::parser::MpSubprogramStmt *>(nullptr));
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

  mlir::Value genFIRLoopIndex(const Fortran::parser::ScalarExpr &x,
                              mlir::Type t) {
    mlir::Value v = genExprValue(*Fortran::semantics::GetExpr(x));
    return builder->create<fir::ConvertOp>(toLocation(), t, v);
  }

  mlir::Value genFIRLoopIndex(const Fortran::parser::ScalarExpr &x) {
    return genFIRLoopIndex(x, mlir::IndexType::get(&mlirContext));
  }

  mlir::FuncOp getFunc(llvm::StringRef name, mlir::FunctionType ty) {
    if (auto func = builder->getNamedFunction(name)) {
      assert(func.getType() == ty);
      return func;
    }
    return builder->createFunction(name, ty);
  }

  /// Lowering of CALL statement
  ///
  /// 1. Determine what function is being called/dispatched to
  /// 2. Build a tuple of arguments to be passed to that function
  /// 3. Emit fir.call/fir.dispatch on arguments
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::CallStmt &stmt) {
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
    [[maybe_unused]] mlir::FuncOp func = getFunc(funName, funTy);
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

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::IfStmt &stmt) {
    if (eval.lowerAsUnstructured()) {
      genFIRConditionalBranch(
          std::get<Fortran::parser::ScalarLogicalExpr>(stmt.t),
          eval.lexicalSuccessor, eval.controlSuccessor);
      return;
    }

    // Generate fir.where.
    mlir::Block *ifStmtBlock = builder->getInsertionBlock();
    fir::WhereOp where;
    genWhereCondition(where, &stmt);
    genFIR(*eval.lexicalSuccessor, /*unstructuredContext*/ false);
    eval.lexicalSuccessor->skip = true;
    builder->setInsertionPointToEnd(ifStmtBlock);
  }

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::WaitStmt &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::WhereStmt &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::ComputedGotoStmt &stmt) {
    auto *exp = Fortran::semantics::GetExpr(
        std::get<Fortran::parser::ScalarIntExpr>(stmt.t));
    auto e1{genExprValue(*exp)};
    (void)e1;
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::ForallStmt &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::ArithmeticIfStmt &stmt) {
    auto *exp =
        Fortran::semantics::GetExpr(std::get<Fortran::parser::Expr>(stmt.t));
    auto e1{genExprValue(*exp)};
    (void)e1;
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::AssignedGotoStmt &) {
    TODO();
  }

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::AssociateConstruct &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::BlockConstruct &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::ChangeTeamConstruct &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::CriticalConstruct &) {
    TODO();
  }

  /// Generate FIR for a DO construct.  There are six variants:
  ///  - unstructured infinite and while loops
  ///  - structured and unstructured increment loops
  ///  - structured and unstructured concurrent loops
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::DoConstruct &) {
    bool unstructuredContext{eval.lowerAsUnstructured()};
    Fortran::lower::pft::Evaluation &doStmtEval = eval.evaluationList->front();
    auto *doStmt = doStmtEval.getIf<Fortran::parser::NonLabelDoStmt>();
    assert(doStmt && "missing DO statement");
    const auto &loopControl =
        std::get<std::optional<Fortran::parser::LoopControl>>(doStmt->t);
    llvm::SmallVector<IncrementLoopInfo, 1> incrementLoopInfo;
    const Fortran::parser::ScalarLogicalExpr *whileCondition = nullptr;
    bool infiniteLoop = !loopControl.has_value();
    if (infiniteLoop) {
      assert(unstructuredContext && "infinite loop must be unstructured");
      startBlock(doStmtEval.localBlocks[0]); // header block
    } else if ((whileCondition =
                    std::get_if<Fortran::parser::ScalarLogicalExpr>(
                        &loopControl->u))) {
      assert(unstructuredContext && "while loop must be unstructured");
      startBlock(doStmtEval.localBlocks[0]); // header block
      genFIRConditionalBranch(*whileCondition, doStmtEval.lexicalSuccessor,
                              doStmtEval.parentConstruct->constructExit);
    } else if (const auto *bounds =
                   std::get_if<Fortran::parser::LoopControl::Bounds>(
                       &loopControl->u)) {
      // "Normal" increment loop.
      incrementLoopInfo.emplace_back(bounds->name.thing.symbol, bounds->lower,
                                     bounds->upper, bounds->step,
                                     genType(*bounds->name.thing.symbol));
      if (unstructuredContext) {
        maybeStartBlock(doStmtEval.block); // preheader block
        incrementLoopInfo[0].headerBlock = doStmtEval.localBlocks[0];
        incrementLoopInfo[0].bodyBlock = doStmtEval.lexicalSuccessor->block;
        incrementLoopInfo[0].successorBlock =
            doStmtEval.parentConstruct->constructExit->block;
      }
    } else {
      const auto *concurrentInfo =
          std::get_if<Fortran::parser::LoopControl::Concurrent>(
              &loopControl->u);
      assert(concurrentInfo && "DO loop variant is invalid");
      TODO();
      // Add entries to incrementLoopInfo.  (Define extra members for a mask.)
    }
    auto n = incrementLoopInfo.size();
    for (decltype(n) i = 0; i < n; ++i) {
      genFIRIncrementLoopBegin(incrementLoopInfo[i]);
    }

    // Generate loop body code.
    for (auto &e : *eval.evaluationList) {
      genFIR(e, unstructuredContext);
    }

    // Generate end loop code.
    if (infiniteLoop || whileCondition) {
      genFIRUnconditionalBranch(doStmtEval.localBlocks[0]);
    } else {
      for (auto i = incrementLoopInfo.size(); i > 0;)
        genFIRIncrementLoopEnd(incrementLoopInfo[--i]);
    }
  }

  /// Generate FIR to begin a structured or unstructured increment loop.
  void genFIRIncrementLoopBegin(IncrementLoopInfo &info) {
    auto location = toLocation();
    mlir::Type type = info.isStructured()
                          ? mlir::IndexType::get(builder->getContext())
                          : info.loopVariableType;
    auto lowerValue = genFIRLoopIndex(info.lowerExpr, type);
    auto upperValue = genFIRLoopIndex(info.upperExpr, type);
    info.stepValue =
        info.stepExpr.has_value()
            ? genFIRLoopIndex(*info.stepExpr, type)
            : (info.isStructured()
                   ? builder->create<mlir::ConstantIndexOp>(location, 1)
                   : builder->createIntegerConstant(info.loopVariableType, 1));
    assert(info.stepValue && "step value must be set");
    info.loopVariable = createTemporary(location, *info.loopVariableSym);

    // Structured loop - generate fir.loop.
    if (info.isStructured()) {
      info.insertionPoint = builder->saveInsertionPoint();
      info.doLoop = builder->create<fir::LoopOp>(location, lowerValue,
                                                 upperValue, info.stepValue);
      builder->setInsertionPointToStart(info.doLoop.getBody());
      // Always store iteration ssa-value to the LCV to avoid missing any
      // aliasing of the LCV.
      auto lcv = builder->create<fir::ConvertOp>(
          location, info.loopVariableType, info.doLoop.getInductionVar());
      builder->create<fir::StoreOp>(location, lcv, info.loopVariable);
      return;
    }

    // Unstructured loop preheader code - initialize tripVariable, loopVariable.
    auto distance =
        builder->create<mlir::SubIOp>(location, upperValue, lowerValue);
    auto adjusted =
        builder->create<mlir::AddIOp>(location, distance, info.stepValue);
    auto tripCount =
        builder->create<mlir::SignedDivIOp>(location, adjusted, info.stepValue);
    info.tripVariable =
        builder->createTemporary(location, localSymbols, info.loopVariableType);
    builder->create<fir::StoreOp>(location, tripCount, info.tripVariable);
    builder->create<fir::StoreOp>(location, lowerValue, info.loopVariable);

    // Unstructured loop header code - generate loop condition.
    startBlock(info.headerBlock);
    mlir::Value tripVariable =
        builder->create<fir::LoadOp>(location, info.tripVariable);
    mlir::Value zero = builder->createIntegerConstant(info.loopVariableType, 0);
    mlir::Value cond = builder->create<mlir::CmpIOp>(
        location, mlir::CmpIPredicate::sgt, tripVariable, zero);
    genFIRConditionalBranch(cond, info.bodyBlock, info.successorBlock);
  }

  /// Generate FIR to end a structured or unstructured increment loop.
  void genFIRIncrementLoopEnd(IncrementLoopInfo &info) {
    mlir::Location location = toLocation();
    if (info.isStructured()) {
      // End fir.loop.
      builder->restoreInsertionPoint(info.insertionPoint);
      return;
    }

    // Unstructured loop - increment loopVariable.
    mlir::Value loopVariable =
        builder->create<fir::LoadOp>(location, info.loopVariable);
    loopVariable =
        builder->create<mlir::AddIOp>(location, loopVariable, info.stepValue);
    builder->create<fir::StoreOp>(location, loopVariable, info.loopVariable);

    // Unstructured loop - decrement tripVariable.
    mlir::Value tripVariable =
        builder->create<fir::LoadOp>(location, info.tripVariable);
    mlir::Value one = builder->create<mlir::ConstantOp>(
        location, builder->getIntegerAttr(info.loopVariableType, 1));
    tripVariable = builder->create<mlir::SubIOp>(location, tripVariable, one);
    builder->create<fir::StoreOp>(location, tripVariable, info.tripVariable);
    genFIRUnconditionalBranch(info.headerBlock);
  }

  /// Generate structured or unstructured FIR for an IF construct.
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::IfConstruct &) {
    if (eval.lowerAsStructured()) {
      // Structured fir.where nest.
      fir::WhereOp where;
      mlir::OpBuilder::InsertPoint insertionPoint =
          builder->saveInsertionPoint();
      for (auto &e : *eval.evaluationList) {
        if (auto *s = e.getIf<Fortran::parser::IfThenStmt>()) {
          // fir.where op
          genWhereCondition(where, s);
        } else if (auto *s = e.getIf<Fortran::parser::ElseIfStmt>()) {
          // otherwise block, then nested fir.where
          switchInsertionPointToOtherwise(where);
          genWhereCondition(where, s);
        } else if (e.isA<Fortran::parser::ElseStmt>()) {
          // otherwise block
          switchInsertionPointToOtherwise(where);
        } else {
          genFIR(e, /*unstructuredContext*/ false);
        }
      }
      builder->restoreInsertionPoint(insertionPoint);
      return;
    }

    // Unstructured branch sequence.
    for (auto &e : *eval.evaluationList) {
      const Fortran::parser::ScalarLogicalExpr *cond = nullptr;
      if (auto *s = e.getIf<Fortran::parser::IfThenStmt>()) {
        maybeStartBlock(e.block);
        cond = &std::get<Fortran::parser::ScalarLogicalExpr>(s->t);
      } else if (auto *s = e.getIf<Fortran::parser::ElseIfStmt>()) {
        startBlock(e.block);
        cond = &std::get<Fortran::parser::ScalarLogicalExpr>(s->t);
      }
      if (cond) {
        genFIRConditionalBranch(
            *cond,
            e.lexicalSuccessor == e.controlSuccessor
                ? e.parentConstruct->constructExit // empty block --> exit
                : e.lexicalSuccessor,              // nonempty block
            e.controlSuccessor);
      } else {
        genFIR(e);
      }
    }
  }

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::CaseConstruct &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::SelectRankConstruct &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::SelectTypeConstruct &) {
    TODO();
  }

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::WhereConstruct &) {
    TODO();
  }

  /// Lower FORALL construct (See 10.2.4)
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::ForallConstruct &forall) {
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
                genFIR(eval, b.statement);
              },
              [&](const Fortran::parser::Statement<Fortran::parser::WhereStmt>
                      &b) {
                setCurrentPosition(b.source);
                genFIR(eval, b.statement);
              },
              [&](const Fortran::parser::WhereConstruct &b) {
                genFIR(eval, b);
              },
              [&](const Fortran::common::Indirection<
                  Fortran::parser::ForallConstruct> &b) {
                genFIR(eval, b.value());
              },
              [&](const Fortran::parser::Statement<Fortran::parser::ForallStmt>
                      &b) {
                setCurrentPosition(b.source);
                genFIR(eval, b.statement);
              },
          },
          s.u);
    }
    TODO();
  }

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::ForallAssignmentStmt &s) {
    std::visit([&](auto &b) { genFIR(eval, b); }, s.u);
  }

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::CompilerDirective &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::OpenMPConstruct &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::OmpEndLoopDirective &) {
    TODO();
  }

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::AssociateStmt &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::EndAssociateStmt &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::BlockStmt &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::EndBlockStmt &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::SelectCaseStmt &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::CaseStmt &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::EndSelectStmt &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::ChangeTeamStmt &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::EndChangeTeamStmt &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::CriticalStmt &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::EndCriticalStmt &) {
    TODO();
  }

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::NonLabelDoStmt &) {} // nop
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::EndDoStmt &) {} // nop

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::IfThenStmt &) {} // nop
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::ElseIfStmt &) {} // nop
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::ElseStmt &) {} // nop
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::EndIfStmt &) {} // nop

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::SelectRankStmt &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::SelectRankCaseStmt &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::SelectTypeStmt &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::TypeGuardStmt &) {
    TODO();
  }

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::WhereConstructStmt &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::MaskedElsewhereStmt &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::ElsewhereStmt &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::EndWhereStmt &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::ForallConstructStmt &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::EndForallStmt &) {
    TODO();
  }

  //
  // Statements that do not have control-flow semantics
  //

  // IO statements (see io.h)

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::BackspaceStmt &stmt) {
    genBackspaceStatement(*this, stmt);
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::CloseStmt &stmt) {
    genCloseStatement(*this, stmt);
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::EndfileStmt &stmt) {
    genEndfileStatement(*this, stmt);
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::FlushStmt &stmt) {
    genFlushStatement(*this, stmt);
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::InquireStmt &stmt) {
    genInquireStatement(*this, stmt);
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::OpenStmt &stmt) {
    genOpenStatement(*this, stmt);
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::PrintStmt &stmt) {
    genPrintStatement(*this, stmt);
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::ReadStmt &stmt) {
    genReadStatement(*this, stmt);
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::RewindStmt &stmt) {
    genRewindStatement(*this, stmt);
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::WriteStmt &stmt) {
    genWriteStatement(*this, stmt);
  }

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::AllocateStmt &) {
    TODO();
  }

  void genCharacterAssignment(const Fortran::evaluate::Assignment &assignment) {
    // Helper to get address and length from an Expr that is a character
    // variable designator
    auto getAddrAndLength =
        [&](const Fortran::lower::SomeExpr &charDesignatorExpr)
        -> Fortran::lower::CharValue {
      auto addr = genExprAddr(charDesignatorExpr);
      const auto &charExpr =
          std::get<Fortran::evaluate::Expr<Fortran::evaluate::SomeCharacter>>(
              charDesignatorExpr.u);
      auto lenExpr = charExpr.LEN();
      assert(lenExpr.has_value() &&
             "could not get expression to compute character length");
      auto len =
          genExprValue(Fortran::evaluate::AsGenericExpr(std::move(*lenExpr)));
      return {addr, len};
    };

    builder->setLocation(toLocation());

    // RHS evaluation.
    // FIXME:  Only works with rhs that are variable reference.
    // Other expression evaluation are not simple copies.
    auto rhs = getAddrAndLength(assignment.rhs);
    // A temp is needed to evaluate rhs until proven it does not depend on lhs.
    auto tempToEvalRhs =
        builder->createCharacterTemp(builder->getCharacterType(rhs), rhs.len);
    builder->createCopy(tempToEvalRhs, rhs, rhs.len);

    // Copy the minimum of the lhs and rhs lengths and pad the lhs remainder
    auto lhs = getAddrAndLength(assignment.lhs);
    auto cmpLen = builder->createHere<mlir::CmpIOp>(mlir::CmpIPredicate::slt,
                                                    lhs.len, rhs.len);
    auto copyCount =
        builder->createHere<mlir::SelectOp>(cmpLen, lhs.len, rhs.len);
    builder->createCopy(lhs, tempToEvalRhs, copyCount);
    builder->createPadding(lhs, copyCount, lhs.len);
  }

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::AssignmentStmt &stmt) {
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
                auto lhsType = assignment.lhs.GetType();
                assert(lhsType && "lhs cannot be typeless");
                if (isNumericScalarCategory(lhsType->category())) {
                  builder->create<fir::StoreOp>(toLocation(),
                                                genExprValue(assignment.rhs),
                                                genExprValue(assignment.lhs));
                } else if (isCharacterCategory(lhsType->category())) {
                  TODO();
                } else {
                  assert(lhsType->category() == Fortran::lower::DerivedCat);
                  TODO();
                }
              } else if (assignment.lhs.Rank() > 0) {
                // Array assignment
                // See Fortran 2018 10.2.1.3 p5, p6, and p7
                TODO();
              } else {
                // Scalar assignments
                auto lhsType = assignment.lhs.GetType();
                assert(lhsType && "lhs cannot be typeless");
                if (isNumericScalarCategory(lhsType->category())) {
                  // Fortran 2018 10.2.1.3 p8 and p9
                  // Conversions are already inserted by semantic
                  // analysis.
                  builder->create<fir::StoreOp>(toLocation(),
                                                genExprValue(assignment.rhs),
                                                genExprAddr(assignment.lhs));
                } else if (isCharacterCategory(lhsType->category())) {
                  // Fortran 2018 10.2.1.3 p10 and p11
                  genCharacterAssignment(assignment);
                } else {
                  assert(lhsType->category() == Fortran::lower::DerivedCat);
                  // Fortran 2018 10.2.1.3 p12 and p13
                  TODO();
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

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::ContinueStmt &) {} // nop
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::DeallocateStmt &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::EventPostStmt &) {
    // call some runtime routine
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::EventWaitStmt &) {
    // call some runtime routine
    TODO();
  }

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::FormTeamStmt &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::LockStmt &) {
    // call some runtime routine
    TODO();
  }

  /// Nullify pointer object list
  ///
  /// For each pointer object, reset the pointer to a disassociated status.
  /// We do this by setting each pointer to null.
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::NullifyStmt &stmt) {
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
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::PointerAssignmentStmt &) {
    TODO();
  }

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::SyncAllStmt &) {
    // call some runtime routine
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::SyncImagesStmt &) {
    // call some runtime routine
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::SyncMemoryStmt &) {
    // call some runtime routine
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::SyncTeamStmt &) {
    // call some runtime routine
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::UnlockStmt &) {
    // call some runtime routine
    TODO();
  }

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::AssignStmt &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::FormatStmt &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::EntryStmt &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::PauseStmt &) {
    // call some runtime routine
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::DataStmt &) {
    TODO();
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::NamelistStmt &) {
    TODO();
  }

  // call FAIL IMAGE in runtime
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::FailImageStmt &stmt) {
    auto callee = genRuntimeFunction(
        Fortran::lower::RuntimeEntryCode::FailImageStatement, *builder);
    llvm::SmallVector<mlir::Value, 1> operands; // FAIL IMAGE has no args
    builder->create<mlir::CallOp>(toLocation(), callee, operands);
  }

  // call STOP, ERROR STOP in runtime
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::StopStmt &stmt) {
    auto callee = genRuntimeFunction(
        Fortran::lower::RuntimeEntryCode::StopStatement, *builder);
    llvm::SmallVector<mlir::Value, 8> operands;
    builder->create<mlir::CallOp>(toLocation(), callee, operands);
  }

  // gen expression, if any; share code with END of procedure
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::ReturnStmt &) {
    if (inMainProgram(&eval)) {
      builder->create<mlir::ReturnOp>(toLocation());
    } else if (auto *stmt = inSubroutine(&eval)) {
      genFIRProcedureExit(stmt);
    } else if (auto *stmt = inFunction(&eval)) {
      auto *symbol = std::get<Fortran::parser::Name>(stmt->t).symbol;
      assert(symbol);
      genFIRFunctionReturn(*symbol);
    } else if (auto *stmt = inMpSubprogram(&eval)) {
      genFIRProcedureExit(stmt);
    } else {
      mlir::emitError(toLocation(), "unknown subprogram type");
    }
  }

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::CycleStmt &) {
    genFIRUnconditionalBranch(eval.controlSuccessor);
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::ExitStmt &) {
    genFIRUnconditionalBranch(eval.controlSuccessor);
  }
  void genFIR(Fortran::lower::pft::Evaluation &eval,
              const Fortran::parser::GotoStmt &) {
    genFIRUnconditionalBranch(eval.controlSuccessor);
  }

  void genFIR(Fortran::lower::pft::Evaluation &eval,
              bool unstructuredContext = true) {
    if (eval.skip) {
      return; // rhs of {Forall,If,Where}Stmt has already been processed
    }
    setCurrentPosition(eval.position);
    if (unstructuredContext) {
      // When transitioning from unstructured to structured code,
      // the structured code might be a target that starts a new block.
      maybeStartBlock(eval.isConstruct() && eval.lowerAsStructured()
                          ? eval.evaluationList->front().block
                          : eval.block);
    }
    std::visit([&](const auto *p) { genFIR(eval, *p); }, eval.u);
    if (unstructuredContext && eval.lowerAsUnstructured() &&
        eval.controlSuccessor && eval.isActionStmt() && blockIsUnterminated()) {
      // Exit from an unstructured IF or SELECT construct block.
      genFIRUnconditionalBranch(eval.controlSuccessor);
    }
  }

  mlir::FuncOp createNewFunction(llvm::StringRef name,
                                 const Fortran::semantics::Symbol *symbol) {
    mlir::FunctionType ty =
        symbol ? genFunctionType(*symbol)
               : mlir::FunctionType::get(llvm::None, llvm::None, &mlirContext);
    return Fortran::lower::FirOpBuilder::createFunction(toLocation(), module,
                                                        name, ty);
  }

  /// Prepare to translate a new function
  void startNewFunction(Fortran::lower::pft::FunctionLikeUnit &funit,
                        llvm::StringRef name,
                        const Fortran::semantics::Symbol *symbol) {
    assert(!builder && "expected nullptr");
    mlir::FuncOp func =
        Fortran::lower::FirOpBuilder::getNamedFunction(module, name);
    if (!func)
      func = createNewFunction(name, symbol);
    builder = new Fortran::lower::FirOpBuilder(func);
    assert(builder && "FirOpBuilder did not instantiate");
    func.addEntryBlock();
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
      if (details->isFunction())
        createTemporary(toLocation(), details->result());
    }

    // Create most function blocks in advance.
    createEmptyBlocks(funit.evaluationList);

    // Reinstate entry block as the current insertion point.
    builder->setInsertionPointToEnd(&func.front());
  }

  /// Create empty blocks for the current function.
  void createEmptyBlocks(
      std::list<Fortran::lower::pft::Evaluation> &evaluationList) {
    for (auto &eval : evaluationList) {
      if (eval.isNewBlock)
        eval.block = builder->createBlock();
      for (size_t i = 0, n = eval.localBlocks.size(); i < n; ++i)
        eval.localBlocks[i] = builder->createBlock();
      if (eval.isConstruct()) {
        if (eval.lowerAsUnstructured()) {
          createEmptyBlocks(*eval.evaluationList);
        } else {
          // A structured construct that is a target starts a new block.
          Fortran::lower::pft::Evaluation &constructStmt =
              eval.evaluationList->front();
          if (constructStmt.isNewBlock)
            constructStmt.block = builder->createBlock();
        }
      }
    }
  }

  /// Return the predicate: "current block does not have a terminator branch".
  bool blockIsUnterminated() {
    mlir::Block *currentBlock = builder->getBlock();
    return currentBlock->empty() || currentBlock->back().isKnownNonTerminator();
  }

  /// Unconditionally switch code insertion to a new block.
  void startBlock(mlir::Block *newBlock) {
    assert(newBlock && "missing block");
    // If the current block does not have a terminator branch,
    // append a fallthrough branch.
    if (blockIsUnterminated()) {
      genFIRUnconditionalBranch(newBlock);
    }
    builder->setInsertionPointToStart(newBlock);
  }

  /// Conditionally switch code insertion to a new block.
  void maybeStartBlock(mlir::Block *newBlock) {
    if (newBlock) {
      startBlock(newBlock);
    }
  }

  /// Cleanup after the function has been translated
  void endNewFunction() {
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
      std::visit([&](auto *p) { genFIR(*p, name, symbol); }, *func.beginStmt);
    } else {
      name = uniquer.doProgramEntry();
    }

    startNewFunction(func, name, symbol);

    // lower this procedure
    for (auto &eval : func.evaluationList) {
      genFIR(eval);
    }
    std::visit([&](auto *p) { genFIR(*p, name, symbol); }, func.endStmt);

    endNewFunction();

    // recursively lower internal procedures
    llvm::Optional<llvm::StringRef> optName{name};
    for (auto &f : func.nestedFunctions) {
      lowerFunc(f, modules, optName);
    }
  }

  void lowerMod(Fortran::lower::pft::ModuleLikeUnit &mod) {
    // FIXME: build the vector of module names
    std::vector<llvm::StringRef> moduleName;

    // FIXME: do we need to visit the module statements?
    for (auto &f : mod.nestedFunctions) {
      lowerFunc(f, moduleName);
    }
  }

  void setCurrentPosition(const Fortran::parser::CharBlock &position) {
    if (position != Fortran::parser::CharBlock{}) {
      currentPosition = position;
    }
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

  mlir::MLIRContext &mlirContext;
  const Fortran::parser::CookedSource *cooked;
  mlir::ModuleOp &module;
  const Fortran::common::IntrinsicTypeDefaultKinds &defaults;
  Fortran::lower::IntrinsicLibrary intrinsics;
  Fortran::lower::FirOpBuilder *builder = nullptr;
  fir::NameUniquer &uniquer;
  Fortran::lower::SymMap localSymbols;
  Fortran::parser::CharBlock currentPosition;
};

} // namespace

void Fortran::lower::LoweringBridge::lower(const Fortran::parser::Program &prg,
                                           fir::NameUniquer &uniquer) {
  auto pft = Fortran::lower::createPFT(prg);
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
