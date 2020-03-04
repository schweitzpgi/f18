//===-- PFTBuilder.cc -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/PFTBuilder.h"
#include "flang/Parser/dump-parse-tree.h"
#include "flang/Parser/parse-tree-visitor.h"
#include "llvm/Support/CommandLine.h"
#include <algorithm>
#include <cassert>
#include <utility>

namespace Fortran::lower {
namespace {

/// Helpers to unveil parser node inside Fortran::parser::Statement<>,
/// Fortran::parser::UnlabeledStatement, and Fortran::common::Indirection<>
template <typename A>
struct RemoveIndirectionHelper {
  using Type = A;
  static constexpr const Type &unwrap(const A &a) { return a; }
};
template <typename A>
struct RemoveIndirectionHelper<common::Indirection<A>> {
  using Type = A;
  static constexpr const Type &unwrap(const common::Indirection<A> &a) {
    return a.value();
  }
};

template <typename A>
const auto &removeIndirection(const A &a) {
  return RemoveIndirectionHelper<A>::unwrap(a);
}

template <typename A>
struct UnwrapStmt {
  static constexpr bool isStmt{false};
};
template <typename A>
struct UnwrapStmt<parser::Statement<A>> {
  static constexpr bool isStmt{true};
  using Type = typename RemoveIndirectionHelper<A>::Type;
  constexpr UnwrapStmt(const parser::Statement<A> &a)
      : unwrapped{removeIndirection(a.statement)}, position{a.source},
        label{a.label} {}
  const Type &unwrapped;
  parser::CharBlock position;
  std::optional<parser::Label> label;
};
template <typename A>
struct UnwrapStmt<parser::UnlabeledStatement<A>> {
  static constexpr bool isStmt{true};
  using Type = typename RemoveIndirectionHelper<A>::Type;
  constexpr UnwrapStmt(const parser::UnlabeledStatement<A> &a)
      : unwrapped{removeIndirection(a.statement)}, position{a.source} {}
  const Type &unwrapped;
  parser::CharBlock position;
  std::optional<parser::Label> label;
};

/// The instantiation of a parse tree visitor (Pre and Post) is extremely
/// expensive in terms of compile and link time.  So one goal here is to
/// limit the bridge to one such instantiation.
class PFTBuilder {
public:
  PFTBuilder() : pgm{new pft::Program}, parentVariantStack{*pgm.get()} {}

  /// Get the result
  std::unique_ptr<pft::Program> result() { return std::move(pgm); }

  template <typename A>
  constexpr bool Pre(const A &a) {
    if constexpr (pft::isFunctionLike<A>) {
      return enterFunction(a);
    } else if constexpr (pft::isConstruct<A>) {
      return enterConstruct(a);
    } else if constexpr (UnwrapStmt<A>::isStmt) {
      using T = typename UnwrapStmt<A>::Type;
      // Node "a" being visited has one of the following types:
      // Statement<T>, Statement<Indirection<T>, UnlabeledStatement<T>,
      // or UnlabeledStatement<Indirection<T>>
      auto stmt{UnwrapStmt<A>(a)};
      if constexpr (pft::isConstructStmt<T> || pft::isOtherStmt<T>) {
        addEvaluation(pft::Evaluation{stmt.unwrapped, parentVariantStack.back(),
                                      stmt.position, stmt.label});
        return false;
      } else if constexpr (std::is_same_v<T, parser::ActionStmt>) {
        addEvaluation(
            makeEvaluationAction(stmt.unwrapped, stmt.position, stmt.label));
        return true;
      }
    }
    return true;
  }

  template <typename A>
  constexpr void Post(const A &) {
    if constexpr (pft::isFunctionLike<A>) {
      exitFunction();
    } else if constexpr (pft::isConstruct<A>) {
      exitConstruct();
    }
  }

  // Module like
  bool Pre(const parser::Module &node) { return enterModule(node); }
  bool Pre(const parser::Submodule &node) { return enterModule(node); }

  void Post(const parser::Module &) { exitModule(); }
  void Post(const parser::Submodule &) { exitModule(); }

  // Block data
  bool Pre(const parser::BlockData &node) {
    addUnit(pft::BlockDataUnit{node, parentVariantStack.back()});
    return false;
  }

  // Get rid of production wrapper
  bool Pre(const parser::UnlabeledStatement<parser::ForallAssignmentStmt>
               &statement) {
    addEvaluation(std::visit(
        [&](const auto &x) {
          return pft::Evaluation{
              x, parentVariantStack.back(), statement.source, {}};
        },
        statement.statement.u));
    return false;
  }
  bool Pre(const parser::Statement<parser::ForallAssignmentStmt> &statement) {
    addEvaluation(std::visit(
        [&](const auto &x) {
          return pft::Evaluation{x, parentVariantStack.back(), statement.source,
                                 statement.label};
        },
        statement.statement.u));
    return false;
  }
  bool Pre(const parser::WhereBodyConstruct &whereBody) {
    return std::visit(
        common::visitors{
            [&](const parser::Statement<parser::AssignmentStmt> &stmt) {
              // Not caught as other AssignmentStmt because it is not
              // wrapped in a parser::ActionStmt.
              addEvaluation(pft::Evaluation{stmt.statement,
                                            parentVariantStack.back(),
                                            stmt.source, stmt.label});
              return false;
            },
            [&](const auto &) { return true; },
        },
        whereBody.u);
  }

private:
  /// Initialize a new module-like unit and make it the builder's focus.
  template <typename A>
  bool enterModule(const A &func) {
    auto &unit = addUnit(pft::ModuleLikeUnit{func, parentVariantStack.back()});
    functionList = &unit.nestedFunctions;
    parentVariantStack.emplace_back(unit);
    return true;
  }

  void exitModule() {
    parentVariantStack.pop_back();
    ResetFunctionList();
  }

  /// Initialize a new function-like unit and make it the builder's focus.
  template <typename A>
  bool enterFunction(const A &func) {
    auto &unit =
        addFunction(pft::FunctionLikeUnit{func, parentVariantStack.back()});
    labelEvaluationMap = &unit.labelEvaluationMap;
    assignSymbolLabelMap = &unit.assignSymbolLabelMap;
    functionList = &unit.nestedFunctions;
    pushEvaluationList(&unit.evaluationList);
    parentVariantStack.emplace_back(unit);
    return true;
  }

  void exitFunction() {
    // Guarantee that there is a branch target after the last user statement.
    static const parser::ContinueStmt endTarget{};
    addEvaluation(
        pft::Evaluation{endTarget, parentVariantStack.back(), {}, {}});
    lastLexicalEvaluation = nullptr;
    analyzeBranches(nullptr, *evaluationListStack.back()); // add branch links
    popEvaluationList();
    labelEvaluationMap = nullptr;
    assignSymbolLabelMap = nullptr;
    parentVariantStack.pop_back();
    ResetFunctionList();
  }

  /// Initialize a new construct and make it the builder's focus.
  template <typename A>
  bool enterConstruct(const A &construct) {
    auto &eval =
        addEvaluation(pft::Evaluation{construct, parentVariantStack.back()});
    eval.evaluationList.reset(new pft::EvaluationList);
    pushEvaluationList(eval.evaluationList.get());
    parentVariantStack.emplace_back(eval);
    constructStack.emplace_back(&eval);
    return true;
  }

  void exitConstruct() {
    popEvaluationList();
    parentVariantStack.pop_back();
    constructStack.pop_back();
  }

  /// Reset functionList to an enclosing function's functionList.
  void ResetFunctionList() {
    if (!parentVariantStack.empty()) {
      std::visit(common::visitors{
                     [&](pft::FunctionLikeUnit *p) {
                       functionList = &p->nestedFunctions;
                     },
                     [&](pft::ModuleLikeUnit *p) {
                       functionList = &p->nestedFunctions;
                     },
                     [&](auto *) { functionList = nullptr; },
                 },
                 parentVariantStack.back().p);
    }
  }

  template <typename A>
  A &addUnit(A &&unit) {
    pgm->getUnits().emplace_back(std::move(unit));
    return std::get<A>(pgm->getUnits().back());
  }

  template <typename A>
  A &addFunction(A &&func) {
    if (functionList) {
      functionList->emplace_back(std::move(func));
      return functionList->back();
    }
    return addUnit(std::move(func));
  }

  // ActionStmt has a couple of non-conforming cases, explicitly handled here.
  // The other cases use an Indirection, which are discarded in the PFT.
  pft::Evaluation makeEvaluationAction(const parser::ActionStmt &statement,
                                       parser::CharBlock position,
                                       std::optional<parser::Label> label) {
    return std::visit(
        common::visitors{
            [&](const auto &x) {
              return pft::Evaluation{removeIndirection(x),
                                     parentVariantStack.back(), position,
                                     label};
            },
        },
        statement.u);
  }

  /// Append an Evaluation to the end of the current list.
  pft::Evaluation &addEvaluation(pft::Evaluation &&eval) {
    assert(functionList && "not in a function");
    assert(evaluationListStack.size() > 0);
    if (constructStack.size() > 0) {
      eval.parentConstruct = constructStack.back();
    }
    evaluationListStack.back()->emplace_back(std::move(eval));
    pft::Evaluation *p = &evaluationListStack.back()->back();
    if (p->isActionStmt() || p->isConstructStmt()) {
      if (lastLexicalEvaluation) {
        lastLexicalEvaluation->lexicalSuccessor = p;
        p->printIndex = lastLexicalEvaluation->printIndex + 1;
      } else {
        p->printIndex = 1;
      }
      lastLexicalEvaluation = p;
    }
    if (p->label.has_value()) {
      labelEvaluationMap->try_emplace(*p->label, p);
    }
    return evaluationListStack.back()->back();
  }

  /// push a new list on the stack of Evaluation lists
  void pushEvaluationList(pft::EvaluationList *eval) {
    assert(functionList && "not in a function");
    assert(eval && eval->empty() && "evaluation list isn't correct");
    evaluationListStack.emplace_back(eval);
  }

  /// pop the current list and return to the last Evaluation list
  void popEvaluationList() {
    assert(functionList && "not in a function");
    evaluationListStack.pop_back();
  }

  /// Mark I/O statement ERR, EOR, and END specifier branch targets.
  template <typename A>
  constexpr void analyzeIoBranches(pft::Evaluation &eval, const A &stmt) {
    if constexpr (std::is_same_v<A, parser::ReadStmt> ||
                  std::is_same_v<A, parser::WriteStmt>) {
      for (const auto &control : stmt.controls) {
        if (std::holds_alternative<parser::ErrLabel>(control.u) ||
            std::holds_alternative<parser::EorLabel>(control.u) ||
            std::holds_alternative<parser::EndLabel>(control.u)) {
          pft::Evaluation *t{labelEvaluationMap->find(stmt->v)->second};
          markBranchTarget(eval, *t);
        }
      }
    }
    if constexpr (std::is_same_v<A, parser::WaitStmt> ||
                  std::is_same_v<A, parser::OpenStmt> ||
                  std::is_same_v<A, parser::CloseStmt> ||
                  std::is_same_v<A, parser::BackspaceStmt> ||
                  std::is_same_v<A, parser::EndfileStmt> ||
                  std::is_same_v<A, parser::RewindStmt> ||
                  std::is_same_v<A, parser::FlushStmt>) {
      for (const auto &spec : stmt.v) {
        if (std::holds_alternative<parser::ErrLabel>(spec.u) ||
            std::holds_alternative<parser::EorLabel>(spec.u) ||
            std::holds_alternative<parser::EndLabel>(spec.u)) {
          pft::Evaluation *t{labelEvaluationMap->find(stmt->v)->second};
          markBranchTarget(eval, *t);
        }
      }
    }
    if constexpr (std::is_same_v<A, parser::InquireStmt>) {
      for (const auto &spec :
           std::get<std::list<parser::InquireSpec>>(stmt.u)) {
        if (std::holds_alternative<parser::ErrLabel>(spec.u) ||
            std::holds_alternative<parser::EorLabel>(spec.u) ||
            std::holds_alternative<parser::EndLabel>(spec.u)) {
          pft::Evaluation *t{labelEvaluationMap->find(stmt->v)->second};
          markBranchTarget(eval, *t);
        }
      }
    }
  }

  void markBranchTarget(pft::Evaluation &sourceEvaluation,
                        pft::Evaluation &targetEvaluation) {
    targetEvaluation.isNewBlock = true;
    sourceEvaluation.isUnstructured = true;
    if (sourceEvaluation.parentConstruct) {
      sourceEvaluation.parentConstruct->isUnstructured = true;
    }
    if (!sourceEvaluation.controlSuccessor) {
      sourceEvaluation.controlSuccessor = &targetEvaluation;
    }
  }
  void markBranchTarget(pft::Evaluation &sourceEvaluation,
                        parser::Label label) {
    pft::Evaluation *targetEvaluation{labelEvaluationMap->find(label)->second};
    assert(targetEvaluation && "missing branch target");
    markBranchTarget(sourceEvaluation, *targetEvaluation);
  }

  /// Set the exit of a construct, possibly from multiple enclosing constructs.
  void setConstructExit(pft::Evaluation &eval) {
    pft::Evaluation *constructExit{
        eval.evaluationList->back().lexicalSuccessor};
    // Exit from enclosing constructs with nop end statements.
    if (eval.parentConstruct &&
        std::visit(common::visitors{
                       [](const parser::EndAssociateStmt *) { return true; },
                       [](const parser::CaseStmt *) { return true; },
                       [](const parser::EndSelectStmt *) { return true; },
                       [](const parser::EndChangeTeamStmt *) { return true; },
                       [](const parser::EndCriticalStmt *) { return true; },
                       [](const parser::ElseIfStmt *) { return true; },
                       [](const parser::ElseStmt *) { return true; },
                       [](const parser::EndIfStmt *) { return true; },
                       [](const parser::SelectRankCaseStmt *) { return true; },
                       [](const parser::TypeGuardStmt *) { return true; },
                       [](const auto *) { return false; },
                   },
                   constructExit->u)) {
      constructExit = eval.parentConstruct->constructExit;
    }
    assert(constructExit && "missing construct exit");
    eval.constructExit = constructExit;
  }

  /// Return the lexical successor of an Evaluation, accounting for (some) nops.
  pft::Evaluation *effectiveLexicalSuccessor(pft::Evaluation &eval) {
    pft::Evaluation *successor{eval.lexicalSuccessor};
    if (!successor->isNewBlock && eval.parentConstruct &&
        &eval.parentConstruct->evaluationList->back() == successor) {
      successor = eval.parentConstruct->constructExit;
    }
    assert(successor && "missing effective lexical successor");
    return successor;
  }

  /// Mark the effective lexical successor of an Evaluation as a new block.
  void markSuccessorNewBlock(pft::Evaluation &eval) {
    effectiveLexicalSuccessor(eval)->isNewBlock = true;
  }

  template <typename A>
  constexpr std::string getConstructName(const A &stmt) {
    if constexpr (std::is_same_v<A, parser::BlockStmt *> ||
                  std::is_same_v<A, const parser::CycleStmt *> ||
                  std::is_same_v<A, const parser::ElseStmt *> ||
                  std::is_same_v<A, const parser::ElsewhereStmt *> ||
                  std::is_same_v<A, const parser::EndAssociateStmt *> ||
                  std::is_same_v<A, const parser::EndBlockStmt *> ||
                  std::is_same_v<A, const parser::EndCriticalStmt *> ||
                  std::is_same_v<A, const parser::EndDoStmt *> ||
                  std::is_same_v<A, const parser::EndForallStmt *> ||
                  std::is_same_v<A, const parser::EndIfStmt *> ||
                  std::is_same_v<A, const parser::EndSelectStmt *> ||
                  std::is_same_v<A, const parser::EndWhereStmt *> ||
                  std::is_same_v<A, const parser::ExitStmt *>) {
      if (stmt->v) {
        return stmt->v->ToString();
      }
    }
    if constexpr (std::is_same_v<A, const parser::AssociateStmt *> ||
                  std::is_same_v<A, const parser::CaseStmt *> ||
                  std::is_same_v<A, const parser::ChangeTeamStmt *> ||
                  std::is_same_v<A, const parser::CriticalStmt *> ||
                  std::is_same_v<A, const parser::ElseIfStmt *> ||
                  std::is_same_v<A, const parser::EndChangeTeamStmt *> ||
                  std::is_same_v<A, const parser::ForallConstructStmt *> ||
                  std::is_same_v<A, const parser::IfThenStmt *> ||
                  std::is_same_v<A, const parser::LabelDoStmt *> ||
                  std::is_same_v<A, const parser::MaskedElsewhereStmt *> ||
                  std::is_same_v<A, const parser::NonLabelDoStmt *> ||
                  std::is_same_v<A, const parser::SelectCaseStmt *> ||
                  std::is_same_v<A, const parser::SelectRankCaseStmt *> ||
                  std::is_same_v<A, const parser::TypeGuardStmt *> ||
                  std::is_same_v<A, const parser::WhereConstructStmt *>) {
      if (auto name{std::get<std::optional<parser::Name>>(stmt->t)}) {
        return name->ToString();
      }
    }
    if constexpr (std::is_same_v<A, const parser::SelectRankStmt *> ||
                  std::is_same_v<A, const parser::SelectTypeStmt *>) {
      if (auto name{std::get<0>(stmt->t)}) {
        return name->ToString();
      }
    }
    return {};
  }

  template <typename A>
  void insertConstructName(const A &stmt, pft::Evaluation *parentConstruct) {
    std::string name{getConstructName(stmt)};
    if (!name.empty()) {
      constructNameMap[name] = parentConstruct;
    }
  }

  /// Insert branch links for a list of Evaluations.
  void analyzeBranches(pft::Evaluation *parentConstruct,
                       std::list<pft::Evaluation> &evaluationList) {
    pft::Evaluation *lastIfConstructEvaluation{nullptr};
    pft::Evaluation *lastIfStmtEvaluation{nullptr};
    for (auto &eval : evaluationList) {
      std::visit(
          common::visitors{
              // Action statements
              [&](const parser::BackspaceStmt *s) {
                analyzeIoBranches(eval, s);
              },
              [&](const parser::CallStmt *s) {
                // Look for alternate return specifiers.
                const auto &args{
                    std::get<std::list<parser::ActualArgSpec>>(s->v.t)};
                for (const auto &arg : args) {
                  const auto &actual{std::get<parser::ActualArg>(arg.t)};
                  if (const auto *altReturn{
                          std::get_if<parser::AltReturnSpec>(&actual.u)}) {
                    markBranchTarget(eval, altReturn->v);
                  }
                }
              },
              [&](const parser::CloseStmt *s) { analyzeIoBranches(eval, s); },
              [&](const parser::CycleStmt *s) {
                std::string name{getConstructName(s)};
                pft::Evaluation *construct{name.empty()
                                               ? doConstructStack.back()
                                               : constructNameMap[name]};
                assert(construct && "missing CYCLE construct");
                markBranchTarget(eval, construct->evaluationList->back());
              },
              [&](const parser::EndfileStmt *s) { analyzeIoBranches(eval, s); },
              [&](const parser::ExitStmt *s) {
                std::string name{getConstructName(s)};
                pft::Evaluation *construct{name.empty()
                                               ? doConstructStack.back()
                                               : constructNameMap[name]};
                assert(construct && "missing EXIT construct");
                markBranchTarget(eval, *construct->constructExit);
              },
              [&](const parser::FlushStmt *s) { analyzeIoBranches(eval, s); },
              [&](const parser::GotoStmt *s) { markBranchTarget(eval, s->v); },
              [&](const parser::IfStmt *) { lastIfStmtEvaluation = &eval; },
              [&](const parser::InquireStmt *s) { analyzeIoBranches(eval, s); },
              [&](const parser::OpenStmt *s) { analyzeIoBranches(eval, s); },
              [&](const parser::ReadStmt *s) { analyzeIoBranches(eval, s); },
              [&](const parser::ReturnStmt *) { eval.isUnstructured = true; },
              [&](const parser::RewindStmt *s) { analyzeIoBranches(eval, s); },
              [&](const parser::StopStmt *) { eval.isUnstructured = true; },
              [&](const parser::WaitStmt *s) { analyzeIoBranches(eval, s); },
              [&](const parser::WriteStmt *s) { analyzeIoBranches(eval, s); },
              [&](const parser::ComputedGotoStmt *s) {
                for (auto &label : std::get<std::list<parser::Label>>(s->t)) {
                  markBranchTarget(eval, label);
                }
              },
              [&](const parser::ArithmeticIfStmt *s) {
                markBranchTarget(eval, std::get<1>(s->t));
                markBranchTarget(eval, std::get<2>(s->t));
                markBranchTarget(eval, std::get<3>(s->t));
              },
              [&](const parser::AssignStmt *s) { // legacy label assignment
                auto &label = std::get<parser::Label>(s->t);
                auto sym = std::get<parser::Name>(s->t).symbol;
                assert(sym && "missing AssignStmt symbol");
                pft::Evaluation *t{labelEvaluationMap->find(label)->second};
                if (!std::get_if<const parser::FormatStmt *const>(&t->u)) {
                  markBranchTarget(eval, label);
                }
                auto iter = assignSymbolLabelMap->find(sym);
                if (iter == assignSymbolLabelMap->end()) {
                  pft::LabelSet labelSet{};
                  labelSet.insert(label);
                  assignSymbolLabelMap->try_emplace(sym, labelSet);
                } else {
                  iter->second.insert(label);
                }
              },
              [&](const parser::AssignedGotoStmt *) {
                // Specific control successors are not in general known.
                // Compensate by directly marking the successor new block.
                markSuccessorNewBlock(eval);
              },

              // Construct statements
              [&](const parser::AssociateStmt *s) {
                insertConstructName(&*s, parentConstruct);
              },
              [&](const parser::BlockStmt *s) {
                insertConstructName(&*s, parentConstruct);
              },
              [&](const parser::SelectCaseStmt *s) {
                insertConstructName(&*s, parentConstruct);
                eval.lexicalSuccessor->isNewBlock = true;
              },
              [&](const parser::CaseStmt *) { eval.isNewBlock = true; },
              [&](const parser::ChangeTeamStmt *s) {
                insertConstructName(&*s, parentConstruct);
              },
              [&](const parser::CriticalStmt *s) {
                insertConstructName(&*s, parentConstruct);
              },
              [&](const parser::NonLabelDoStmt *s) {
                insertConstructName(&*s, parentConstruct);
                doConstructStack.push_back(parentConstruct);
                auto &control{
                    std::get<std::optional<parser::LoopControl>>(s->t)};
                // eval.block is the loop preheader block, which will be set
                // elsewhere if the NonLabelDoStmt is itself a target.
                // eval.localBlocks[0] is the loop header block.
                eval.localBlocks.emplace_back(nullptr);
                if (!control.has_value()) {
                  eval.isUnstructured = true; // infinite loop
                  return;
                }
                eval.lexicalSuccessor->isNewBlock = true;
                eval.controlSuccessor = &evaluationList.back();
                if (std::holds_alternative<parser::ScalarLogicalExpr>(
                        control->u)) {
                  eval.isUnstructured = true; // while loop
                }
                // Defer additional processing for a concurrent loop to the
                // EndDoStmt, when it is known if the loop is structured or not.
              },
              [&](const parser::EndDoStmt *) {
                pft::Evaluation &doEval{evaluationList.front()};
                eval.controlSuccessor = &doEval;
                doConstructStack.pop_back();
                if (parentConstruct->lowerAsStructured()) {
                  return;
                }
                parentConstruct->constructExit->isNewBlock = true;
                const auto &doStmt{doEval.getIf<parser::NonLabelDoStmt>()};
                assert(doStmt && "missing NonLabelDoStmt");
                auto &control{
                    std::get<std::optional<parser::LoopControl>>(doStmt->t)};
                if (!control.has_value()) {
                  return; // infinite loop
                }
                const auto *concurrent{
                    std::get_if<parser::LoopControl::Concurrent>(&control->u)};
                if (!concurrent) {
                  return;
                }
                // Unstructured concurrent loop.  NonLabelDoStmt code accounts
                // for one concurrent loop dimension.  Reserve preheader,
                // header, and latch blocks for the remaining dimensions, and
                // one block for a mask expression.
                const auto &header{
                    std::get<parser::ConcurrentHeader>(concurrent->t)};
                auto dims{
                    std::get<std::list<parser::ConcurrentControl>>(header.t)
                        .size()};
                for (; dims > 1; --dims) {
                  doEval.localBlocks.emplace_back(nullptr); // preheader
                  doEval.localBlocks.emplace_back(nullptr); // header
                  eval.localBlocks.emplace_back(nullptr);   // latch
                }
                if (std::get<std::optional<parser::ScalarLogicalExpr>>(
                        header.t)) {
                  doEval.localBlocks.emplace_back(nullptr); // mask
                }
              },
              [&](const parser::IfThenStmt *s) {
                insertConstructName(&*s, parentConstruct);
                eval.lexicalSuccessor->isNewBlock = true;
                lastIfConstructEvaluation = &eval;
              },
              [&](const parser::ElseIfStmt *s) {
                eval.isNewBlock = true;
                eval.lexicalSuccessor->isNewBlock = true;
                lastIfConstructEvaluation->controlSuccessor = &eval;
                lastIfConstructEvaluation = &eval;
              },
              [&](const parser::ElseStmt *s) {
                eval.isNewBlock = true;
                lastIfConstructEvaluation->controlSuccessor = &eval;
                lastIfConstructEvaluation = &eval;
              },
              [&](const parser::EndIfStmt *) {
                if (parentConstruct->lowerAsUnstructured()) {
                  parentConstruct->constructExit->isNewBlock = true;
                }
                lastIfConstructEvaluation->controlSuccessor =
                    parentConstruct->constructExit;
                lastIfConstructEvaluation = nullptr;
              },
              [&](const parser::SelectRankStmt *s) {
                insertConstructName(&*s, parentConstruct);
                eval.lexicalSuccessor->isNewBlock = true;
              },
              [&](const parser::SelectRankCaseStmt *) {
                eval.isNewBlock = true;
              },
              [&](const parser::SelectTypeStmt *s) {
                insertConstructName(&*s, parentConstruct);
                eval.lexicalSuccessor->isNewBlock = true;
              },
              [&](const parser::TypeGuardStmt *) { eval.isNewBlock = true; },

              // Constructs - set (unstructured construct) exit targets
              [&](const parser::AssociateConstruct *) {
                setConstructExit(eval);
              },
              [&](const parser::BlockConstruct *) {
                // EndBlockStmt may have exit code.
                eval.constructExit = &eval.evaluationList->back();
              },
              [&](const parser::CaseConstruct *) { setConstructExit(eval); },
              [&](const parser::ChangeTeamConstruct *) {
                setConstructExit(eval);
              },
              [&](const parser::CriticalConstruct *) {
                setConstructExit(eval);
              },
              [&](const parser::DoConstruct *) { setConstructExit(eval); },
              [&](const parser::IfConstruct *) { setConstructExit(eval); },
              [&](const parser::SelectRankConstruct *) {
                setConstructExit(eval);
              },
              [&](const parser::SelectTypeConstruct *) {
                setConstructExit(eval);
              },

              [](const auto *) { /* do nothing */ },
          },
          eval.u);

      // Analyze branches in a nested construct.
      if (eval.evaluationList) {
        analyzeBranches(&eval, *eval.evaluationList);
      }

      // Insert branch links for an unstructured IF statement.
      if (lastIfStmtEvaluation && lastIfStmtEvaluation != &eval) {
        // eval is the action substatement of an IfStmt.
        if (eval.lowerAsUnstructured()) {
          eval.isNewBlock = true;
          markSuccessorNewBlock(eval);
          lastIfStmtEvaluation->isUnstructured = true;
        }
        lastIfStmtEvaluation->controlSuccessor =
            effectiveLexicalSuccessor(eval);
        lastIfStmtEvaluation = nullptr;
      }

      // Set the successor of the last statement in an IF or SELECT block.
      if (!eval.controlSuccessor && eval.lexicalSuccessor &&
          std::visit(
              common::visitors{
                  [](const parser::CaseStmt *) { return true; },
                  [](const parser::ElseIfStmt *) { return true; },
                  [](const parser::ElseStmt *) { return true; },
                  [](const parser::SelectRankCaseStmt *) { return true; },
                  [](const parser::TypeGuardStmt *) { return true; },
                  [](const auto *) { return false; },
              },
              eval.lexicalSuccessor->u)) {
        eval.controlSuccessor = parentConstruct->constructExit;
        eval.lexicalSuccessor->isNewBlock = true;
      }

      // The lexical successor of a branch starts a new block.
      if (eval.controlSuccessor && eval.isActionStmt()) {
        markSuccessorNewBlock(eval);
      }

      // Propagate isUnstructured flag to enclosing construct.
      if (parentConstruct && eval.isUnstructured) {
        parentConstruct->isUnstructured = true;
      }
    }
  }

  std::unique_ptr<pft::Program> pgm;
  /// functionList points to the internal or module procedure function list
  /// of a FunctionLikeUnit or a ModuleLikeUnit.  It may be null.
  std::list<pft::FunctionLikeUnit> *functionList{nullptr};
  std::vector<pft::ParentVariant> parentVariantStack;
  std::vector<pft::Evaluation *> constructStack{};
  std::vector<pft::Evaluation *> doConstructStack{};
  /// evaluationListStack is the current nested construct evaluationList state.
  std::vector<pft::EvaluationList *> evaluationListStack{};
  llvm::DenseMap<parser::Label, pft::Evaluation *> *labelEvaluationMap{nullptr};
  pft::SymbolLabelMap *assignSymbolLabelMap{nullptr};
  std::map<std::string, pft::Evaluation *> constructNameMap{};
  pft::Evaluation *lastLexicalEvaluation{nullptr};
};

template <typename Label, typename A>
constexpr bool hasLabel(const A &stmt) {
  auto isLabel{
      [](const auto &v) { return std::holds_alternative<Label>(v.u); }};
  if constexpr (std::is_same_v<A, parser::ReadStmt> ||
                std::is_same_v<A, parser::WriteStmt>) {
    return std::any_of(std::begin(stmt.controls), std::end(stmt.controls),
                       isLabel);
  }
  if constexpr (std::is_same_v<A, parser::WaitStmt>) {
    return std::any_of(std::begin(stmt.v), std::end(stmt.v), isLabel);
  }
  if constexpr (std::is_same_v<Label, parser::ErrLabel>) {
    if constexpr (common::HasMember<
                      A, std::tuple<parser::OpenStmt, parser::CloseStmt,
                                    parser::BackspaceStmt, parser::EndfileStmt,
                                    parser::RewindStmt, parser::FlushStmt>>)
      return std::any_of(std::begin(stmt.v), std::end(stmt.v), isLabel);
    if constexpr (std::is_same_v<A, parser::InquireStmt>) {
      const auto &specifiers{std::get<std::list<parser::InquireSpec>>(stmt.u)};
      return std::any_of(std::begin(specifiers), std::end(specifiers), isLabel);
    }
  }
  return false;
}

class PFTDumper {
public:
  void dumpPFT(llvm::raw_ostream &outputStream, pft::Program &pft) {
    for (auto &unit : pft.getUnits()) {
      std::visit(common::visitors{
                     [&](pft::BlockDataUnit &unit) {
                       outputStream << getNodeIndex(unit) << " ";
                       outputStream << "BlockData: ";
                       outputStream << "\nEndBlockData\n\n";
                     },
                     [&](pft::FunctionLikeUnit &func) {
                       dumpFunctionLikeUnit(outputStream, func);
                     },
                     [&](pft::ModuleLikeUnit &unit) {
                       dumpModuleLikeUnit(outputStream, unit);
                     },
                 },
                 unit);
    }
  }

  llvm::StringRef evaluationName(pft::Evaluation &eval) {
    return eval.visit(common::visitors{
        [](const auto &parseTreeNode) {
          return parser::ParseTreeDumper::GetNodeName(parseTreeNode);
        },
    });
  }

  void dumpEvaluationList(llvm::raw_ostream &outputStream,
                          pft::EvaluationList &evaluationList, int indent = 1) {
    static const std::string white{"                                      ++"};
    std::string indentString{white.substr(0, indent * 2)};
    for (pft::Evaluation &eval : evaluationList) {
      llvm::StringRef name{evaluationName(eval)};
      std::string bang{eval.isUnstructured ? "!" : ""};
      if (eval.isConstruct()) {
        outputStream << indentString << "<<" << name << bang << ">>";
        if (eval.constructExit) {
          outputStream << " -> " << eval.constructExit->printIndex;
        }
        outputStream << '\n';
        dumpEvaluationList(outputStream, *eval.evaluationList, indent + 1);
        outputStream << indentString << "<<End " << name << bang << ">>\n";
        continue;
      }
      outputStream << indentString;
      if (eval.printIndex) {
        outputStream << eval.printIndex << ' ';
      }
      if (eval.isNewBlock) {
        outputStream << '^';
      }
      if (eval.localBlocks.size()) {
        outputStream << '*';
      }
      outputStream << name << bang;
      if (eval.isActionStmt() || eval.isConstructStmt()) {
        if (eval.controlSuccessor) {
          outputStream << " -> " << eval.controlSuccessor->printIndex;
        }
      }
      if (eval.position.size()) {
        outputStream << ": " << eval.position.ToString();
      }
      outputStream << '\n';
    }
  }

  void dumpFunctionLikeUnit(llvm::raw_ostream &outputStream,
                            pft::FunctionLikeUnit &functionLikeUnit) {
    outputStream << getNodeIndex(functionLikeUnit) << " ";
    llvm::StringRef unitKind{};
    std::string name{};
    std::string header{};
    if (functionLikeUnit.beginStmt) {
      std::visit(
          common::visitors{
              [&](const parser::Statement<parser::ProgramStmt> *statement) {
                unitKind = "Program";
                name = statement->statement.v.ToString();
              },
              [&](const parser::Statement<parser::FunctionStmt> *statement) {
                unitKind = "Function";
                name =
                    std::get<parser::Name>(statement->statement.t).ToString();
                header = statement->source.ToString();
              },
              [&](const parser::Statement<parser::SubroutineStmt> *statement) {
                unitKind = "Subroutine";
                name =
                    std::get<parser::Name>(statement->statement.t).ToString();
                header = statement->source.ToString();
              },
              [&](const parser::Statement<parser::MpSubprogramStmt>
                      *statement) {
                unitKind = "MpSubprogram";
                name = statement->statement.v.ToString();
                header = statement->source.ToString();
              },
              [&](auto *) {},
          },
          *functionLikeUnit.beginStmt);
    } else {
      unitKind = "Program";
      name = "<anonymous>";
    }
    outputStream << unitKind << ' ' << name;
    if (header.size())
      outputStream << ": " << header;
    outputStream << '\n';
    dumpEvaluationList(outputStream, functionLikeUnit.evaluationList);
    if (!functionLikeUnit.nestedFunctions.empty()) {
      outputStream << "\nContains\n";
      for (auto &func : functionLikeUnit.nestedFunctions)
        dumpFunctionLikeUnit(outputStream, func);
      outputStream << "EndContains\n";
    }
    outputStream << "End" << unitKind << ' ' << name << "\n\n";
  }

  void dumpModuleLikeUnit(llvm::raw_ostream &outputStream,
                          pft::ModuleLikeUnit &moduleLikeUnit) {
    outputStream << getNodeIndex(moduleLikeUnit) << " ";
    outputStream << "ModuleLike: ";
    outputStream << "\nContains\n";
    for (auto &func : moduleLikeUnit.nestedFunctions)
      dumpFunctionLikeUnit(outputStream, func);
    outputStream << "EndContains\nEndModuleLike\n\n";
  }

  template <typename T>
  std::size_t getNodeIndex(const T &node) {
    auto addr{static_cast<const void *>(&node)};
    auto it{nodeIndexes.find(addr)};
    if (it != nodeIndexes.end()) {
      return it->second;
    }
    nodeIndexes.try_emplace(addr, nextIndex);
    return nextIndex++;
  }
  std::size_t getNodeIndex(const pft::Program &) { return 0; }

private:
  llvm::DenseMap<const void *, std::size_t> nodeIndexes;
  std::size_t nextIndex{1}; // 0 is the root
};

template <typename A, typename T>
pft::FunctionLikeUnit::FunctionStatement getFunctionStmt(const T &func) {
  return pft::FunctionLikeUnit::FunctionStatement{
      &std::get<parser::Statement<A>>(func.t)};
}
template <typename A, typename T>
pft::ModuleLikeUnit::ModuleStatement getModuleStmt(const T &mod) {
  return pft::ModuleLikeUnit::ModuleStatement{
      &std::get<parser::Statement<A>>(mod.t)};
}

} // namespace

llvm::cl::opt<bool> clDisableStructuredFir(
    "no-structured-fir", llvm::cl::desc("disable generation of structured FIR"),
    llvm::cl::init(false), llvm::cl::Hidden);

bool pft::Evaluation::lowerAsStructured() const {
  return !lowerAsUnstructured();
}

bool pft::Evaluation::lowerAsUnstructured() const {
  return isUnstructured || clDisableStructuredFir;
}

pft::FunctionLikeUnit::FunctionLikeUnit(const parser::MainProgram &func,
                                        const pft::ParentVariant &parent)
    : ProgramUnit{func, parent} {
  auto &ps{
      std::get<std::optional<parser::Statement<parser::ProgramStmt>>>(func.t)};
  if (ps.has_value()) {
    const parser::Statement<parser::ProgramStmt> &statement{ps.value()};
    beginStmt = &statement;
  }
  endStmt = getFunctionStmt<parser::EndProgramStmt>(func);
}

pft::FunctionLikeUnit::FunctionLikeUnit(const parser::FunctionSubprogram &func,
                                        const pft::ParentVariant &parent)
    : ProgramUnit{func, parent},
      beginStmt{getFunctionStmt<parser::FunctionStmt>(func)},
      endStmt{getFunctionStmt<parser::EndFunctionStmt>(func)} {}

pft::FunctionLikeUnit::FunctionLikeUnit(
    const parser::SubroutineSubprogram &func, const pft::ParentVariant &parent)
    : ProgramUnit{func, parent},
      beginStmt{getFunctionStmt<parser::SubroutineStmt>(func)},
      endStmt{getFunctionStmt<parser::EndSubroutineStmt>(func)} {}

pft::FunctionLikeUnit::FunctionLikeUnit(
    const parser::SeparateModuleSubprogram &func,
    const pft::ParentVariant &parent)
    : ProgramUnit{func, parent},
      beginStmt{getFunctionStmt<parser::MpSubprogramStmt>(func)},
      endStmt{getFunctionStmt<parser::EndMpSubprogramStmt>(func)} {}

pft::ModuleLikeUnit::ModuleLikeUnit(const parser::Module &m,
                                    const pft::ParentVariant &parent)
    : ProgramUnit{m, parent}, beginStmt{getModuleStmt<parser::ModuleStmt>(m)},
      endStmt{getModuleStmt<parser::EndModuleStmt>(m)} {}

pft::ModuleLikeUnit::ModuleLikeUnit(const parser::Submodule &m,
                                    const pft::ParentVariant &parent)
    : ProgramUnit{m, parent}, beginStmt{getModuleStmt<parser::SubmoduleStmt>(
                                  m)},
      endStmt{getModuleStmt<parser::EndSubmoduleStmt>(m)} {}

pft::BlockDataUnit::BlockDataUnit(const parser::BlockData &bd,
                                  const pft::ParentVariant &parent)
    : ProgramUnit{bd, parent} {}

std::unique_ptr<pft::Program> createPFT(const parser::Program &root) {
  PFTBuilder walker;
  Walk(root, walker);
  return walker.result();
}

void dumpPFT(llvm::raw_ostream &outputStream, pft::Program &pft) {
  PFTDumper{}.dumpPFT(outputStream, pft);
}

void pft::Program::dump() { dumpPFT(llvm::errs(), *this); }

} // namespace Fortran::lower
