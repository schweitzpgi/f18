//===-- PFTBuilder.cc -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/lower/PFTBuilder.h"
#include "NSAliases.h"
#include "flang/parser/dump-parse-tree.h"
#include "flang/parser/parse-tree-visitor.h"
#include "llvm/ADT/DenseMap.h"
#include <algorithm>
#include <cassert>
#include <utility>

namespace Fortran::lower {
namespace {

/// Helpers to unveil parser node inside Pa::Statement<>,
/// Pa::UnlabeledStatement, and Co::Indirection<>
template <typename A>
struct RemoveIndirectionHelper {
  using Type = A;
  static constexpr const Type &unwrap(const A &a) { return a; }
};
template <typename A>
struct RemoveIndirectionHelper<Co::Indirection<A>> {
  using Type = A;
  static constexpr const Type &unwrap(const Co::Indirection<A> &a) {
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
struct UnwrapStmt<Pa::Statement<A>> {
  static constexpr bool isStmt{true};
  using Type = typename RemoveIndirectionHelper<A>::Type;
  constexpr UnwrapStmt(const Pa::Statement<A> &a)
      : unwrapped{removeIndirection(a.statement)}, pos{a.source}, lab{a.label} {
  }
  const Type &unwrapped;
  Pa::CharBlock pos;
  std::optional<Pa::Label> lab;
};
template <typename A>
struct UnwrapStmt<Pa::UnlabeledStatement<A>> {
  static constexpr bool isStmt{true};
  using Type = typename RemoveIndirectionHelper<A>::Type;
  constexpr UnwrapStmt(const Pa::UnlabeledStatement<A> &a)
      : unwrapped{removeIndirection(a.statement)}, pos{a.source} {}
  const Type &unwrapped;
  Pa::CharBlock pos;
  std::optional<Pa::Label> lab;
};

/// The instantiation of a parse tree visitor (Pre and Post) is extremely
/// expensive in terms of compile and link time, so one goal here is to limit
/// the bridge to one such instantiation.
class PFTBuilder {
public:
  PFTBuilder() : pgm{new PFT::Program}, parents{*pgm.get()} {}

  /// Get the result
  std::unique_ptr<PFT::Program> result() { return std::move(pgm); }

  template <typename A>
  constexpr bool Pre(const A &a) {
    bool visit{true};
    if constexpr (PFT::isFunctionLike<A>) {
      return enterFunc(a);
    } else if constexpr (PFT::isConstruct<A>) {
      return enterConstruct(a);
    } else if constexpr (UnwrapStmt<A>::isStmt) {
      using T = typename UnwrapStmt<A>::Type;
      // Node "a" being visited has one of the following types:
      // Statement<T>, Statement<Indirection<T>, UnlabeledStatement<T>,
      // or UnlabeledStatement<Indirection<T>>
      auto stmt{UnwrapStmt<A>(a)};
      if constexpr (PFT::isConstructStmt<T> || PFT::isOtherStmt<T>) {
        addEval(PFT::Evaluation{stmt.unwrapped, parents.back(), stmt.pos,
                                stmt.lab});
        visit = false;
      } else if constexpr (std::is_same_v<T, Pa::ActionStmt>) {
        addEval(makeEvalAction(stmt.unwrapped, stmt.pos, stmt.lab));
        visit = false;
      }
    }
    return visit;
  }

  template <typename A>
  constexpr void Post(const A &) {
    if constexpr (PFT::isFunctionLike<A>) {
      exitFunc();
    } else if constexpr (PFT::isConstruct<A>) {
      exitConstruct();
    }
  }

  // Module like
  bool Pre(const Pa::Module &node) { return enterModule(node); }
  bool Pre(const Pa::Submodule &node) { return enterModule(node); }

  void Post(const Pa::Module &) { exitModule(); }
  void Post(const Pa::Submodule &) { exitModule(); }

  // Block data
  bool Pre(const Pa::BlockData &node) {
    addUnit(PFT::BlockDataUnit{node, parents.back()});
    return false;
  }

  // Get rid of production wrapper
  bool Pre(const Pa::UnlabeledStatement<Pa::ForallAssignmentStmt>
               &statement) {
    addEval(std::visit(
        [&](const auto &x) {
          return PFT::Evaluation{x, parents.back(), statement.source, {}};
        },
        statement.statement.u));
    return false;
  }
  bool Pre(const Pa::Statement<Pa::ForallAssignmentStmt> &statement) {
    addEval(std::visit(
        [&](const auto &x) {
          return PFT::Evaluation{x, parents.back(), statement.source,
                                 statement.label};
        },
        statement.statement.u));
    return false;
  }
  bool Pre(const Pa::WhereBodyConstruct &whereBody) {
    return std::visit(
        Co::visitors{
            [&](const Pa::Statement<Pa::AssignmentStmt> &stmt) {
              // Not caught as other AssignmentStmt because it is not
              // wrapped in a Pa::ActionStmt.
              addEval(PFT::Evaluation{stmt.statement, parents.back(),
                                      stmt.source, stmt.label});
              return false;
            },
            [&](const auto &) { return true; },
        },
        whereBody.u);
  }

private:
  // ActionStmt has a couple of non-conforming cases, which get handled
  // explicitly here.  The other cases use an Indirection, which we discard in
  // the PFT.
  PFT::Evaluation makeEvalAction(const Pa::ActionStmt &statement,
                                 Pa::CharBlock pos,
                                 std::optional<Pa::Label> lab) {
    return std::visit(
        Co::visitors{
            [&](const auto &x) {
              return PFT::Evaluation{removeIndirection(x), parents.back(), pos,
                                     lab};
            },
        },
        statement.u);
  }

  // When we enter a function-like structure, we want to build a new unit and
  // set the builder's cursors to point to it.
  template <typename A>
  bool enterFunc(const A &func) {
    auto &unit = addFunc(PFT::FunctionLikeUnit{func, parents.back()});
    funclist = &unit.funcs;
    pushEval(&unit.evals);
    parents.emplace_back(unit);
    return true;
  }
  /// Make funclist to point to current parent function list if it exists.
  void setFunctListToParentFuncs() {
    if (!parents.empty()) {
      std::visit(Co::visitors{
                     [&](PFT::FunctionLikeUnit *p) { funclist = &p->funcs; },
                     [&](PFT::ModuleLikeUnit *p) { funclist = &p->funcs; },
                     [&](auto *) { funclist = nullptr; },
                 },
                 parents.back().p);
    }
  }

  void exitFunc() {
    popEval();
    parents.pop_back();
    setFunctListToParentFuncs();
  }

  // When we enter a construct structure, we want to build a new construct and
  // set the builder's evaluation cursor to point to it.
  template <typename A>
  bool enterConstruct(const A &construct) {
    auto &con = addEval(PFT::Evaluation{construct, parents.back()});
    con.subs.reset(new PFT::EvaluationCollection);
    pushEval(con.subs.get());
    parents.emplace_back(con);
    return true;
  }

  void exitConstruct() {
    popEval();
    parents.pop_back();
  }

  // When we enter a module structure, we want to build a new module and
  // set the builder's function cursor to point to it.
  template <typename A>
  bool enterModule(const A &func) {
    auto &unit = addUnit(PFT::ModuleLikeUnit{func, parents.back()});
    funclist = &unit.funcs;
    parents.emplace_back(unit);
    return true;
  }

  void exitModule() {
    parents.pop_back();
    setFunctListToParentFuncs();
  }

  template <typename A>
  A &addUnit(A &&unit) {
    pgm->getUnits().emplace_back(std::move(unit));
    return std::get<A>(pgm->getUnits().back());
  }

  template <typename A>
  A &addFunc(A &&func) {
    if (funclist) {
      funclist->emplace_back(std::move(func));
      return funclist->back();
    }
    return addUnit(std::move(func));
  }

  /// move the Evaluation to the end of the current list
  PFT::Evaluation &addEval(PFT::Evaluation &&eval) {
    assert(funclist && "not in a function");
    assert(evallist.size() > 0);
    evallist.back()->emplace_back(std::move(eval));
    return evallist.back()->back();
  }

  /// push a new list on the stack of Evaluation lists
  void pushEval(PFT::EvaluationCollection *eval) {
    assert(funclist && "not in a function");
    assert(eval && eval->empty() && "evaluation list isn't correct");
    evallist.emplace_back(eval);
  }

  /// pop the current list and return to the last Evaluation list
  void popEval() {
    assert(funclist && "not in a function");
    evallist.pop_back();
  }

  std::unique_ptr<PFT::Program> pgm;
  /// funclist points to FunctionLikeUnit::funcs list (resp.
  /// ModuleLikeUnit::funcs) when building a FunctionLikeUnit (resp.
  /// ModuleLikeUnit) to store internal procedures (resp. module procedures).
  /// Otherwise (e.g. when building the top level Program), it is null.
  std::list<PFT::FunctionLikeUnit> *funclist{nullptr};
  /// evallist is a stack of pointer to FunctionLikeUnit::evals (or
  /// Evaluation::subs) that are being build.
  std::vector<PFT::EvaluationCollection *> evallist;
  std::vector<PFT::ParentType> parents;
};

template <typename Label, typename A>
constexpr bool hasLabel(const A &stmt) {
  auto isLabel{
      [](const auto &v) { return std::holds_alternative<Label>(v.u); }};
  if constexpr (std::is_same_v<A, Pa::ReadStmt> ||
                std::is_same_v<A, Pa::WriteStmt>) {
    return std::any_of(std::begin(stmt.controls), std::end(stmt.controls),
                       isLabel);
  }
  if constexpr (std::is_same_v<A, Pa::WaitStmt>) {
    return std::any_of(std::begin(stmt.v), std::end(stmt.v), isLabel);
  }
  if constexpr (std::is_same_v<Label, Pa::ErrLabel>) {
    if constexpr (Co::HasMember<
                      A, std::tuple<Pa::OpenStmt, Pa::CloseStmt,
                                    Pa::BackspaceStmt, Pa::EndfileStmt,
                                    Pa::RewindStmt, Pa::FlushStmt>>)
      return std::any_of(std::begin(stmt.v), std::end(stmt.v), isLabel);
    if constexpr (std::is_same_v<A, Pa::InquireStmt>) {
      const auto &specifiers{std::get<std::list<Pa::InquireSpec>>(stmt.u)};
      return std::any_of(std::begin(specifiers), std::end(specifiers), isLabel);
    }
  }
  return false;
}

bool hasAltReturns(const Pa::CallStmt &callStmt) {
  const auto &args{std::get<std::list<Pa::ActualArgSpec>>(callStmt.v.t)};
  for (const auto &arg : args) {
    const auto &actual{std::get<Pa::ActualArg>(arg.t)};
    if (std::holds_alternative<Pa::AltReturnSpec>(actual.u))
      return true;
  }
  return false;
}

/// Determine if `callStmt` has alternate returns and if so set `e` to be the
/// origin of a switch-like control flow
///
/// \param cstr points to the current construct. It may be null at the top-level
/// of a FunctionLikeUnit.
void altRet(PFT::Evaluation &evaluation, const Pa::CallStmt &callStmt,
            PFT::Evaluation *cstr) {
  if (hasAltReturns(callStmt))
    evaluation.setCFG(PFT::CFGAnnotation::Switch, cstr);
}

/// \param cstr points to the current construct. It may be null at the top-level
/// of a FunctionLikeUnit.
void annotateEvalListCFG(PFT::EvaluationCollection &evaluationCollection,
                         PFT::Evaluation *cstr) {
  bool nextIsTarget = false;
  for (auto &eval : evaluationCollection) {
    eval.isTarget = nextIsTarget;
    nextIsTarget = false;
    if (auto *subs{eval.getConstructEvals()}) {
      annotateEvalListCFG(*subs, &eval);
      // assume that the entry and exit are both possible branch targets
      nextIsTarget = true;
    }

    if (eval.isActionOrGenerated() && eval.lab.has_value())
      eval.isTarget = true;
    eval.visit(Co::visitors{
        [&](const Pa::CallStmt &statement) {
          altRet(eval, statement, cstr);
        },
        [&](const Pa::CycleStmt &) {
          eval.setCFG(PFT::CFGAnnotation::Goto, cstr);
        },
        [&](const Pa::ExitStmt &) {
          eval.setCFG(PFT::CFGAnnotation::Goto, cstr);
        },
        [&](const Pa::FailImageStmt &) {
          eval.setCFG(PFT::CFGAnnotation::Terminate, cstr);
        },
        [&](const Pa::GotoStmt &) {
          eval.setCFG(PFT::CFGAnnotation::Goto, cstr);
        },
        [&](const Pa::IfStmt &) {
          eval.setCFG(PFT::CFGAnnotation::CondGoto, cstr);
        },
        [&](const Pa::ReturnStmt &) {
          eval.setCFG(PFT::CFGAnnotation::Return, cstr);
        },
        [&](const Pa::StopStmt &) {
          eval.setCFG(PFT::CFGAnnotation::Terminate, cstr);
        },
        [&](const Pa::ArithmeticIfStmt &) {
          eval.setCFG(PFT::CFGAnnotation::Switch, cstr);
        },
        [&](const Pa::AssignedGotoStmt &) {
          eval.setCFG(PFT::CFGAnnotation::IndGoto, cstr);
        },
        [&](const Pa::ComputedGotoStmt &) {
          eval.setCFG(PFT::CFGAnnotation::Switch, cstr);
        },
        [&](const Pa::WhereStmt &) {
          // fir.loop + fir.where around the next stmt
          eval.isTarget = true;
          eval.setCFG(PFT::CFGAnnotation::Iterative, cstr);
        },
        [&](const Pa::ForallStmt &) {
          // fir.loop around the next stmt
          eval.isTarget = true;
          eval.setCFG(PFT::CFGAnnotation::Iterative, cstr);
        },
        [&](PFT::CGJump &) { eval.setCFG(PFT::CFGAnnotation::Goto, cstr); },
        [&](const Pa::SelectCaseStmt &) {
          eval.setCFG(PFT::CFGAnnotation::Switch, cstr);
        },
        [&](const Pa::NonLabelDoStmt &) {
          eval.isTarget = true;
          eval.setCFG(PFT::CFGAnnotation::Iterative, cstr);
        },
        [&](const Pa::EndDoStmt &) {
          eval.isTarget = true;
          eval.setCFG(PFT::CFGAnnotation::Goto, cstr);
        },
        [&](const Pa::IfThenStmt &) {
          eval.setCFG(PFT::CFGAnnotation::CondGoto, cstr);
        },
        [&](const Pa::ElseIfStmt &) {
          eval.setCFG(PFT::CFGAnnotation::CondGoto, cstr);
        },
        [&](const Pa::SelectRankStmt &) {
          eval.setCFG(PFT::CFGAnnotation::Switch, cstr);
        },
        [&](const Pa::SelectTypeStmt &) {
          eval.setCFG(PFT::CFGAnnotation::Switch, cstr);
        },
        [&](const Pa::WhereConstruct &) {
          // mark the WHERE as if it were a DO loop
          eval.isTarget = true;
          eval.setCFG(PFT::CFGAnnotation::Iterative, cstr);
        },
        [&](const Pa::WhereConstructStmt &) {
          eval.setCFG(PFT::CFGAnnotation::CondGoto, cstr);
        },
        [&](const Pa::MaskedElsewhereStmt &) {
          eval.isTarget = true;
          eval.setCFG(PFT::CFGAnnotation::CondGoto, cstr);
        },
        [&](const Pa::ForallConstructStmt &) {
          eval.isTarget = true;
          eval.setCFG(PFT::CFGAnnotation::Iterative, cstr);
        },

        [&](const auto &stmt) {
          // Handle statements with similar impact on control flow
          using IoStmts = std::tuple<Pa::BackspaceStmt, Pa::CloseStmt,
                                     Pa::EndfileStmt, Pa::FlushStmt,
                                     Pa::InquireStmt, Pa::OpenStmt,
                                     Pa::ReadStmt, Pa::RewindStmt,
                                     Pa::WaitStmt, Pa::WriteStmt>;

          using TargetStmts =
              std::tuple<Pa::EndAssociateStmt, Pa::EndBlockStmt,
                         Pa::CaseStmt, Pa::EndSelectStmt,
                         Pa::EndChangeTeamStmt, Pa::EndCriticalStmt,
                         Pa::ElseStmt, Pa::EndIfStmt,
                         Pa::SelectRankCaseStmt, Pa::TypeGuardStmt,
                         Pa::ElsewhereStmt, Pa::EndWhereStmt,
                         Pa::EndForallStmt>;

          using DoNothingConstructStmts =
              std::tuple<Pa::BlockStmt, Pa::AssociateStmt,
                         Pa::CriticalStmt, Pa::ChangeTeamStmt>;

          using A = std::decay_t<decltype(stmt)>;
          if constexpr (Co::HasMember<A, IoStmts>) {
            if (hasLabel<Pa::ErrLabel>(stmt) ||
                hasLabel<Pa::EorLabel>(stmt) ||
                hasLabel<Pa::EndLabel>(stmt))
              eval.setCFG(PFT::CFGAnnotation::IoSwitch, cstr);
          } else if constexpr (Co::HasMember<A, TargetStmts>) {
            eval.isTarget = true;
          } else if constexpr (Co::HasMember<A, DoNothingConstructStmts>) {
            // Explicitly do nothing for these construct statements
          } else {
            static_assert(!PFT::isConstructStmt<A>,
                          "All ConstructStmts impact on the control flow "
                          "should be explicitly handled");
          }
          /* else do nothing */
        },
    });
  }
}

/// Annotate the PFT with CFG source decorations (see CFGAnnotation) and mark
/// potential branch targets
inline void annotateFuncCFG(PFT::FunctionLikeUnit &functionLikeUnit) {
  annotateEvalListCFG(functionLikeUnit.evals, nullptr);
  for (auto &internalFunc : functionLikeUnit.funcs)
    annotateFuncCFG(internalFunc);
}

class PFTDumper {
public:
  void dumpPFT(L::raw_ostream &outputStream, PFT::Program &pft) {
    for (auto &unit : pft.getUnits()) {
      std::visit(Co::visitors{
                     [&](PFT::BlockDataUnit &unit) {
                       outputStream << getNodeIndex(unit) << " ";
                       outputStream << "BlockData: ";
                       outputStream << "\nEndBlockData\n\n";
                     },
                     [&](PFT::FunctionLikeUnit &func) {
                       dumpFunctionLikeUnit(outputStream, func);
                     },
                     [&](PFT::ModuleLikeUnit &unit) {
                       dumpModuleLikeUnit(outputStream, unit);
                     },
                 },
                 unit);
    }
    resetIndexes();
  }

  L::StringRef evalName(PFT::Evaluation &eval) {
    return eval.visit(Co::visitors{
        [](const PFT::CGJump) { return "CGJump"; },
        [](const auto &parseTreeNode) {
          return Pa::ParseTreeDumper::GetNodeName(parseTreeNode);
        },
    });
  }

  void dumpEvalList(L::raw_ostream &outputStream,
                    PFT::EvaluationCollection &evaluationCollection,
                    int indent = 1) {
    static const std::string white{"                                      ++"};
    std::string indentString{white.substr(0, indent * 2)};
    for (PFT::Evaluation &eval : evaluationCollection) {
      outputStream << indentString << getNodeIndex(eval) << " ";
      L::StringRef name{evalName(eval)};
      if (auto *subs{eval.getConstructEvals()}) {
        outputStream << "<<" << name << ">>";
        outputStream << "\n";
        dumpEvalList(outputStream, *subs, indent + 1);
        outputStream << indentString << "<<End" << name << ">>\n";
      } else {
        outputStream << name;
        outputStream << ": " << eval.pos.ToString() + "\n";
      }
    }
  }

  void dumpFunctionLikeUnit(L::raw_ostream &outputStream,
                            PFT::FunctionLikeUnit &functionLikeUnit) {
    outputStream << getNodeIndex(functionLikeUnit) << " ";
    L::StringRef unitKind{};
    std::string name{};
    std::string header{};
    if (functionLikeUnit.beginStmt) {
      std::visit(
          Co::visitors{
              [&](const Pa::Statement<Pa::ProgramStmt> *statement) {
                unitKind = "Program";
                name = statement->statement.v.ToString();
              },
              [&](const Pa::Statement<Pa::FunctionStmt> *statement) {
                unitKind = "Function";
                name =
                    std::get<Pa::Name>(statement->statement.t).ToString();
                header = statement->source.ToString();
              },
              [&](const Pa::Statement<Pa::SubroutineStmt> *statement) {
                unitKind = "Subroutine";
                name =
                    std::get<Pa::Name>(statement->statement.t).ToString();
                header = statement->source.ToString();
              },
              [&](const Pa::Statement<Pa::MpSubprogramStmt>
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
    dumpEvalList(outputStream, functionLikeUnit.evals);
    if (!functionLikeUnit.funcs.empty()) {
      outputStream << "\nContains\n";
      for (auto &func : functionLikeUnit.funcs)
        dumpFunctionLikeUnit(outputStream, func);
      outputStream << "EndContains\n";
    }
    outputStream << "End" << unitKind << ' ' << name << "\n\n";
  }

  void dumpModuleLikeUnit(L::raw_ostream &outputStream,
                          PFT::ModuleLikeUnit &moduleLikeUnit) {
    outputStream << getNodeIndex(moduleLikeUnit) << " ";
    outputStream << "ModuleLike: ";
    outputStream << "\nContains\n";
    for (auto &func : moduleLikeUnit.funcs)
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
  std::size_t getNodeIndex(const PFT::Program &) { return 0; }

  void resetIndexes() {
    nodeIndexes.clear();
    nextIndex = 1;
  }

private:
  L::DenseMap<const void *, std::size_t> nodeIndexes;
  std::size_t nextIndex{1}; // 0 is the root
};

template <typename A, typename T>
PFT::FunctionLikeUnit::FunctionStatement getFunctionStmt(const T &func) {
  return PFT::FunctionLikeUnit::FunctionStatement{
      &std::get<Pa::Statement<A>>(func.t)};
}
template <typename A, typename T>
PFT::ModuleLikeUnit::ModuleStatement getModuleStmt(const T &mod) {
  return PFT::ModuleLikeUnit::ModuleStatement{
      &std::get<Pa::Statement<A>>(mod.t)};
}

} // namespace

PFT::FunctionLikeUnit::FunctionLikeUnit(const Pa::MainProgram &func,
                                        const PFT::ParentType &parent)
    : ProgramUnit{func, parent} {
  auto &ps{
      std::get<std::optional<Pa::Statement<Pa::ProgramStmt>>>(func.t)};
  if (ps.has_value()) {
    const Pa::Statement<Pa::ProgramStmt> &statement{ps.value()};
    beginStmt = &statement;
  }
  endStmt = getFunctionStmt<Pa::EndProgramStmt>(func);
}

PFT::FunctionLikeUnit::FunctionLikeUnit(const Pa::FunctionSubprogram &func,
                                        const PFT::ParentType &parent)
    : ProgramUnit{func, parent},
      beginStmt{getFunctionStmt<Pa::FunctionStmt>(func)},
      endStmt{getFunctionStmt<Pa::EndFunctionStmt>(func)} {}

PFT::FunctionLikeUnit::FunctionLikeUnit(
    const Pa::SubroutineSubprogram &func, const PFT::ParentType &parent)
    : ProgramUnit{func, parent},
      beginStmt{getFunctionStmt<Pa::SubroutineStmt>(func)},
      endStmt{getFunctionStmt<Pa::EndSubroutineStmt>(func)} {}

PFT::FunctionLikeUnit::FunctionLikeUnit(
    const Pa::SeparateModuleSubprogram &func, const PFT::ParentType &parent)
    : ProgramUnit{func, parent},
      beginStmt{getFunctionStmt<Pa::MpSubprogramStmt>(func)},
      endStmt{getFunctionStmt<Pa::EndMpSubprogramStmt>(func)} {}

PFT::ModuleLikeUnit::ModuleLikeUnit(const Pa::Module &m,
                                    const PFT::ParentType &parent)
    : ProgramUnit{m, parent}, beginStmt{getModuleStmt<Pa::ModuleStmt>(m)},
      endStmt{getModuleStmt<Pa::EndModuleStmt>(m)} {}

PFT::ModuleLikeUnit::ModuleLikeUnit(const Pa::Submodule &m,
                                    const PFT::ParentType &parent)
    : ProgramUnit{m, parent}, beginStmt{getModuleStmt<Pa::SubmoduleStmt>(
                                  m)},
      endStmt{getModuleStmt<Pa::EndSubmoduleStmt>(m)} {}

PFT::BlockDataUnit::BlockDataUnit(const Pa::BlockData &bd,
                                  const PFT::ParentType &parent)
    : ProgramUnit{bd, parent} {}

std::unique_ptr<PFT::Program> createPFT(const Pa::Program &root) {
  PFTBuilder walker;
  Walk(root, walker);
  return walker.result();
}

void annotateControl(PFT::Program &pft) {
  for (auto &unit : pft.getUnits()) {
    std::visit(Co::visitors{
                   [](PFT::BlockDataUnit &) {},
                   [](PFT::FunctionLikeUnit &func) { annotateFuncCFG(func); },
                   [](PFT::ModuleLikeUnit &unit) {
                     for (auto &func : unit.funcs)
                       annotateFuncCFG(func);
                   },
               },
               unit);
  }
}

void dumpPFT(L::raw_ostream &outputStream, PFT::Program &pft) {
  PFTDumper{}.dumpPFT(outputStream, pft);
}

void PFT::Program::dump() {
  dumpPFT(L::errs(), *this);
}

} // namespace Fortran::lower
