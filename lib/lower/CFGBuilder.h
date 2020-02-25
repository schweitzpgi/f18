//===-- lower/cfg-builder.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_CFGBUILDER_H_
#define FORTRAN_LOWER_CFGBUILDER_H_

/// Traverse the PFT and complete the CFG by drawing the arcs, pruning unused
/// potential targets, making implied jumps explicit, etc.
class CfgBuilder {

  Fortran::lower::pft::Evaluation *
  getEvalByLabel(const Fortran::parser::Label &label) {
    auto iter = labels.find(label);
    if (iter != labels.end()) {
      return iter->second;
    }
    return nullptr;
  }

  /// Collect all the potential targets and initialize them to unreferenced
  void
  resetPotentialTargets(std::list<Fortran::lower::pft::Evaluation> &evals) {
    for (auto &e : evals) {
      if (e.isTarget) {
        e.isTarget = false;
      }
      if (e.lab.has_value()) {
        labels.try_emplace(*e.lab, &e);
      }
      if (e.subs) {
        resetPotentialTargets(*e.subs);
      }
    }
  }

  /// cache ASSIGN statements that may yield a live branch target
  void cacheAssigns(std::list<Fortran::lower::pft::Evaluation> &evals) {
    for (auto &e : evals) {
      e.visit(Fortran::common::visitors{
          [&](const Fortran::parser::AssignStmt &stmt) {
            auto *trg =
                getEvalByLabel(std::get<Fortran::parser::Label>(stmt.t));
            auto *sym = std::get<Fortran::parser::Name>(stmt.t).symbol;
            assert(sym);
            auto jter = assignedGotoMap.find(sym);
            if (jter == assignedGotoMap.end()) {
              std::list<Fortran::lower::pft::Evaluation *> lst = {trg};
              assignedGotoMap.try_emplace(sym, lst);
            } else {
              jter->second.emplace_back(trg);
            }
          },
          [](auto &) { /* do nothing */ },
      });
      if (e.subs) {
        cacheAssigns(*e.subs);
      }
    }
  }

  void deannotate(std::list<Fortran::lower::pft::Evaluation> &evals) {
    for (auto &e : evals) {
      e.cfg = Fortran::lower::pft::CFGAnnotation::None;
      if (e.subs) {
        deannotate(*e.subs);
      }
    }
  }

  bool structuredCheck(std::list<Fortran::lower::pft::Evaluation> &evals) {
    for (auto &e : evals) {
      if (auto *s = e.getIf<Fortran::parser::DoConstruct>()) {
        return s->IsDoWhile() ? false : structuredCheck(*e.subs);
      }
      if (e.isA<Fortran::parser::IfConstruct>()) {
        return structuredCheck(*e.subs);
      }
      if (e.subs) {
        return false;
      }
      switch (e.cfg) {
      case Fortran::lower::pft::CFGAnnotation::None:
        break;
      case Fortran::lower::pft::CFGAnnotation::CondGoto:
        break;
      case Fortran::lower::pft::CFGAnnotation::Iterative:
        break;
      case Fortran::lower::pft::CFGAnnotation::FirStructuredOp:
        break;
      case Fortran::lower::pft::CFGAnnotation::IndGoto:
        return false;
      case Fortran::lower::pft::CFGAnnotation::IoSwitch:
        return false;
      case Fortran::lower::pft::CFGAnnotation::Switch:
        return false;
      case Fortran::lower::pft::CFGAnnotation::Return:
        return false;
      case Fortran::lower::pft::CFGAnnotation::Terminate:
        return false;
      case Fortran::lower::pft::CFGAnnotation::Goto:
        if (!e.isA<Fortran::parser::EndDoStmt>()) {
          return false;
        }
        break;
      }
    }
    return true;
  }

  void wrapIterationSpaces(std::list<Fortran::lower::pft::Evaluation> &evals) {
    for (auto &e : evals) {
      if (e.isA<Fortran::parser::DoConstruct>())
        if (structuredCheck(*e.subs)) {
          deannotate(*e.subs);
          e.cfg = Fortran::lower::pft::CFGAnnotation::FirStructuredOp;
          continue;
        }
      if (e.isA<Fortran::parser::IfConstruct>())
        if (structuredCheck(*e.subs)) {
          deannotate(*e.subs);
          e.cfg = Fortran::lower::pft::CFGAnnotation::FirStructuredOp;
          continue;
        }
      // FIXME: ForallConstruct? WhereConstruct?
      if (e.subs) {
        wrapIterationSpaces(*e.subs);
      }
    }
  }

  /// Add source->sink edge to CFG map
  void addSourceToSink(Fortran::lower::pft::Evaluation *src,
                       Fortran::lower::pft::Evaluation *snk) {
    auto iter = cfgMap.find(src);
    if (iter == cfgMap.end()) {
      CFGSinkListType sink{snk};
      cfgEdgeSetPool.emplace_back(std::move(sink));
      auto rc{cfgMap.try_emplace(src, &cfgEdgeSetPool.back())};
      assert(rc.second && "insert failed unexpectedly");
      (void)rc; // for release build
      return;
    }
    for (auto *s : *iter->second)
      if (s == snk) {
        return;
      }
    iter->second->push_back(snk);
  }

  void addSourceToSink(Fortran::lower::pft::Evaluation *src,
                       const Fortran::parser::Label &label) {
    auto iter = labels.find(label);
    assert(iter != labels.end());
    addSourceToSink(src, iter->second);
  }

  /// Find the next ELSE IF, ELSE or END IF statement in the list
  template <typename A>
  A nextFalseTarget(A iter, const A &endi) {
    for (; iter != endi; ++iter)
      if (iter->visit(Fortran::common::visitors{
              [&](const Fortran::parser::ElseIfStmt &) { return true; },
              [&](const Fortran::parser::ElseStmt &) { return true; },
              [&](const Fortran::parser::EndIfStmt &) { return true; },
              [](auto &) { return false; },
          })) {
        break;
      }
    return iter;
  }

  /// Add branches for this IF block like construct.
  /// Branch to the "true block", the "false block", and from the end of the
  /// true block to the end of the construct.
  template <typename A>
  void doNextIfBlock(std::list<Fortran::lower::pft::Evaluation> &evals,
                     Fortran::lower::pft::Evaluation &e, const A &iter,
                     const A &endif) {
    A i{iter};
    A j{nextFalseTarget(++i, endif)};
    auto *cstr = std::get<Fortran::lower::pft::Evaluation *>(e.parent.p);
    Fortran::lower::pft::CGJump jump{*endif};
    A k{evals.insert(
        j, Fortran::lower::pft::Evaluation{std::move(jump), j->parent})};
    if (i == j) {
      // block was empty, so adjust "true" target
      i = k;
    }
    addSourceToSink(&*k, cstr);
    addSourceToSink(&e, &*i);
    addSourceToSink(&e, &*j);
  }

  /// Determine which branch targets are reachable. The target map must
  /// already be initialized.
  void reachabilityAnalysis(std::list<Fortran::lower::pft::Evaluation> &evals) {
    for (auto iter = evals.begin(); iter != evals.end(); ++iter) {
      auto &e = *iter;
      switch (e.cfg) {
      case Fortran::lower::pft::CFGAnnotation::None:
        // do nothing - does not impart control flow
        break;
      case Fortran::lower::pft::CFGAnnotation::Goto:
        e.visit(Fortran::common::visitors{
            [&](const Fortran::parser::CycleStmt &) {
              // FIXME: deal with construct name
              auto *cstr =
                  std::get<Fortran::lower::pft::Evaluation *>(e.parent.p);
              addSourceToSink(&e, &cstr->subs->front());
            },
            [&](const Fortran::parser::ExitStmt &) {
              // FIXME: deal with construct name
              auto *cstr =
                  std::get<Fortran::lower::pft::Evaluation *>(e.parent.p);
              addSourceToSink(&e, &cstr->subs->back());
            },
            [&](const Fortran::parser::GotoStmt &stmt) {
              addSourceToSink(&e, stmt.v);
            },
            [&](const Fortran::parser::EndDoStmt &) {
              // the END DO is the loop exit landing pad
              // insert a JUMP as the backedge right before the END DO
              auto *cstr =
                  std::get<Fortran::lower::pft::Evaluation *>(e.parent.p);
              Fortran::lower::pft::CGJump jump{cstr->subs->front()};
              Fortran::lower::pft::Evaluation jumpEval{std::move(jump),
                                                       iter->parent};
              evals.insert(iter, std::move(jumpEval));
              addSourceToSink(&e, &cstr->subs->front());
            },
            [&](const Fortran::lower::pft::CGJump &jump) {
              addSourceToSink(&e, &jump.target);
            },
            [](auto &) { assert(false && "unhandled GOTO case"); },
        });
        break;
      case Fortran::lower::pft::CFGAnnotation::CondGoto:
        e.visit(Fortran::common::visitors{
            [&](const Fortran::parser::IfStmt &) {
              // check if these are marked; they must targets here
              auto i{iter};
              addSourceToSink(&e, &*(++i));
              addSourceToSink(&e, &*(++i));
            },
            [&](const Fortran::parser::IfThenStmt &) {
              doNextIfBlock(evals, e, iter, evals.end());
            },
            [&](const Fortran::parser::ElseIfStmt &) {
              doNextIfBlock(evals, e, iter, evals.end());
            },
            [](const Fortran::parser::WhereConstructStmt &) { TODO(); },
            [](const Fortran::parser::MaskedElsewhereStmt &) { TODO(); },
            [](auto &) { assert(false && "unhandled CGOTO case"); },
        });
        break;
      case Fortran::lower::pft::CFGAnnotation::IndGoto:
        e.visit(Fortran::common::visitors{
            [&](const Fortran::parser::AssignedGotoStmt &stmt) {
              auto *sym = std::get<Fortran::parser::Name>(stmt.t).symbol;
              if (assignedGotoMap.find(sym) != assignedGotoMap.end())
                for (auto *x : assignedGotoMap[sym]) {
                  addSourceToSink(&e, x);
                }
              for (auto &l :
                   std::get<std::list<Fortran::parser::Label>>(stmt.t)) {
                addSourceToSink(&e, l);
              }
            },
            [](auto &) { assert(false && "unhandled IGOTO case"); },
        });
        break;
      case Fortran::lower::pft::CFGAnnotation::IoSwitch:
        e.visit(Fortran::common::visitors{
            [](const Fortran::parser::BackspaceStmt &) { TODO(); },
            [](const Fortran::parser::CloseStmt &) { TODO(); },
            [](const Fortran::parser::EndfileStmt &) { TODO(); },
            [](const Fortran::parser::FlushStmt &) { TODO(); },
            [](const Fortran::parser::InquireStmt &) { TODO(); },
            [](const Fortran::parser::OpenStmt &) { TODO(); },
            [](const Fortran::parser::ReadStmt &) { TODO(); },
            [](const Fortran::parser::RewindStmt &) { TODO(); },
            [](const Fortran::parser::WaitStmt &) { TODO(); },
            [](const Fortran::parser::WriteStmt &) { TODO(); },
            [](auto &) { assert(false && "unhandled IO switch case"); },
        });
        break;
      case Fortran::lower::pft::CFGAnnotation::Switch:
        e.visit(Fortran::common::visitors{
            [](const Fortran::parser::CallStmt &) { TODO(); },
            [](const Fortran::parser::ArithmeticIfStmt &) { TODO(); },
            [](const Fortran::parser::ComputedGotoStmt &) { TODO(); },
            [](const Fortran::parser::SelectCaseStmt &) { TODO(); },
            [](const Fortran::parser::SelectRankStmt &) { TODO(); },
            [](const Fortran::parser::SelectTypeStmt &) { TODO(); },
            [](auto &) { assert(false && "unhandled switch case"); },
        });
        break;
      case Fortran::lower::pft::CFGAnnotation::Iterative:
        e.visit(Fortran::common::visitors{
            [](const Fortran::parser::NonLabelDoStmt &) { TODO(); },
            [](const Fortran::parser::WhereStmt &) { TODO(); },
            [](const Fortran::parser::ForallStmt &) { TODO(); },
            [](const Fortran::parser::WhereConstruct &) { TODO(); },
            [](const Fortran::parser::ForallConstructStmt &) { TODO(); },
            [](auto &) { assert(false && "unhandled loop case"); },
        });
        break;
      case Fortran::lower::pft::CFGAnnotation::FirStructuredOp:
        // do not visit the subs
        continue;
      case Fortran::lower::pft::CFGAnnotation::Return:
        // do nothing - exits the function
        break;
      case Fortran::lower::pft::CFGAnnotation::Terminate:
        // do nothing - exits the function
        break;
      }
      if (e.subs) {
        reachabilityAnalysis(*e.subs);
      }
    }
  }

  void setActualTargets(std::list<Fortran::lower::pft::Evaluation> &) {
    for (auto &lst1 : cfgEdgeSetPool)
      for (auto *e : lst1) {
        e->isTarget = true;
      }
  }

  CFGMapType &cfgMap;
  std::list<CFGSinkListType> &cfgEdgeSetPool;

  llvm::DenseMap<Fortran::parser::Label, Fortran::lower::pft::Evaluation *>
      labels;
  std::map<Fortran::semantics::Symbol *,
           std::list<Fortran::lower::pft::Evaluation *>>
      assignedGotoMap;

public:
  CfgBuilder(CFGMapType &m, std::list<CFGSinkListType> &p)
      : cfgMap{m}, cfgEdgeSetPool{p} {}

  void run(Fortran::lower::pft::FunctionLikeUnit &func) {
    resetPotentialTargets(func.evals);
    cacheAssigns(func.evals);
    wrapIterationSpaces(func.evals);
    reachabilityAnalysis(func.evals);
    setActualTargets(func.evals);
  }
};

#endif // FORTRAN_LOWER_BRIDGE_CFG_BUILDER_H_
