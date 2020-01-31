//===-- lib/lower/cfg-builder.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_CFG_BUILDER_H_
#define FORTRAN_LOWER_CFG_BUILDER_H_

/// Traverse the PFT and complete the CFG by drawing the arcs, pruning unused
/// potential targets, making implied jumps explicit, etc.
class CfgBuilder {

  PFT::Evaluation *getEvalByLabel(const Pa::Label &label) {
    auto iter = labels.find(label);
    if (iter != labels.end()) {
      return iter->second;
    }
    return nullptr;
  }

  /// Collect all the potential targets and initialize them to unreferenced
  void resetPotentialTargets(std::list<PFT::Evaluation> &evals) {
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
  void cacheAssigns(std::list<PFT::Evaluation> &evals) {
    for (auto &e : evals) {
      std::visit(Co::visitors{
                     [&](const Pa::AssignStmt *stmt) {
                       auto *trg = getEvalByLabel(std::get<Pa::Label>(stmt->t));
                       auto *sym = std::get<Pa::Name>(stmt->t).symbol;
                       assert(sym);
                       auto jter = assignedGotoMap.find(sym);
                       if (jter == assignedGotoMap.end()) {
                         std::list<PFT::Evaluation *> lst = {trg};
                         assignedGotoMap.try_emplace(sym, lst);
                       } else {
                         jter->second.emplace_back(trg);
                       }
                     },
                     [](auto) { /* do nothing */ },
                 },
                 e.u);
      if (e.subs) {
        cacheAssigns(*e.subs);
      }
    }
  }

  void deannotate(std::list<PFT::Evaluation> &evals) {
    for (auto &e : evals) {
      e.cfg = PFT::CFGAnnotation::None;
      if (e.subs) {
        deannotate(*e.subs);
      }
    }
  }

  bool structuredCheck(std::list<PFT::Evaluation> &evals) {
    for (auto &e : evals) {
      if (auto **s = std::get_if<const Pa::DoConstruct *>(&e.u)) {
        return (*s)->IsDoWhile() ? false : structuredCheck(*e.subs);
      }
      if (std::holds_alternative<const Pa::IfConstruct *>(e.u)) {
        return structuredCheck(*e.subs);
      }
      if (e.subs) {
        return false;
      }
      switch (e.cfg) {
      case PFT::CFGAnnotation::None:
        break;
      case PFT::CFGAnnotation::CondGoto:
        break;
      case PFT::CFGAnnotation::Iterative:
        break;
      case PFT::CFGAnnotation::FirStructuredOp:
        break;
      case PFT::CFGAnnotation::IndGoto:
        return false;
      case PFT::CFGAnnotation::IoSwitch:
        return false;
      case PFT::CFGAnnotation::Switch:
        return false;
      case PFT::CFGAnnotation::Return:
        return false;
      case PFT::CFGAnnotation::Terminate:
        return false;
      case PFT::CFGAnnotation::Goto:
        if (!std::holds_alternative<const Pa::EndDoStmt *>(e.u)) {
          return false;
        }
        break;
      }
    }
    return true;
  }

  void wrapIterationSpaces(std::list<PFT::Evaluation> &evals) {
    for (auto &e : evals) {
      if (std::holds_alternative<const Pa::DoConstruct *>(e.u))
        if (structuredCheck(*e.subs)) {
          deannotate(*e.subs);
          e.cfg = PFT::CFGAnnotation::FirStructuredOp;
          continue;
        }
      if (std::holds_alternative<const Pa::IfConstruct *>(e.u))
        if (structuredCheck(*e.subs)) {
          deannotate(*e.subs);
          e.cfg = PFT::CFGAnnotation::FirStructuredOp;
          continue;
        }
      // FIXME: ForallConstruct? WhereConstruct?
      if (e.subs) {
        wrapIterationSpaces(*e.subs);
      }
    }
  }

  /// Add source->sink edge to CFG map
  void addSourceToSink(PFT::Evaluation *src, PFT::Evaluation *snk) {
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

  void addSourceToSink(PFT::Evaluation *src, const Pa::Label &label) {
    auto iter = labels.find(label);
    assert(iter != labels.end());
    addSourceToSink(src, iter->second);
  }

  /// Find the next ELSE IF, ELSE or END IF statement in the list
  template <typename A>
  A nextFalseTarget(A iter, const A &endi) {
    for (; iter != endi; ++iter)
      if (std::visit(Co::visitors{
                         [&](const Pa::ElseIfStmt *) { return true; },
                         [&](const Pa::ElseStmt *) { return true; },
                         [&](const Pa::EndIfStmt *) { return true; },
                         [](auto) { return false; },
                     },
                     iter->u)) {
        break;
      }
    return iter;
  }

  /// Add branches for this IF block like construct.
  /// Branch to the "true block", the "false block", and from the end of the
  /// true block to the end of the construct.
  template <typename A>
  void doNextIfBlock(std::list<PFT::Evaluation> &evals, PFT::Evaluation &e,
                     const A &iter, const A &endif) {
    A i{iter};
    A j{nextFalseTarget(++i, endif)};
    auto *cstr = std::get<PFT::Evaluation *>(e.parent);
    PFT::CGJump jump{&*endif};
    A k{evals.insert(j, PFT::Evaluation{std::move(jump), j->parent})};
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
  void reachabilityAnalysis(std::list<PFT::Evaluation> &evals) {
    for (auto iter = evals.begin(); iter != evals.end(); ++iter) {
      auto &e = *iter;
      switch (e.cfg) {
      case PFT::CFGAnnotation::None:
        // do nothing - does not impart control flow
        break;
      case PFT::CFGAnnotation::Goto:
        std::visit(
            Co::visitors{
                [&](const Pa::CycleStmt *) {
                  // FIXME: deal with construct name
                  auto *cstr = std::get<PFT::Evaluation *>(e.parent);
                  addSourceToSink(&e, &cstr->subs->front());
                },
                [&](const Pa::ExitStmt *) {
                  // FIXME: deal with construct name
                  auto *cstr = std::get<PFT::Evaluation *>(e.parent);
                  addSourceToSink(&e, &cstr->subs->back());
                },
                [&](const Pa::GotoStmt *stmt) { addSourceToSink(&e, stmt->v); },
                [&](const Pa::EndDoStmt *) {
                  // the END DO is the loop exit landing pad
                  // insert a JUMP as the backedge right before the END DO
                  auto *cstr = std::get<PFT::Evaluation *>(e.parent);
                  PFT::CGJump jump{&cstr->subs->front()};
                  PFT::Evaluation jumpEval{std::move(jump), iter->parent};
                  evals.insert(iter, std::move(jumpEval));
                  addSourceToSink(&e, &cstr->subs->front());
                },
                [&](const PFT::CGJump &jump) {
                  addSourceToSink(&e, jump.target);
                },
                [](auto) { assert(false && "unhandled GOTO case"); },
            },
            e.u);
        break;
      case PFT::CFGAnnotation::CondGoto:
        std::visit(Co::visitors{
                       [&](const Pa::IfStmt *) {
                         // check if these are marked; they must targets here
                         auto i{iter};
                         addSourceToSink(&e, &*(++i));
                         addSourceToSink(&e, &*(++i));
                       },
                       [&](const Pa::IfThenStmt *) {
                         doNextIfBlock(evals, e, iter, evals.end());
                       },
                       [&](const Pa::ElseIfStmt *) {
                         doNextIfBlock(evals, e, iter, evals.end());
                       },
                       [](const Pa::WhereConstructStmt *) { TODO(); },
                       [](const Pa::MaskedElsewhereStmt *) { TODO(); },
                       [](auto) { assert(false && "unhandled CGOTO case"); },
                   },
                   e.u);
        break;
      case PFT::CFGAnnotation::IndGoto:
        std::visit(Co::visitors{
                       [&](const Pa::AssignedGotoStmt *stmt) {
                         auto *sym = std::get<Pa::Name>(stmt->t).symbol;
                         if (assignedGotoMap.find(sym) != assignedGotoMap.end())
                           for (auto *x : assignedGotoMap[sym]) {
                             addSourceToSink(&e, x);
                           }
                         for (auto &l :
                              std::get<std::list<Pa::Label>>(stmt->t)) {
                           addSourceToSink(&e, l);
                         }
                       },
                       [](auto) { assert(false && "unhandled IGOTO case"); },
                   },
                   e.u);
        break;
      case PFT::CFGAnnotation::IoSwitch:
        std::visit(
            Co::visitors{
                [](const Pa::BackspaceStmt *) { TODO(); },
                [](const Pa::CloseStmt *) { TODO(); },
                [](const Pa::EndfileStmt *) { TODO(); },
                [](const Pa::FlushStmt *) { TODO(); },
                [](const Pa::InquireStmt *) { TODO(); },
                [](const Pa::OpenStmt *) { TODO(); },
                [](const Pa::ReadStmt *) { TODO(); },
                [](const Pa::RewindStmt *) { TODO(); },
                [](const Pa::WaitStmt *) { TODO(); },
                [](const Pa::WriteStmt *) { TODO(); },
                [](auto) { assert(false && "unhandled IO switch case"); },
            },
            e.u);
        break;
      case PFT::CFGAnnotation::Switch:
        std::visit(Co::visitors{
                       [](const Pa::CallStmt *) { TODO(); },
                       [](const Pa::ArithmeticIfStmt *) { TODO(); },
                       [](const Pa::ComputedGotoStmt *) { TODO(); },
                       [](const Pa::SelectCaseStmt *) { TODO(); },
                       [](const Pa::SelectRankStmt *) { TODO(); },
                       [](const Pa::SelectTypeStmt *) { TODO(); },
                       [](auto) { assert(false && "unhandled switch case"); },
                   },
                   e.u);
        break;
      case PFT::CFGAnnotation::Iterative:
        std::visit(Co::visitors{
                       [](const Pa::NonLabelDoStmt *) { TODO(); },
                       [](const Pa::WhereStmt *) { TODO(); },
                       [](const Pa::ForallStmt *) { TODO(); },
                       [](const Pa::WhereConstruct *) { TODO(); },
                       [](const Pa::ForallConstructStmt *) { TODO(); },
                       [](auto) { assert(false && "unhandled loop case"); },
                   },
                   e.u);
        break;
      case PFT::CFGAnnotation::FirStructuredOp:
        // do not visit the subs
        continue;
      case PFT::CFGAnnotation::Return:
        // do nothing - exits the function
        break;
      case PFT::CFGAnnotation::Terminate:
        // do nothing - exits the function
        break;
      }
      if (e.subs) {
        reachabilityAnalysis(*e.subs);
      }
    }
  }

  void setActualTargets(std::list<PFT::Evaluation> &) {
    for (auto &lst1 : cfgEdgeSetPool)
      for (auto *e : lst1) {
        e->isTarget = true;
      }
  }

  CFGMapType &cfgMap;
  std::list<CFGSinkListType> &cfgEdgeSetPool;

  L::DenseMap<Pa::Label, PFT::Evaluation *> labels;
  std::map<Se::Symbol *, std::list<PFT::Evaluation *>> assignedGotoMap;

public:
  CfgBuilder(CFGMapType &m, std::list<CFGSinkListType> &p)
      : cfgMap{m}, cfgEdgeSetPool{p} {}

  void run(PFT::FunctionLikeUnit &func) {
    resetPotentialTargets(func.evals);
    cacheAssigns(func.evals);
    wrapIterationSpaces(func.evals);
    reachabilityAnalysis(func.evals);
    setActualTargets(func.evals);
  }
};

#endif // FORTRAN_LOWER_BRIDGE_CFG_BUILDER_H_
