//===-- Lower/PFTBuilder.h -- PFT builder -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// PFT (Pre-FIR Tree) interface.
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_PFTBUILDER_H
#define FORTRAN_LOWER_PFTBUILDER_H

#include "flang/Common/template.h"
#include "flang/Parser/parse-tree.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

namespace mlir {
class Block;
}

namespace Fortran::lower {
namespace pft {

struct Evaluation;
struct Program;
struct ModuleLikeUnit;
struct FunctionLikeUnit;

// Using std::list for a list of Evaluations preserves the option to do
// O(1) time insertions anywhere.
using EvaluationList = std::list<Evaluation>;

struct ParentVariant {
  template <typename A>
  ParentVariant(A &parentVariant) : p{&parentVariant} {}
  const std::variant<Program *, ModuleLikeUnit *, FunctionLikeUnit *,
                     Evaluation *>
      p;
};

/// Classify the parse-tree nodes from ExecutablePartConstruct

using ActionStmts = std::tuple<
    parser::AllocateStmt, parser::AssignmentStmt, parser::BackspaceStmt,
    parser::CallStmt, parser::CloseStmt, parser::ContinueStmt,
    parser::CycleStmt, parser::DeallocateStmt, parser::EndfileStmt,
    parser::EventPostStmt, parser::EventWaitStmt, parser::ExitStmt,
    parser::FailImageStmt, parser::FlushStmt, parser::FormTeamStmt,
    parser::GotoStmt, parser::IfStmt, parser::InquireStmt, parser::LockStmt,
    parser::NullifyStmt, parser::OpenStmt, parser::PointerAssignmentStmt,
    parser::PrintStmt, parser::ReadStmt, parser::ReturnStmt, parser::RewindStmt,
    parser::StopStmt, parser::SyncAllStmt, parser::SyncImagesStmt,
    parser::SyncMemoryStmt, parser::SyncTeamStmt, parser::UnlockStmt,
    parser::WaitStmt, parser::WhereStmt, parser::WriteStmt,
    parser::ComputedGotoStmt, parser::ForallStmt, parser::ArithmeticIfStmt,
    parser::AssignStmt, parser::AssignedGotoStmt, parser::PauseStmt>;

using OtherStmts = std::tuple<parser::FormatStmt, parser::EntryStmt,
                              parser::DataStmt, parser::NamelistStmt>;

using ConstructStmts = std::tuple<
    parser::AssociateStmt, parser::EndAssociateStmt, parser::BlockStmt,
    parser::EndBlockStmt, parser::SelectCaseStmt, parser::CaseStmt,
    parser::EndSelectStmt, parser::ChangeTeamStmt, parser::EndChangeTeamStmt,
    parser::CriticalStmt, parser::EndCriticalStmt, parser::NonLabelDoStmt,
    parser::EndDoStmt, parser::IfThenStmt, parser::ElseIfStmt, parser::ElseStmt,
    parser::EndIfStmt, parser::SelectRankStmt, parser::SelectRankCaseStmt,
    parser::SelectTypeStmt, parser::TypeGuardStmt, parser::WhereConstructStmt,
    parser::MaskedElsewhereStmt, parser::ElsewhereStmt, parser::EndWhereStmt,
    parser::ForallConstructStmt, parser::EndForallStmt>;

using Constructs =
    std::tuple<parser::AssociateConstruct, parser::BlockConstruct,
               parser::CaseConstruct, parser::ChangeTeamConstruct,
               parser::CriticalConstruct, parser::DoConstruct,
               parser::IfConstruct, parser::SelectRankConstruct,
               parser::SelectTypeConstruct, parser::WhereConstruct,
               parser::ForallConstruct, parser::CompilerDirective,
               parser::OpenMPConstruct, parser::OmpEndLoopDirective>;

template <typename A>
static constexpr bool isActionStmt{common::HasMember<A, ActionStmts>};

template <typename A>
static constexpr bool isOtherStmt{common::HasMember<A, OtherStmts>};

template <typename A>
static constexpr bool isConstructStmt{common::HasMember<A, ConstructStmts>};

template <typename A>
static constexpr bool isConstruct{common::HasMember<A, Constructs>};

template <typename A>
static constexpr bool isIntermediateConstructStmt{common::HasMember<
    A, std::tuple<parser::CaseStmt, parser::ElseIfStmt, parser::ElseStmt,
                  parser::SelectRankCaseStmt, parser::TypeGuardStmt>>};

template <typename A>
static constexpr bool isNopConstructStmt{common::HasMember<
    A, std::tuple<parser::EndAssociateStmt, parser::CaseStmt,
                  parser::EndSelectStmt, parser::ElseIfStmt, parser::ElseStmt,
                  parser::EndIfStmt, parser::SelectRankCaseStmt,
                  parser::TypeGuardStmt>>};

template <typename A>
static constexpr bool isFunctionLike{common::HasMember<
    A, std::tuple<parser::MainProgram, parser::FunctionSubprogram,
                  parser::SubroutineSubprogram,
                  parser::SeparateModuleSubprogram>>};

using LabelSet = llvm::SmallSet<parser::Label, 5>;
using SymbolLabelMap = std::map<semantics::Symbol *, LabelSet>;

/// Function-like units contain lists of evaluations.  These can be simple
/// statements or constructs, where a construct contains its own evaluations.
struct Evaluation {
  using EvaluationTuple = common::CombineTuples<ActionStmts, OtherStmts,
                                                ConstructStmts, Constructs>;

  /// Hide non-nullable pointers to the parse-tree node.
  /// Build type std::variant<const A* const, const B* const, ...>
  /// from EvaluationTuple type (std::tuple<A, B, ...>).
  template <typename A>
  using MakeRefType = const A *const;
  using EvaluationVariant = common::CombineVariants<
      common::MapTemplate<MakeRefType, EvaluationTuple>>;
  template <typename A>
  constexpr auto visit(A visitor) const {
    return std::visit(common::visitors{
                          [&](const auto *p) { return visitor(*p); },
                          [&](auto &r) { return visitor(r); },
                      },
                      u);
  }
  template <typename A>
  constexpr const A *getIf() const {
    auto *ptr = std::get_if<MakeRefType<A>>(&u);
    return ptr ? *ptr : nullptr;
  }
  template <typename A>
  constexpr bool isA() const {
    return std::holds_alternative<MakeRefType<A>>(u);
  }

  Evaluation() = delete;
  Evaluation(const Evaluation &) = delete;
  Evaluation(Evaluation &&) = default;

  /// General ctor
  template <typename A>
  Evaluation(const A &a, const ParentVariant &parentVariant,
             const parser::CharBlock &position,
             const std::optional<parser::Label> &label)
      : u{&a}, parentVariant{parentVariant}, position{position}, label{label} {}

  /// Construct ctor
  template <typename A>
  Evaluation(const A &a, const ParentVariant &parentVariant)
      : u{&a}, parentVariant{parentVariant} {
    static_assert(pft::isConstruct<A>, "must be a construct");
  }

  /// Evaluation classification predicates.
  constexpr bool isActionStmt() const {
    return visit(common::visitors{
        [](auto &r) { return pft::isActionStmt<std::decay_t<decltype(r)>>; }});
  }
  constexpr bool isOtherStmt() const {
    return visit(common::visitors{
        [](auto &r) { return pft::isOtherStmt<std::decay_t<decltype(r)>>; }});
  }
  constexpr bool isConstructStmt() const {
    return visit(common::visitors{[](auto &r) {
      return pft::isConstructStmt<std::decay_t<decltype(r)>>;
    }});
  }
  constexpr bool isConstruct() const {
    return visit(common::visitors{
        [](auto &r) { return pft::isConstruct<std::decay_t<decltype(r)>>; }});
  }
  constexpr bool isIntermediateConstructStmt() const {
    return visit(common::visitors{[](auto &r) {
      return pft::isIntermediateConstructStmt<std::decay_t<decltype(r)>>;
    }});
  }
  constexpr bool isNopConstructStmt() const {
    return visit(common::visitors{[](auto &r) {
      return pft::isNopConstructStmt<std::decay_t<decltype(r)>>;
    }});
  }

  /// Return FunctionLikeUnit to which this evaluation
  /// belongs. Nullptr if it does not belong to such unit.
  FunctionLikeUnit *getOwningProcedure() const;

  bool lowerAsStructured() const;
  bool lowerAsUnstructured() const;

  // FIR generation looks primarily at PFT statement (leaf) nodes.  So members
  // such as lexicalSuccessor and the various block fields are only applicable
  // to statement nodes.  One exception is that an internal construct node is
  // a convenient place for a constructExit link that applies to exits from any
  // statement within the construct.  The controlSuccessor member is used for
  // nonlexical successors, such as linking to a GOTO target.  For multiway
  // branches, controlSuccessor is set to one of the targets (might as well be
  // the first target).  Successor and exit links always target statements.
  //
  // An unstructured construct is one that contains some form of goto.  This
  // is indicated by the isUnstructured member flag, which may be set on a
  // statement and propagated to enclosing constructs.  This distinction allows
  // a structured IF or DO statement to be materialized with custom structured
  // FIR operations.  An unstructured statement is materialized as mlir
  // operation sequences that include explicit branches.
  //
  // There are two mlir::Block members.  The block member is set for statements
  // that begin a new block.  If a statement may have more than one associated
  // block, this member must be the block that would be the target of a branch
  // to the statement.  The prime example of a statement that may have multiple
  // associated blocks is NonLabelDoStmt, which may have a loop preheader block
  // for loop initialization code, and always has a header block that is the
  // target of the loop back edge.  If the NonLabelDoStmt is a concurrent loop,
  // there may be an arbitrary number of nested preheader, header, and mask
  // blocks.  Any such additional blocks in the localBlocks member are local
  // to a construct and cannot be the target of an unstructured branch.  For
  // NonLabelDoStmt, the block member designates the preheader block, which may
  // be absent if loop initialization code may be appended to a predecessor
  // block.  The primary loop header block is localBlocks[0], with additional
  // DO CONCURRENT blocks at localBlocks[1], etc.
  //
  // The printIndex member is only set for statements.  It is used for dumps
  // and does not affect FIR generation.  It may also be helpful for debugging.

  EvaluationVariant u;
  ParentVariant parentVariant;
  parser::CharBlock position{};
  std::optional<parser::Label> label{};
  std::unique_ptr<EvaluationList> evaluationList; // nested evaluations
  Evaluation *parentConstruct{nullptr};  // set for nodes below the top level
  Evaluation *lexicalSuccessor{nullptr}; // set for ActionStmt, ConstructStmt
  Evaluation *controlSuccessor{nullptr}; // set for some statements
  Evaluation *constructExit{nullptr};    // set for constructs
  bool isNewBlock{false};                // evaluation begins a new basic block
  bool isUnstructured{false};        // evaluation has unstructured control flow
  bool skip{false};                  // evaluation has been processed in advance
  class mlir::Block *block{nullptr}; // isNewBlock block
  llvm::SmallVector<mlir::Block *, 1> localBlocks{}; // construct local blocks
  int printIndex{0}; // (ActionStmt, ConstructStmt) evaluation index for dumps
};

/// A program is a list of program units.
/// These units can be function like, module like, or block data.
struct ProgramUnit {
  template <typename A>
  ProgramUnit(const A &ptr, const ParentVariant &parentVariant)
      : p{&ptr}, parentVariant{parentVariant} {}
  ProgramUnit(ProgramUnit &&) = default;
  ProgramUnit(const ProgramUnit &) = delete;

  const std::variant<
      const parser::MainProgram *, const parser::FunctionSubprogram *,
      const parser::SubroutineSubprogram *, const parser::Module *,
      const parser::Submodule *, const parser::SeparateModuleSubprogram *,
      const parser::BlockData *>
      p;
  ParentVariant parentVariant;
};

/// Function-like units may contain evaluations (executable statements) and
/// nested function-like units (internal procedures and function statements).
struct FunctionLikeUnit : public ProgramUnit {
  // wrapper statements for function-like syntactic structures
  using FunctionStatement =
      std::variant<const parser::Statement<parser::ProgramStmt> *,
                   const parser::Statement<parser::EndProgramStmt> *,
                   const parser::Statement<parser::FunctionStmt> *,
                   const parser::Statement<parser::EndFunctionStmt> *,
                   const parser::Statement<parser::SubroutineStmt> *,
                   const parser::Statement<parser::EndSubroutineStmt> *,
                   const parser::Statement<parser::MpSubprogramStmt> *,
                   const parser::Statement<parser::EndMpSubprogramStmt> *>;

  FunctionLikeUnit(const parser::MainProgram &f,
                   const ParentVariant &parentVariant);
  FunctionLikeUnit(const parser::FunctionSubprogram &f,
                   const ParentVariant &parentVariant);
  FunctionLikeUnit(const parser::SubroutineSubprogram &f,
                   const ParentVariant &parentVariant);
  FunctionLikeUnit(const parser::SeparateModuleSubprogram &f,
                   const ParentVariant &parentVariant);
  FunctionLikeUnit(FunctionLikeUnit &&) = default;
  FunctionLikeUnit(const FunctionLikeUnit &) = delete;

  bool isMainProgram() {
    return std::holds_alternative<
        const parser::Statement<parser::EndProgramStmt> *>(endStmt);
  }

  const parser::FunctionStmt *getFunction() {
    return getA<parser::FunctionStmt>();
  }
  const parser::SubroutineStmt *getSubroutine() {
    return getA<parser::SubroutineStmt>();
  }
  const parser::MpSubprogramStmt *getMpSubprogram() {
    return getA<parser::MpSubprogramStmt>();
  }

  bool isMainProgram() const {
    return !beginStmt ||
           std::holds_alternative<
               const parser::Statement<parser::ProgramStmt> *>(*beginStmt);
  }

  /// Returns reference to the subprogram symbol of this FunctionLikeUnit.
  /// Dies if the FunctionLikeUnit is not a subprogram.
  const semantics::Symbol &getSubprogramSymbol() const {
    assert(symbol && "not inside a procedure");
    return *symbol;
  }

  /// Anonymous programs do not have a begin statement
  std::optional<FunctionStatement> beginStmt;
  FunctionStatement endStmt;
  EvaluationList evaluationList;
  llvm::DenseMap<parser::Label, Evaluation *> labelEvaluationMap;
  SymbolLabelMap assignSymbolLabelMap;
  std::list<FunctionLikeUnit> nestedFunctions;
  /// Symbol associated to this FunctionLikeUnit.
  /// Null if the FunctionLikeUnit is an anonymous program.
  /// The symbol has MainProgramDetails for named programs, otherwise it has
  /// SubprogramDetails.
  const semantics::Symbol *symbol{nullptr};

private:
  template <typename A>
  const A *getA() {
    if (beginStmt) {
      if (auto p =
              std::get_if<const parser::Statement<A> *>(&beginStmt.value()))
        return &(*p)->statement;
    }
    return nullptr;
  }
};

/// Module-like units contain a list of function-like units.
struct ModuleLikeUnit : public ProgramUnit {
  // wrapper statements for module-like syntactic structures
  using ModuleStatement =
      std::variant<const parser::Statement<parser::ModuleStmt> *,
                   const parser::Statement<parser::EndModuleStmt> *,
                   const parser::Statement<parser::SubmoduleStmt> *,
                   const parser::Statement<parser::EndSubmoduleStmt> *>;

  ModuleLikeUnit(const parser::Module &m, const ParentVariant &parentVariant);
  ModuleLikeUnit(const parser::Submodule &m,
                 const ParentVariant &parentVariant);
  ~ModuleLikeUnit() = default;
  ModuleLikeUnit(ModuleLikeUnit &&) = default;
  ModuleLikeUnit(const ModuleLikeUnit &) = delete;

  ModuleStatement beginStmt;
  ModuleStatement endStmt;
  std::list<FunctionLikeUnit> nestedFunctions;
};

struct BlockDataUnit : public ProgramUnit {
  BlockDataUnit(const parser::BlockData &bd,
                const ParentVariant &parentVariant);
  BlockDataUnit(BlockDataUnit &&) = default;
  BlockDataUnit(const BlockDataUnit &) = delete;
};

/// A Program is the top-level root of the PFT.
struct Program {
  using Units = std::variant<FunctionLikeUnit, ModuleLikeUnit, BlockDataUnit>;

  Program() = default;
  Program(Program &&) = default;
  Program(const Program &) = delete;

  std::list<Units> &getUnits() { return units; }

  /// LLVM dump method on a Program.
  void dump();

private:
  std::list<Units> units;
};

} // namespace pft

/// Create a PFT (Pre-FIR Tree) from the parse tree.
///
/// A PFT is a light weight tree over the parse tree that is used to create FIR.
/// The PFT captures pointers back into the parse tree, so the parse tree must
/// not be changed between the construction of the PFT and its last use.  The
/// PFT captures a structured view of a program.  A program is a list of units.
/// A function like unit contains a list of evaluations.  An evaluation is
/// either a statement, or a construct with a nested list of evaluations.
std::unique_ptr<pft::Program> createPFT(const parser::Program &root);

/// Dumper for displaying a PFT.
void dumpPFT(llvm::raw_ostream &outputStream, pft::Program &pft);

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_PFTBUILDER_H
