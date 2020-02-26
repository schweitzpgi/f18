//===-- Mangler.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/lower/Mangler.h"
#include "flang/common/reference.h"
#include "flang/lower/Utils.h"
#include "flang/optimizer/Support/InternalNames.h"
#include "flang/semantics/tools.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"

namespace {

// recursively build the vector of module scopes
void moduleNames(const Fortran::semantics::Scope *scope,
                 llvm::SmallVector<llvm::StringRef, 2> &result) {
  if (scope->kind() == Fortran::semantics::Scope::Kind::Global) {
    return;
  }
  moduleNames(&scope->parent(), result);
  if (scope->kind() == Fortran::semantics::Scope::Kind::Module)
    if (auto *symbol = scope->symbol())
      result.emplace_back(toStringRef(symbol->name()));
}

llvm::SmallVector<llvm::StringRef, 2>
moduleNames(const Fortran::semantics::Scope *scope) {
  llvm::SmallVector<llvm::StringRef, 2> result;
  moduleNames(scope, result);
  return result;
}

llvm::Optional<llvm::StringRef>
hostName(const Fortran::semantics::Scope *scope) {
  if (scope->kind() == Fortran::semantics::Scope::Kind::Subprogram) {
    auto &parent = scope->parent();
    if (parent.kind() == Fortran::semantics::Scope::Kind::Subprogram)
      if (auto *symbol = parent.symbol()) {
        return {toStringRef(symbol->name())};
      }
  }
  return {};
}

} // namespace

// Mangle the name of `symbol` to make it unique within FIR's symbol table using
// the FIR name mangler, `mangler`
std::string
Fortran::lower::mangle::mangleName(fir::NameUniquer &uniquer,
                                   const Fortran::semantics::SymbolRef symbol) {
  return std::visit(
      Fortran::common::visitors{
          [&](const Fortran::semantics::MainProgramDetails &) {
            return uniquer.doProgramEntry().str();
          },
          [&](const Fortran::semantics::SubprogramDetails &) {
            auto &cb{symbol->name()};
            auto modNames{moduleNames(symbol->scope())};
            return uniquer.doProcedure(modNames, hostName(symbol->scope()),
                                       toStringRef(cb));
          },
          [&](const Fortran::semantics::ProcEntityDetails &) {
            auto &cb{symbol->name()};
            auto modNames{moduleNames(symbol->scope())};
            return uniquer.doProcedure(modNames, hostName(symbol->scope()),
                                       toStringRef(cb));
          },
          [](const auto &) -> std::string {
            assert(false);
            return {};
          },
      },
      symbol->details());
}

std::string Fortran::lower::mangle::demangleName(llvm::StringRef name) {
  auto result{fir::NameUniquer::deconstruct(name)};
  return result.second.name;
}
