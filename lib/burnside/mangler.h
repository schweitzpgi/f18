//===-- lib/burnside/mangler.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_BURNSIDE_MANGLER_H_
#define FORTRAN_BURNSIDE_MANGLER_H_

#include <string>

namespace fir {
struct NameUniquer;
}

namespace llvm {
class StringRef;
}

namespace Fortran {
namespace common {
template<typename> class Reference;
}

namespace semantics {
class Symbol;
}

namespace burnside {
using SymbolRef = common::Reference<const semantics::Symbol>;

namespace mangle {

/// Convert a front-end Symbol to an internal name
std::string mangleName(fir::NameUniquer &uniquer, const SymbolRef symbol);

std::string demangleName(llvm::StringRef name);

}  // mangle
}  // burnside
}  // Fortran

#endif  // FORTRAN_BURNSIDE_MANGLER_H_
