//===-- NSAliases.h -- namespace aliases ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Define a number of namespace aliases so that a large percentage of the code
// isn't dominated by repetitive namespace tags.  This improves clarity and
// comprehension.
//
// Short names for Fortran subnamespaces tend to use 2 letter abbreviations.
// Other namespaces use 1 letter.
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_NSALIASES_H
#define FORTRAN_LOWER_NSALIASES_H

namespace llvm {}
namespace mlir {}
namespace Fortran {
namespace common {}
namespace evaluate {}
namespace lower {
namespace pft {}
} // namespace lower
namespace lower::mangle {}
namespace parser {}
namespace runtime {
namespace io {}
} // namespace runtime
namespace semantics {}
} // namespace Fortran

namespace Br = Fortran::lower;
namespace Co = Fortran::common;
namespace Ev = Fortran::evaluate;
namespace Io = Fortran::runtime::io;
namespace L = llvm;
namespace Ma = Fortran::lower::mangle;
namespace M = mlir;
namespace Pa = Fortran::parser;
namespace PFT = Fortran::lower::pft;
namespace Rt = Fortran::runtime;
namespace Se = Fortran::semantics;

#endif // FORTRAN_LOWER_NSALIASES_H
