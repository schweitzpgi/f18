//===-- lib/lower/io.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_IO_H_
#define FORTRAN_LOWER_IO_H_

namespace mlir {
class OpBuilder;
class Location;
class ValueRange;
} // namespace mlir

namespace Fortran {

namespace parser {
struct BackspaceStmt;
struct CloseStmt;
struct EndfileStmt;
struct FlushStmt;
struct InquireStmt;
struct OpenStmt;
struct PrintStmt;
struct ReadStmt;
struct RewindStmt;
struct WriteStmt;
} // namespace parser

/// Experimental IO lowering to FIR + runtime. The Runtime design is under
/// design.
/// FIXME This interface is also not final. Should it be based on parser::..
/// nodes and lower expressions as needed or should it get every expression
/// already lowered as mlir::Value? (currently second options, not sure it
/// will provide enough information for complex IO statements).
namespace lower {

class AbstractConverter;
class BridgeImpl;

/// Generate IO call(s) for BACKSPACE
void genBackspaceStatement(AbstractConverter &, const parser::BackspaceStmt &);

/// Generate IO call(s) for CLOSE
void genCloseStatement(AbstractConverter &, const parser::CloseStmt &);

/// Generate IO call(s) for ENDFILE
void genEndfileStatement(AbstractConverter &, const parser::EndfileStmt &);

/// Generate IO call(s) for FLUSH
void genFlushStatement(AbstractConverter &, const parser::FlushStmt &);

/// Generate IO call(s) for INQUIRE
void genInquireStatement(AbstractConverter &, const parser::InquireStmt &);

/// Generate IO call(s) for OPEN
void genOpenStatement(AbstractConverter &, const parser::OpenStmt &);

/// Generate IO call(s) for PRINT
void genPrintStatement(AbstractConverter &, const parser::PrintStmt &);

/// Generate IO call(s) for READ
void genReadStatement(AbstractConverter &, const parser::ReadStmt &);

/// Generate IO call(s) for REWIND
void genRewindStatement(AbstractConverter &, const parser::RewindStmt &);

/// Generate IO call(s) for WRITE
void genWriteStatement(AbstractConverter &, const parser::WriteStmt &);

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_IO_H_
