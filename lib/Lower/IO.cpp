//===-- IO.cpp -- I/O statement lowering ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/IO.h"
#include "../../runtime/io-api.h"
#include "RTBuilder.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/FIRBuilder.h"
#include "flang/Lower/Runtime.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/tools.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#define TODO() llvm_unreachable("not yet implemented")

using namespace Fortran::runtime::io;

#define NAMIFY_HELPER(X) #X
#define NAMIFY(X) NAMIFY_HELPER(IONAME(X))
#define mkIOKey(X) mkKey(IONAME(X))

namespace Fortran::lower {
/// Static table of IO runtime calls
///
/// This logical map contains the name and type builder function for each IO
/// runtime function listed in the tuple. This table is fully constructed at
/// compile-time. Use the `mkIOKey` macro to access the table.
static constexpr std::tuple<
    mkIOKey(BeginInternalArrayListOutput), mkIOKey(BeginInternalArrayListInput),
    mkIOKey(BeginInternalArrayFormattedOutput),
    mkIOKey(BeginInternalArrayFormattedInput), mkIOKey(BeginInternalListOutput),
    mkIOKey(BeginInternalListInput), mkIOKey(BeginInternalFormattedOutput),
    mkIOKey(BeginInternalFormattedInput), mkIOKey(BeginInternalNamelistOutput),
    mkIOKey(BeginInternalNamelistInput), mkIOKey(BeginExternalListOutput),
    mkIOKey(BeginExternalListInput), mkIOKey(BeginExternalFormattedOutput),
    mkIOKey(BeginExternalFormattedInput), mkIOKey(BeginUnformattedOutput),
    mkIOKey(BeginUnformattedInput), mkIOKey(BeginExternalNamelistOutput),
    mkIOKey(BeginExternalNamelistInput), mkIOKey(BeginAsynchronousOutput),
    mkIOKey(BeginAsynchronousInput), mkIOKey(BeginWait), mkIOKey(BeginWaitAll),
    mkIOKey(BeginClose), mkIOKey(BeginFlush), mkIOKey(BeginBackspace),
    mkIOKey(BeginEndfile), mkIOKey(BeginRewind), mkIOKey(BeginOpenUnit),
    mkIOKey(BeginOpenNewUnit), mkIOKey(BeginInquireUnit),
    mkIOKey(BeginInquireFile), mkIOKey(BeginInquireIoLength),
    mkIOKey(EnableHandlers), mkIOKey(SetAdvance), mkIOKey(SetBlank),
    mkIOKey(SetDecimal), mkIOKey(SetDelim), mkIOKey(SetPad), mkIOKey(SetPos),
    mkIOKey(SetRec), mkIOKey(SetRound), mkIOKey(SetSign),
    mkIOKey(OutputDescriptor), mkIOKey(InputDescriptor),
    mkIOKey(OutputUnformattedBlock), mkIOKey(InputUnformattedBlock),
    mkIOKey(OutputInteger64), mkIOKey(InputInteger), mkIOKey(OutputReal32),
    mkIOKey(InputReal32), mkIOKey(OutputReal64), mkIOKey(InputReal64),
    mkIOKey(OutputComplex64), mkIOKey(OutputComplex32), mkIOKey(OutputAscii),
    mkIOKey(InputAscii), mkIOKey(OutputLogical), mkIOKey(InputLogical),
    mkIOKey(SetAccess), mkIOKey(SetAction), mkIOKey(SetAsynchronous),
    mkIOKey(SetEncoding), mkIOKey(SetForm), mkIOKey(SetPosition),
    mkIOKey(SetRecl), mkIOKey(SetStatus), mkIOKey(SetFile), mkIOKey(GetNewUnit),
    mkIOKey(GetSize), mkIOKey(GetIoLength), mkIOKey(GetIoMsg),
    mkIOKey(InquireCharacter), mkIOKey(InquireLogical),
    mkIOKey(InquirePendingId), mkIOKey(InquireInteger64),
    mkIOKey(EndIoStatement)>
    newIOTable;
} // namespace Fortran::lower

using namespace Fortran::lower;

/// Helper function to retrieve the name of the IO function given the key `A`
template <typename A>
static constexpr const char *getName() {
  return std::get<A>(newIOTable).name;
}

/// Helper function to retrieve the type model signature builder of the IO
/// function as defined by the key `A`
template <typename A>
static constexpr FuncTypeBuilderFunc getTypeModel() {
  return std::get<A>(newIOTable).getTypeModel();
}

inline bool isCharacterLiteral(mlir::Type argTy) {
  if (auto arrTy = argTy.dyn_cast<fir::SequenceType>())
    return arrTy.getEleTy().isa<fir::CharacterType>();
  return false;
}

inline int64_t getLength(mlir::Type argTy) {
  return argTy.cast<fir::SequenceType>().getShape()[0];
}

/// Get (or generate) the MLIR FuncOp for a given IO runtime function.
template <typename E>
static mlir::FuncOp getIORuntimeFunc(Fortran::lower::FirOpBuilder &builder) {
  auto name = getName<E>();
  auto func = builder.getNamedFunction(name);
  if (func)
    return func;
  auto funTy = getTypeModel<E>()(builder.getContext());
  func = builder.createFunction(name, funTy);
  func.setAttr("fir.runtime", builder.getUnitAttr());
  func.setAttr("fir.io", builder.getUnitAttr());
  return func;
}

/// Generate a call to end an IO statement
static mlir::Value genEndIO(Fortran::lower::FirOpBuilder &builder,
                            mlir::Location loc, mlir::Value cookie) {
  // Terminate IO
  auto endIOFunc = getIORuntimeFunc<mkIOKey(EndIoStatement)>(builder);
  llvm::SmallVector<mlir::Value, 1> endArgs{cookie};
  auto call = builder.create<mlir::CallOp>(loc, endIOFunc, endArgs);
  return call.getResult(0);
}

using FormatItems = std::optional<std::pair<mlir::Value, mlir::Value>>;

/// Translate a list of format-items into a set of call-backs that can be
/// emitted into the MLIR stream before each data item is processed
static FormatItems lowerFormat(Fortran::lower::AbstractConverter &converter,
                               const Fortran::parser::Format &format) {
  FormatItems formatItems;
  std::visit(Fortran::common::visitors{
                 [](const Fortran::parser::DefaultCharExpr
                        &) { /* string expression */ },
                 [](const Fortran::parser::Label &) { /* FORMAT statement */ },
                 [](const Fortran::parser::Star &) {},
             },
             format.u);
  return formatItems;
}

static mlir::FuncOp getOutputRuntimeFunc(Fortran::lower::FirOpBuilder &builder,
                                         mlir::Type type) {
  if (auto ty = type.dyn_cast<mlir::IntegerType>()) {
    if (ty.getWidth() == 1)
      return getIORuntimeFunc<mkIOKey(OutputLogical)>(builder);
    return getIORuntimeFunc<mkIOKey(OutputInteger64)>(builder);
  } else if (auto ty = type.dyn_cast<mlir::FloatType>()) {
    if (ty.getWidth() <= 32)
      return getIORuntimeFunc<mkIOKey(OutputReal32)>(builder);
    return getIORuntimeFunc<mkIOKey(OutputReal64)>(builder);
  } else if (auto ty = type.dyn_cast<fir::CplxType>()) {
    if (ty.getFKind() <= 4)
      return getIORuntimeFunc<mkIOKey(OutputComplex32)>(builder);
    return getIORuntimeFunc<mkIOKey(OutputComplex64)>(builder);
  } else if (auto ty = type.dyn_cast<fir::LogicalType>()) {
    return getIORuntimeFunc<mkIOKey(OutputLogical)>(builder);
  } else if (auto ty = type.dyn_cast<fir::BoxType>()) {
    return getIORuntimeFunc<mkIOKey(OutputDescriptor)>(builder);
  } else {
    return getIORuntimeFunc<mkIOKey(OutputAscii)>(builder);
  }
}

/// The I/O library interface requires that COMPLEX and CHARACTER typed values
/// be extracted and passed as separate values.
static llvm::SmallVector<mlir::Value, 4>
splitArguments(Fortran::lower::FirOpBuilder &builder, mlir::Location loc,
               mlir::Value arg) {
  mlir::Value zero;
  mlir::Value one;
  mlir::Type argTy = arg.getType();
  mlir::MLIRContext *context = argTy.getContext();
  const bool isComplex = argTy.isa<fir::CplxType>();
  const bool isCharLit = isCharacterLiteral(argTy);
  const bool isBoxChar = argTy.isa<fir::BoxCharType>();

  if (isComplex || isCharLit || isBoxChar) {
    // Only create these constants when needed and not every time
    zero = builder.create<mlir::ConstantOp>(loc, builder.getI64IntegerAttr(0));
    one = builder.create<mlir::ConstantOp>(loc, builder.getI64IntegerAttr(1));
  }
  if (isComplex) {
    auto eleTy =
        fir::RealType::get(context, argTy.cast<fir::CplxType>().getFKind());
    mlir::Value realPart =
        builder.create<fir::ExtractValueOp>(loc, eleTy, arg, zero);
    mlir::Value imaginaryPart =
        builder.create<fir::ExtractValueOp>(loc, eleTy, arg, one);
    return {realPart, imaginaryPart};
  }
  if (isBoxChar) {
    mlir::Type ptrTy =
        fir::ReferenceType::get(mlir::IntegerType::get(8, context));
    mlir::Type sizeTy = mlir::IntegerType::get(64, context);
    mlir::Value pointerPart =
        builder.create<fir::ExtractValueOp>(loc, ptrTy, arg, zero);
    mlir::Value sizePart =
        builder.create<fir::ExtractValueOp>(loc, sizeTy, arg, one);
    return {pointerPart, sizePart};
  }
  if (isCharLit) {
    mlir::Value variable = builder.create<fir::AllocaOp>(loc, argTy);
    builder.create<fir::StoreOp>(loc, arg, variable);
    mlir::Value sizePart = builder.create<mlir::ConstantOp>(
        loc, builder.getI64IntegerAttr(getLength(argTy)));
    return {variable, sizePart};
  }
  return {arg};
}

/// Generate a call to an Output I/O function.
/// The specific call is determined dynamically by the argument type.
static void genOutputRuntimeFunc(Fortran::lower::FirOpBuilder &builder,
                                 mlir::Location loc, mlir::Type argType,
                                 mlir::Value cookie,
                                 llvm::SmallVector<mlir::Value, 4> &operands) {
  int i = 1;
  llvm::SmallVector<mlir::Value, 4> actuals{cookie};
  auto outputFunc = getOutputRuntimeFunc(builder, argType);

  for (auto &op : operands)
    actuals.emplace_back(builder.create<fir::ConvertOp>(
        loc, outputFunc.getType().getInput(i++), op));
  builder.create<mlir::CallOp>(loc, outputFunc, actuals);
}

//===----------------------------------------------------------------------===//
// Default argument generation.
//===----------------------------------------------------------------------===//

static mlir::Value getDefaultFilename(Fortran::lower::FirOpBuilder &builder,
                                      mlir::Location loc, mlir::Type toType) {
  mlir::Value null =
      builder.create<mlir::ConstantOp>(loc, builder.getI64IntegerAttr(0));
  return builder.create<fir::ConvertOp>(loc, toType, null);
}

static mlir::Value getDefaultLineNo(Fortran::lower::FirOpBuilder &builder,
                                    mlir::Location loc, mlir::Type toType) {
  return builder.create<mlir::ConstantOp>(loc,
                                          builder.getIntegerAttr(toType, 0));
}

template <typename A>
static std::tuple<mlir::Value, mlir::Value, mlir::Value>
lowerStringLit(Fortran::lower::AbstractConverter &converter, mlir::Location loc,
               const A &syntax, mlir::Type ty0, mlir::Type ty1,
               mlir::Type ty2 = {}) {
  auto &builder = converter.getFirOpBuilder();
  auto *expr = Fortran::semantics::GetExpr(syntax);
  auto buffer = converter.genExprValue(expr, loc);
  auto buff = builder.create<fir::ConvertOp>(loc, ty0, buffer);
  mlir::Type ty = buffer.getType();
  auto lenVal = ty.cast<fir::SequenceType>().getShape()[0];
  auto len = builder.create<mlir::ConstantOp>(
      loc, builder.getIntegerAttr(ty1, lenVal));
  if (ty2) {
    mlir::Type eleTy = ty.cast<fir::SequenceType>().getEleTy();
    auto kindVal = eleTy.cast<fir::CharacterType>().getFKind();
    auto kind = builder.create<mlir::ConstantOp>(
        loc, builder.getIntegerAttr(ty2, kindVal));
    return {buff, len, kind};
  }
  return {buff, len, mlir::Value{}};
}

//===----------------------------------------------------------------------===//
// Handle I/O statement specifiers.
// These are threaded together for a single statement via the passed cookie.
//===----------------------------------------------------------------------===//

template <typename A>
mlir::Value genIOOption(Fortran::lower::AbstractConverter &converter,
                        mlir::Location loc, mlir::Value cookie, const A &spec);
template <>
mlir::Value genIOOption<Fortran::parser::FileNameExpr>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value cookie, const Fortran::parser::FileNameExpr &spec) {
  auto &builder = converter.getFirOpBuilder();
  auto ioFunc = getIORuntimeFunc<mkIOKey(SetFile)>(builder);
  mlir::FunctionType ioFuncTy = ioFunc.getType();
  auto tup = lowerStringLit(converter, loc, spec, ioFuncTy.getInput(1),
                            ioFuncTy.getInput(2), ioFuncTy.getInput(3));
  llvm::SmallVector<mlir::Value, 4> ioArgs{cookie, std::get<0>(tup),
                                           std::get<1>(tup), std::get<2>(tup)};
  return builder.create<mlir::CallOp>(loc, ioFunc, ioArgs).getResult(0);
}

template <>
mlir::Value genIOOption<Fortran::parser::ConnectSpec::CharExpr>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value cookie, const Fortran::parser::ConnectSpec::CharExpr &spec) {
  auto &builder = converter.getFirOpBuilder();
  mlir::FuncOp ioFunc;
  switch (std::get<Fortran::parser::ConnectSpec::CharExpr::Kind>(spec.t)) {
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Access:
    ioFunc = getIORuntimeFunc<mkIOKey(SetAccess)>(builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Action:
    ioFunc = getIORuntimeFunc<mkIOKey(SetAction)>(builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Asynchronous:
    ioFunc = getIORuntimeFunc<mkIOKey(SetAsynchronous)>(builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Blank:
    ioFunc = getIORuntimeFunc<mkIOKey(SetBlank)>(builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Decimal:
    ioFunc = getIORuntimeFunc<mkIOKey(SetDecimal)>(builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Delim:
    ioFunc = getIORuntimeFunc<mkIOKey(SetDelim)>(builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Encoding:
    ioFunc = getIORuntimeFunc<mkIOKey(SetEncoding)>(builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Form:
    ioFunc = getIORuntimeFunc<mkIOKey(SetForm)>(builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Pad:
    ioFunc = getIORuntimeFunc<mkIOKey(SetPad)>(builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Position:
    ioFunc = getIORuntimeFunc<mkIOKey(SetPosition)>(builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Round:
    ioFunc = getIORuntimeFunc<mkIOKey(SetRound)>(builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Sign:
    ioFunc = getIORuntimeFunc<mkIOKey(SetSign)>(builder);
    break;
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Convert:
    llvm_unreachable("CONVERT not part of the runtime::io interface");
  case Fortran::parser::ConnectSpec::CharExpr::Kind::Dispose:
    llvm_unreachable("DISPOSE not part of the runtime::io interface");
  }
  mlir::FunctionType ioFuncTy = ioFunc.getType();
  auto tup = lowerStringLit(
      converter, loc, std::get<Fortran::parser::ScalarDefaultCharExpr>(spec.t),
      ioFuncTy.getInput(1), ioFuncTy.getInput(2));
  llvm::SmallVector<mlir::Value, 4> ioArgs = {cookie, std::get<0>(tup),
                                              std::get<1>(tup)};
  return builder.create<mlir::CallOp>(loc, ioFunc, ioArgs).getResult(0);
}

template <>
mlir::Value genIOOption<Fortran::parser::ConnectSpec::Recl>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value cookie, const Fortran::parser::ConnectSpec::Recl &spec) {
  auto &builder = converter.getFirOpBuilder();
  mlir::FuncOp ioFunc = getIORuntimeFunc<mkIOKey(SetRecl)>(builder);
  mlir::FunctionType ioFuncTy = ioFunc.getType();
  auto expr = converter.genExprValue(Fortran::semantics::GetExpr(spec.v), loc);
  auto reclVal =
      builder.create<fir::ConvertOp>(loc, ioFuncTy.getInput(1), expr);
  llvm::SmallVector<mlir::Value, 4> ioArgs = {cookie, reclVal};
  return builder.create<mlir::CallOp>(loc, ioFunc, ioArgs).getResult(0);
}
template <>
mlir::Value genIOOption<Fortran::parser::StatusExpr>(
    Fortran::lower::AbstractConverter &converter, mlir::Location loc,
    mlir::Value cookie, const Fortran::parser::StatusExpr &spec) {
  auto &builder = converter.getFirOpBuilder();
  mlir::FuncOp ioFunc = getIORuntimeFunc<mkIOKey(SetStatus)>(builder);
  mlir::FunctionType ioFuncTy = ioFunc.getType();
  auto tup = lowerStringLit(converter, loc, spec.v, ioFuncTy.getInput(1),
                            ioFuncTy.getInput(2));
  llvm::SmallVector<mlir::Value, 4> ioArgs = {cookie, std::get<0>(tup),
                                              std::get<1>(tup)};
  return builder.create<mlir::CallOp>(loc, ioFunc, ioArgs).getResult(0);
}

template <>
mlir::Value genIOOption<Fortran::parser::MsgVariable>(
    Fortran::lower::AbstractConverter &, mlir::Location, mlir::Value cookie,
    const Fortran::parser::MsgVariable &) {
  // see error handling
  return cookie;
}
template <>
mlir::Value genIOOption<Fortran::parser::StatVariable>(
    Fortran::lower::AbstractConverter &, mlir::Location, mlir::Value cookie,
    const Fortran::parser::StatVariable &) {
  // see error handling
  return cookie;
}
template <>
mlir::Value
genIOOption<Fortran::parser::ErrLabel>(Fortran::lower::AbstractConverter &,
                                       mlir::Location, mlir::Value cookie,
                                       const Fortran::parser::ErrLabel &) {
  // see error handling
  return cookie;
}

namespace {
struct ErrorHandling {
  bool hasErrorHandlers() { return ioStat || err || end || eor || ioMsg; }
  bool ioStat{};
  bool err{};
  bool end{};
  bool eor{};
  bool ioMsg{};
};
} // namespace

template <typename A>
void genErrorOption(ErrorHandling &handle, const A &) {}
template <>
void genErrorOption<Fortran::parser::MsgVariable>(
    ErrorHandling &eh, const Fortran::parser::MsgVariable &) {
  eh.ioMsg = true;
}
template <>
void genErrorOption<Fortran::parser::StatVariable>(
    ErrorHandling &eh, const Fortran::parser::StatVariable &) {
  eh.ioStat = true;
}
template <>
void genErrorOption<Fortran::parser::ErrLabel>(
    ErrorHandling &eh, const Fortran::parser::ErrLabel &) {
  eh.err = true;
}

template <typename SEEK, typename A>
static bool hasMem(const A &stmt) {
  for (const auto &spec : stmt.v)
    if (std::holds_alternative<SEEK>(spec.u))
      return true;
  return false;
}

/// Get the sought expression from the specifier list.
template <typename SEEK, typename A>
static const Fortran::semantics::SomeExpr *getMem(const A &stmt) {
  for (const auto &spec : stmt.v)
    if (auto *f = std::get_if<SEEK>(&spec.u))
      return Fortran::semantics::GetExpr(f->v);
  llvm_unreachable("must have a file unit");
}

/// For each specifier, build the appropriate call, threading the cookie, and
/// return the final ssa-value of the cookie.
template <typename A>
static mlir::Value threadSpecs(Fortran::lower::AbstractConverter &converter,
                               mlir::Location loc, mlir::Value cookie,
                               const A &stmt) {
  for (const auto &spec : stmt.v) {
    cookie = std::visit(
        Fortran::common::visitors{
            [&](const Fortran::parser::FileUnitNumber &) { return cookie; },
            [&](const Fortran::parser::ConnectSpec::Newunit &) {
              return cookie;
            },
            [&](const auto &x) {
              return genIOOption(converter, loc, cookie, x);
            }},
        spec.u);
  }
  return cookie;
}

template <typename A>
static mlir::Value threadErrors(Fortran::lower::AbstractConverter &converter,
                                mlir::Location loc, mlir::Value cookie,
                                const A &stmt) {
  ErrorHandling eh{};
  for (const auto &spec : stmt.v)
    std::visit(Fortran::common::visitors{[&](const auto &x) {
                 genErrorOption(eh, x);
               }},
               spec.u);
  if (eh.hasErrorHandlers()) {
    auto &builder = converter.getFirOpBuilder();
    mlir::FuncOp ioFunc = getIORuntimeFunc<mkIOKey(EnableHandlers)>(builder);
    mlir::FunctionType ioFuncTy = ioFunc.getType();
    auto hasIoStat = builder.create<mlir::ConstantOp>(
        loc, builder.getIntegerAttr(ioFuncTy.getInput(1), eh.ioStat));
    auto hasErr = builder.create<mlir::ConstantOp>(
        loc, builder.getIntegerAttr(ioFuncTy.getInput(2), eh.err));
    auto hasEnd = builder.create<mlir::ConstantOp>(
        loc, builder.getIntegerAttr(ioFuncTy.getInput(3), eh.end));
    auto hasEor = builder.create<mlir::ConstantOp>(
        loc, builder.getIntegerAttr(ioFuncTy.getInput(4), eh.eor));
    auto hasIoMsg = builder.create<mlir::ConstantOp>(
        loc, builder.getIntegerAttr(ioFuncTy.getInput(5), eh.ioMsg));
    llvm::SmallVector<mlir::Value, 4> ioArgs = {cookie, hasIoStat, hasErr,
                                                hasEnd, hasEor,    hasIoMsg};
    cookie = builder.create<mlir::CallOp>(loc, ioFunc, ioArgs).getResult(0);
  }
  return cookie;
}

//===----------------------------------------------------------------------===//

/// Lower print statement assuming a dummy runtime interface for now.
static void lowerPrintStatement(Fortran::lower::AbstractConverter &converter,
                                mlir::Location loc, mlir::ValueRange args,
                                const Fortran::parser::Format &format) {
  mlir::FuncOp beginFunc;
  auto &builder = converter.getFirOpBuilder();
  auto formatItems = lowerFormat(converter, format);
  if (formatItems.has_value()) {
    // has a format
    TODO();
  } else {
    beginFunc = getIORuntimeFunc<mkIOKey(BeginExternalListOutput)>(builder);
  }
  mlir::FunctionType beginFuncTy = beginFunc.getType();

  // Initiate io
  mlir::Value defaultUnit = builder.create<mlir::ConstantOp>(
      loc, builder.getIntegerAttr(beginFuncTy.getInput(0),
                                  Fortran::runtime::io::DefaultUnit));
  auto srcFileName = getDefaultFilename(builder, loc, beginFuncTy.getInput(1));
  auto lineNo = getDefaultLineNo(builder, loc, beginFuncTy.getInput(2));
  llvm::SmallVector<mlir::Value, 4> beginArgs{defaultUnit, srcFileName, lineNo};
  mlir::Value cookie =
      builder.create<mlir::CallOp>(loc, beginFunc, beginArgs).getResult(0);

  // Call data transfer runtime function
  for (mlir::Value arg : args) {
    // need some special handling for COMPLEX and CHARACTER
    auto operands = splitArguments(builder, loc, arg);
    genOutputRuntimeFunc(builder, loc, arg.getType(), cookie, operands);
  }
  genEndIO(builder, loc, cookie);
}

//===----------------------------------------------------------------------===//
// Generators for each I/O statement type.
//===----------------------------------------------------------------------===//

template <typename K, typename S>
static mlir::Value genBasicIOStmt(Fortran::lower::AbstractConverter &converter,
                                  const S &stmt) {
  auto &builder = converter.getFirOpBuilder();
  auto loc = converter.getCurrentLocation();
  auto beginFunc = getIORuntimeFunc<K>(builder);
  mlir::FunctionType beginFuncTy = beginFunc.getType();
  auto unit = converter.genExprValue(
      getMem<Fortran::parser::FileUnitNumber>(stmt), loc);
  auto un = builder.create<fir::ConvertOp>(loc, beginFuncTy.getInput(0), unit);
  auto file = getDefaultFilename(builder, loc, beginFuncTy.getInput(1));
  auto line = getDefaultLineNo(builder, loc, beginFuncTy.getInput(2));
  llvm::SmallVector<mlir::Value, 4> args{un, file, line};
  auto cookie = builder.create<mlir::CallOp>(loc, beginFunc, args).getResult(0);
  cookie = threadErrors(converter, loc, cookie, stmt);
  cookie = threadSpecs(converter, loc, cookie, stmt);
  return genEndIO(builder, converter.getCurrentLocation(), cookie);
}

mlir::Value Fortran::lower::genBackspaceStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::BackspaceStmt &stmt) {
  return genBasicIOStmt<mkIOKey(BeginBackspace)>(converter, stmt);
}

mlir::Value Fortran::lower::genEndfileStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::EndfileStmt &stmt) {
  return genBasicIOStmt<mkIOKey(BeginEndfile)>(converter, stmt);
}

mlir::Value
Fortran::lower::genFlushStatement(Fortran::lower::AbstractConverter &converter,
                                  const Fortran::parser::FlushStmt &stmt) {
  return genBasicIOStmt<mkIOKey(BeginFlush)>(converter, stmt);
}

mlir::Value
Fortran::lower::genRewindStatement(Fortran::lower::AbstractConverter &converter,
                                   const Fortran::parser::RewindStmt &stmt) {
  return genBasicIOStmt<mkIOKey(BeginRewind)>(converter, stmt);
}

mlir::Value
Fortran::lower::genOpenStatement(Fortran::lower::AbstractConverter &converter,
                                 const Fortran::parser::OpenStmt &stmt) {
  auto &builder = converter.getFirOpBuilder();
  mlir::FuncOp beginFunc;
  llvm::SmallVector<mlir::Value, 4> beginArgs;
  auto loc = converter.getCurrentLocation();
  if (hasMem<Fortran::parser::FileUnitNumber>(stmt)) {
    beginFunc = getIORuntimeFunc<mkIOKey(BeginOpenUnit)>(builder);
    mlir::FunctionType beginFuncTy = beginFunc.getType();
    auto unit = converter.genExprValue(
        getMem<Fortran::parser::FileUnitNumber>(stmt), loc);
    beginArgs.push_back(
        builder.create<fir::ConvertOp>(loc, beginFuncTy.getInput(0), unit));
    beginArgs.push_back(
        getDefaultFilename(builder, loc, beginFuncTy.getInput(1)));
    beginArgs.push_back(
        getDefaultLineNo(builder, loc, beginFuncTy.getInput(2)));
  } else {
    assert(hasMem<Fortran::parser::ConnectSpec::Newunit>(stmt));
    beginFunc = getIORuntimeFunc<mkIOKey(BeginOpenNewUnit)>(builder);
    mlir::FunctionType beginFuncTy = beginFunc.getType();
    beginArgs.push_back(
        getDefaultFilename(builder, loc, beginFuncTy.getInput(0)));
    beginArgs.push_back(
        getDefaultLineNo(builder, loc, beginFuncTy.getInput(1)));
  }
  auto cookie =
      builder.create<mlir::CallOp>(loc, beginFunc, beginArgs).getResult(0);
  cookie = threadErrors(converter, loc, cookie, stmt);
  cookie = threadSpecs(converter, loc, cookie, stmt);
  return genEndIO(builder, loc, cookie);
}

mlir::Value
Fortran::lower::genCloseStatement(Fortran::lower::AbstractConverter &converter,
                                  const Fortran::parser::CloseStmt &stmt) {
  return genBasicIOStmt<mkIOKey(BeginClose)>(converter, stmt);
}

mlir::Value
Fortran::lower::genWaitStatement(Fortran::lower::AbstractConverter &converter,
                                 const Fortran::parser::WaitStmt &stmt) {
  auto &builder = converter.getFirOpBuilder();
  auto loc = converter.getCurrentLocation();
  bool hasId = hasMem<Fortran::parser::IdExpr>(stmt);
  mlir::FuncOp beginFunc =
      hasId ? getIORuntimeFunc<mkIOKey(BeginWait)>(builder)
            : getIORuntimeFunc<mkIOKey(BeginWaitAll)>(builder);
  mlir::FunctionType beginFuncTy = beginFunc.getType();
  auto unit = converter.genExprValue(
      getMem<Fortran::parser::FileUnitNumber>(stmt), loc);
  auto un = builder.create<fir::ConvertOp>(loc, beginFuncTy.getInput(0), unit);
  llvm::SmallVector<mlir::Value, 4> args{un};
  if (hasId) {
    auto id =
        converter.genExprValue(getMem<Fortran::parser::IdExpr>(stmt), loc);
    args.push_back(
        builder.create<fir::ConvertOp>(loc, beginFuncTy.getInput(1), id));
  }
  auto cookie = builder.create<mlir::CallOp>(loc, beginFunc, args).getResult(0);
  cookie = threadErrors(converter, loc, cookie, stmt);
  return genEndIO(builder, converter.getCurrentLocation(), cookie);
}

void Fortran::lower::genPrintStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::PrintStmt &stmt) {
  llvm::SmallVector<mlir::Value, 4> args;
  for (auto &item : std::get<std::list<Fortran::parser::OutputItem>>(stmt.t)) {
    if (auto *pe = std::get_if<Fortran::parser::Expr>(&item.u)) {
      auto loc = converter.genLocation(pe->source);
      args.push_back(
          converter.genExprValue(Fortran::semantics::GetExpr(*pe), loc));
    } else {
      TODO(); // TODO implied do
    }
  }
  lowerPrintStatement(converter, converter.getCurrentLocation(), args,
                      std::get<Fortran::parser::Format>(stmt.t));
}

mlir::Value
Fortran::lower::genReadStatement(Fortran::lower::AbstractConverter &converter,
                                 const Fortran::parser::ReadStmt &) {
  auto &builder = converter.getFirOpBuilder();
  mlir::FuncOp beginFunc;
  // if (...
  beginFunc = getIORuntimeFunc<mkIOKey(BeginExternalListInput)>(builder);
  // else if (...
  TODO();
  return {};
}

mlir::Value
Fortran::lower::genWriteStatement(Fortran::lower::AbstractConverter &converter,
                                  const Fortran::parser::WriteStmt &) {
  auto &builder = converter.getFirOpBuilder();
  mlir::FuncOp beginFunc;
  // if (...
  beginFunc = getIORuntimeFunc<mkIOKey(BeginExternalListOutput)>(builder);
  // else if (...
  TODO();
  return {};
}

mlir::Value Fortran::lower::genInquireStatement(
    Fortran::lower::AbstractConverter &converter,
    const Fortran::parser::InquireStmt &) {
  auto &builder = converter.getFirOpBuilder();
  mlir::FuncOp beginFunc;
  // if (...
  beginFunc = getIORuntimeFunc<mkIOKey(BeginInquireUnit)>(builder);
  // else if (...
  beginFunc = getIORuntimeFunc<mkIOKey(BeginInquireFile)>(builder);
  // else
  beginFunc = getIORuntimeFunc<mkIOKey(BeginInquireIoLength)>(builder);
  TODO();
  return {};
}
