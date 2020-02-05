//===-- IO.cpp -- I/O statement lowering ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/lower/IO.h"
#include "../../runtime/io-api.h"
#include "NSAliases.h"
#include "RTBuilder.h"
#include "flang/lower/Bridge.h"
#include "flang/lower/OpBuilder.h"
#include "flang/lower/Runtime.h"
#include "flang/optimizer/Dialect/FIROps.h"
#include "flang/parser/parse-tree.h"
#include "flang/semantics/tools.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"

#define NAMIFY_HELPER(X) #X
#define NAMIFY(X) NAMIFY_HELPER(IONAME(X))

#define TODO() assert(false && "not yet implemented")

using namespace Io;
namespace Fortran::lower {
namespace {

#define mkIOKey(X) mkKey(IONAME(X))

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
    mkIOKey(OutputInteger64), mkIOKey(InputInteger64), mkIOKey(OutputReal32),
    mkIOKey(InputReal32), mkIOKey(OutputReal64), mkIOKey(InputReal64),
    mkIOKey(OutputComplex64), mkIOKey(OutputComplex32), mkIOKey(OutputAscii),
    mkIOKey(InputAscii), mkIOKey(OutputLogical), mkIOKey(InputLogical),
    mkIOKey(SetAccess), mkIOKey(SetAction), mkIOKey(SetAsynchronous),
    mkIOKey(SetEncoding), mkIOKey(SetEncoding), mkIOKey(SetForm),
    mkIOKey(SetPosition), mkIOKey(SetRecl), mkIOKey(SetStatus),
    mkIOKey(SetFile), mkIOKey(GetNewUnit), mkIOKey(GetSize),
    mkIOKey(GetIoLength), mkIOKey(GetIoMsg), mkIOKey(InquireCharacter),
    mkIOKey(InquireLogical), mkIOKey(InquirePendingId),
    mkIOKey(InquireInteger64), mkIOKey(EndIoStatement)>
    newIOTable;

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

/// Get (or generate) the MLIR FuncOp for a given IO runtime function. This
/// replaces getIORuntimeFunction.
template <typename E>
M::FuncOp getIORuntimeFunc(M::OpBuilder &builder) {
  M::ModuleOp module{getModule(&builder)};
  auto name = getName<E>();
  auto func = getNamedFunction(module, name);
  if (func)
    return func;
  auto funTy = getTypeModel<E>()(builder.getContext());
  func = createFunction(module, name, funTy);
  func.setAttr("fir.runtime", builder.getUnitAttr());
  func.setAttr("fir.io", builder.getUnitAttr());
  return func;
}

/// Generate a call to end an IO statement
M::Value genEndIO(M::OpBuilder &builder, M::Location loc, M::Value cookie) {
  // Terminate IO
  M::FuncOp endIOFunc = getIORuntimeFunc<mkIOKey(EndIoStatement)>(builder);
  L::SmallVector<M::Value, 1> endArgs{cookie};
  auto call = builder.create<M::CallOp>(loc, endIOFunc, endArgs);
  return call.getResult(0);
}

using FormatItems = std::optional<std::pair<M::Value, M::Value>>;

/// Translate a list of format-items into a set of call-backs that can be
/// emitted into the MLIR stream before each data item is processed
FormatItems lowerFormat(AbstractConverter &converter,
                        const Pa::Format &format) {
  FormatItems formatItems;
  std::visit(Co::visitors{
                 [](const Pa::DefaultCharExpr &) { /* string expression */ },
                 [](const Pa::Label &) { /* FORMAT statement */ },
                 [](const Pa::Star &) {},
             },
             format.u);
  return formatItems;
}

M::FuncOp getOutputRuntimeFunc(M::OpBuilder &builder, M::Type type) {
  if (auto ty = type.dyn_cast<M::IntegerType>()) {
    if (ty.getWidth() == 1)
      return getIORuntimeFunc<mkIOKey(OutputLogical)>(builder);
    return getIORuntimeFunc<mkIOKey(OutputInteger64)>(builder);
  } else if (auto ty = type.dyn_cast<M::FloatType>()) {
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
L::SmallVector<M::Value, 4> splitArguments(M::OpBuilder &builder,
                                           M::Location loc, M::Value arg) {
  M::Value zero;
  M::Value one;
  M::Type argTy = arg.getType();
  M::MLIRContext *context = argTy.getContext();
  bool isComplex = argTy.isa<fir::CplxType>();
  bool isBoxChar = argTy.isa<fir::BoxCharType>();

  if (isComplex || isBoxChar) {
    // Only create these constants when needed and not every time
    zero = builder.create<M::ConstantOp>(loc, builder.getI64IntegerAttr(0));
    one = builder.create<M::ConstantOp>(loc, builder.getI64IntegerAttr(1));
  }
  if (isComplex) {
    auto eleTy =
        fir::RealType::get(context, argTy.cast<fir::CplxType>().getFKind());
    M::Value realPart =
        builder.create<fir::ExtractValueOp>(loc, eleTy, arg, zero);
    M::Value imaginaryPart =
        builder.create<fir::ExtractValueOp>(loc, eleTy, arg, one);
    return {realPart, imaginaryPart};
  }
  if (isBoxChar) {
    M::Type ptrTy = fir::ReferenceType::get(M::IntegerType::get(8, context));
    M::Type sizeTy = M::IntegerType::get(64, context);
    M::Value pointerPart =
        builder.create<fir::ExtractValueOp>(loc, ptrTy, arg, zero);
    M::Value sizePart =
        builder.create<fir::ExtractValueOp>(loc, sizeTy, arg, one);
    return {pointerPart, sizePart};
  }
  return {arg};
}

/// Generate a call to an Output I/O function.
/// The specific call is determined dynamically by the argument type.
void genOutputRuntimeFunc(M::OpBuilder &builder, M::Location loc,
                          M::Type argType, M::Value cookie,
                          L::SmallVector<M::Value, 4> &operands) {
  int i = 1;
  L::SmallVector<M::Value, 4> actuals{cookie};
  auto outputFunc = getOutputRuntimeFunc(builder, argType);

  for (auto &op : operands)
    actuals.emplace_back(builder.create<fir::ConvertOp>(
        loc, outputFunc.getType().getInput(i++), op));
  builder.create<M::CallOp>(loc, outputFunc, actuals);
}

/// Lower print statement assuming a dummy runtime interface for now.
void lowerPrintStatement(AbstractConverter &converter, M::Location loc,
                         M::ValueRange args, const Pa::Format &format) {
  M::FuncOp beginFunc;
  M::OpBuilder &builder = converter.getOpBuilder();
  auto formatItems = lowerFormat(converter, format);
  if (formatItems.has_value()) {
    // has a format
    TODO();
  } else {
    beginFunc = getIORuntimeFunc<mkIOKey(BeginExternalListOutput)>(builder);
  }
  M::FunctionType beginFuncTy = beginFunc.getType();

  // Initiate io
  M::Value defaultUnit = builder.create<M::ConstantOp>(
      loc, builder.getIntegerAttr(beginFuncTy.getInput(0), 1));
  M::Value null =
      builder.create<M::ConstantOp>(loc, builder.getI64IntegerAttr(0));
  M::Value srcFileName =
      builder.create<fir::ConvertOp>(loc, beginFuncTy.getInput(1), null);
  M::Value lineNo = builder.create<M::ConstantOp>(
      loc, builder.getIntegerAttr(beginFuncTy.getInput(2), 0));
  L::SmallVector<M::Value, 3> beginArgs{defaultUnit, srcFileName, lineNo};
  M::Value cookie =
      builder.create<M::CallOp>(loc, beginFunc, beginArgs).getResult(0);

  // Call data transfer runtime function
  for (M::Value arg : args) {
    auto operands = splitArguments(builder, loc, arg);
    genOutputRuntimeFunc(builder, loc, arg.getType(), cookie, operands);
  }
  genEndIO(builder, loc, cookie);
}

L::SmallVector<M::Value, 4>
lowerBeginArgsPositionOrFlush(AbstractConverter &converter, M::Location loc,
                              const std::list<Pa::PositionOrFlushSpec> &specs) {
  L::SmallVector<M::Value, 4> args;
  // 1. find the unit number expression and append it
  for (auto &sp : specs)
    if (auto *un = std::get_if<Pa::FileUnitNumber>(&sp.u)) {
      auto *expr{Se::GetExpr(un->v)};
      args.push_back(converter.genExprValue(expr, loc));
      break;
    }
  // 2 & 3. add the filename and line as extracted from `loc`
  // FIXME
  return args;
}

/// 3 of the 4 cases in a position (or flush) spec concern the error handling of
/// the statement. We handle those 3 cases here.
M::Value lowerErrorHandlingPositionOrFlush(
    AbstractConverter &converter, M::Value endRes,
    const std::list<Pa::PositionOrFlushSpec> &specs) {
  M::Value result;
  auto builder{converter.getOpBuilder()};
  auto loc = converter.getCurrentLocation();
  for (auto &sp : specs) {
    std::visit(
        Co::visitors{
            [](const Pa::FileUnitNumber &) {
              // this is passed to the BeginFoo function
              // do nothing here
            },
            [&](const Pa::MsgVariable &var) {
              // call GetIoMsg, passing it the address of `var` and a length (in
              // bytes, or?). Effectively, we have to decompose a boxchar here.
              // TODO: this has to be a CHARACTER type, no?
              M::Value varAddr = converter.genExprAddr(Se::GetExpr(var), loc);
              M::FuncOp getIoMsg = getIORuntimeFunc<mkIOKey(GetIoMsg)>(builder);
              L::SmallVector<M::Value, 1> ioMsgArgs{
                  builder.create<fir::ConvertOp>(
                      loc, getModel<char *>()(builder.getContext()), varAddr)
                  /*, FIXME add length here */};
              builder.create<M::CallOp>(loc, getIoMsg, ioMsgArgs);
            },
            [&](const Pa::StatVariable &var) {
              /* store `endRes` to the variable `var` */
              M::Value varAddr = converter.genExprAddr(Se::GetExpr(var), loc);
              builder.create<fir::StoreOp>(loc, endRes, varAddr);
            },
            [&](const Pa::ErrLabel &label) {
              /* pass the `endRes` value for `fir.switch` op */
              result = endRes;
            },
        },
        sp.u);
  }
  return result;
}

/// Generate IO calls for any of the "position or flush" like IO statements.
/// This is templatized with a statement type `S` and a key `K` for genericity.
template <typename K, typename S>
M::Value genPosOrFlushLikeStmt(AbstractConverter &converter, const S &stmt) {
  auto builder = converter.getOpBuilder();
  auto loc = converter.getCurrentLocation();
  auto beginFunc = getIORuntimeFunc<K>(builder);
  auto args = lowerBeginArgsPositionOrFlush(converter, loc, stmt.v);
  auto call = builder.create<M::CallOp>(loc, beginFunc, args);
  // FIXME: add call to EnableHandlers as apropos
  auto cookie = call.getResult(0);
  auto endVal = genEndIO(builder, converter.getCurrentLocation(), cookie);
  return lowerErrorHandlingPositionOrFlush(converter, endVal, stmt.v);
}
} // namespace

M::Value genBackspaceStatement(AbstractConverter &converter,
                               const Pa::BackspaceStmt &stmt) {
  return genPosOrFlushLikeStmt<mkIOKey(BeginBackspace)>(converter, stmt);
}

M::Value genEndfileStatement(AbstractConverter &converter,
                             const Pa::EndfileStmt &stmt) {
  return genPosOrFlushLikeStmt<mkIOKey(BeginEndfile)>(converter, stmt);
}

M::Value genFlushStatement(AbstractConverter &converter,
                           const Pa::FlushStmt &stmt) {
  return genPosOrFlushLikeStmt<mkIOKey(BeginFlush)>(converter, stmt);
}

M::Value genRewindStatement(AbstractConverter &converter,
                            const Pa::RewindStmt &stmt) {
  return genPosOrFlushLikeStmt<mkIOKey(BeginRewind)>(converter, stmt);
}

M::Value genOpenStatement(AbstractConverter &converter, const Pa::OpenStmt &) {
  auto builder = converter.getOpBuilder();
  M::FuncOp beginFunc;
  // if (...
  beginFunc = getIORuntimeFunc<mkIOKey(BeginOpenUnit)>(builder);
  // else
  beginFunc = getIORuntimeFunc<mkIOKey(BeginOpenNewUnit)>(builder);
  TODO();
  return {};
}

M::Value genCloseStatement(AbstractConverter &converter,
                           const Pa::CloseStmt &) {
  auto builder = converter.getOpBuilder();
  M::FuncOp beginFunc{getIORuntimeFunc<mkIOKey(BeginClose)>(builder)};
  (void)beginFunc;
  TODO();
  return {};
}

void genPrintStatement(AbstractConverter &converter,
                       const Pa::PrintStmt &stmt) {
  L::SmallVector<M::Value, 4> args;
  for (auto &item : std::get<std::list<Pa::OutputItem>>(stmt.t)) {
    if (auto *pe{std::get_if<Pa::Expr>(&item.u)}) {
      auto loc{converter.genLocation(pe->source)};
      args.push_back(converter.genExprValue(Se::GetExpr(*pe), loc));
    } else {
      TODO(); // TODO implied do
    }
  }
  lowerPrintStatement(converter, converter.getCurrentLocation(), args,
                      std::get<Pa::Format>(stmt.t));
}

M::Value genReadStatement(AbstractConverter &converter, const Pa::ReadStmt &) {
  auto builder = converter.getOpBuilder();
  M::FuncOp beginFunc;
  // if (...
  beginFunc = getIORuntimeFunc<mkIOKey(BeginExternalListInput)>(builder);
  // else if (...
  TODO();
  return {};
}

M::Value genWriteStatement(AbstractConverter &converter,
                           const Pa::WriteStmt &) {
  auto builder = converter.getOpBuilder();
  M::FuncOp beginFunc;
  // if (...
  beginFunc = getIORuntimeFunc<mkIOKey(BeginExternalListOutput)>(builder);
  // else if (...
  TODO();
  return {};
}

M::Value genInquireStatement(AbstractConverter &converter,
                             const Pa::InquireStmt &) {
  auto builder = converter.getOpBuilder();
  M::FuncOp beginFunc;
  // if (...
  beginFunc = getIORuntimeFunc<mkIOKey(BeginInquireUnit)>(builder);
  // else if (...
  beginFunc = getIORuntimeFunc<mkIOKey(BeginInquireFile)>(builder);
  // else
  beginFunc = getIORuntimeFunc<mkIOKey(BeginInquireIoLength)>(builder);
  TODO();
  return {};
}
} // namespace Fortran::lower
