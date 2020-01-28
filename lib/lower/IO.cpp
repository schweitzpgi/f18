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
#include "fir/Dialect/FIROps.h"
#include "flang/lower/Bridge.h"
#include "flang/lower/OpBuilder.h"
#include "flang/lower/Runtime.h"
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

/// Define actions to sort runtime functions. One actions
/// may be associated to one or more runtime function.
/// Actions are the keys in the StaticMultimapView used to
/// hold the io runtime description in a static constexpr way.
enum class IOAction { BeginExternalList, Output, EndIO };

class IORuntimeDescription : public RuntimeStaticDescription {
public:
  using Key = IOAction;
  constexpr IORuntimeDescription(IOAction act, const char *s, MaybeTypeCode r,
                                 TypeCodeVector a)
      : RuntimeStaticDescription{s, r, a}, key{act} {}
  static M::Type getIOCookieType(M::MLIRContext *context) {
    return getMLIRType(TypeCode::IOCookie, context);
  }
  IOAction key;
};

using IORuntimeMap = StaticMultimapView<IORuntimeDescription>;

using RT = RuntimeStaticDescription;
using RType = typename RT::TypeCode;
using Args = typename RT::TypeCodeVector;
using IOA = IOAction;

/// This is were the IO runtime are to be described.
/// The array need to be sorted on the Actions.
/// Experimental runtime for now.
static constexpr IORuntimeDescription ioRuntimeTable[]{
    {IOA::BeginExternalList, NAMIFY(BeginExternalListOutput), RType::IOCookie,
     Args::create<RType::i32>()},
    {IOA::Output, NAMIFY(OutputInteger64), RT::voidTy,
     Args::create<RType::IOCookie, RType::i64>()},
    {IOA::Output, NAMIFY(OutputReal64), RT::voidTy,
     Args::create<RType::IOCookie, RType::f64>()},
    {IOA::EndIO, NAMIFY(EndIOStatement), RT::voidTy,
     Args::create<RType::IOCookie>()},
};

static constexpr IORuntimeMap ioRuntimeMap{ioRuntimeTable};

/// This helper can be used to access io runtime functions that
/// are mapped to an IOAction that must be mapped to one and
/// exactly one runtime function. This constraint is enforced
/// at compile time. This search is resolved at compile time.
template <IORuntimeDescription::Key key>
static M::FuncOp getIORuntimeFunction(M::OpBuilder &builder) {
  static constexpr auto runtimeDescription{ioRuntimeMap.find(key)};
  static_assert(runtimeDescription != ioRuntimeMap.end());
  return runtimeDescription->getFuncOp(builder);
}

/// This helper can be used to access io runtime functions that
/// are mapped to Output IOAction that must be mapped to at least one
/// runtime function but can be mapped to more functions.
/// This helper returns the function that has the same
/// M::FunctionType as the one seeked. It may therefore dynamically fail
/// if no function mapped to the Action has the seeked M::FunctionType.
static M::FuncOp getOutputRuntimeFunction(M::OpBuilder &builder, M::Type type) {
  static constexpr auto descriptionRange{ioRuntimeMap.getRange(IOA::Output)};
  static_assert(!descriptionRange.empty());

  M::MLIRContext *context{getModule(&builder).getContext()};
  L::SmallVector<M::Type, 2> argTypes{
      IORuntimeDescription::getIOCookieType(context), type};

  M::FunctionType seekedType{M::FunctionType::get(argTypes, {}, context)};
  for (const auto &description : descriptionRange) {
    if (description.getMLIRFunctionType(context) == seekedType)
      return description.getFuncOp(builder);
  }
  assert(false && "IO output runtime function not defined for this type");
  return {};
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
  M::FuncOp endIOFunc{getIORuntimeFunc<mkIOKey(EndIoStatement)>(builder)};
  L::SmallVector<M::Value, 1> endArgs{cookie};
  return builder.create<M::CallOp>(loc, endIOFunc, endArgs).getResult(0);
}

/// Lower print statement assuming a dummy runtime interface for now.
void lowerPrintStatement(M::OpBuilder &builder, M::Location loc, int format,
                         M::ValueRange args) {
  M::FuncOp beginFunc =
      getIORuntimeFunc<mkIOKey(BeginExternalListOutput)>(builder);

  // Initiate io
  M::Type externalUnitType{builder.getIntegerType(32)};
  M::Value defaultUnit{builder.create<M::ConstantOp>(
      loc, builder.getIntegerAttr(externalUnitType, 1))};
  L::SmallVector<M::Value, 1> beginArgs{defaultUnit};
  M::Value cookie{
      builder.create<M::CallOp>(loc, beginFunc, beginArgs).getResult(0)};

  // Call data transfer runtime function
  for (M::Value arg : args) {
    // FIXME: this loop is still using the older table
    L::SmallVector<M::Value, 1> operands{cookie, arg};
    M::FuncOp outputFunc{getOutputRuntimeFunction(builder, arg.getType())};
    builder.create<M::CallOp>(loc, outputFunc, operands);
  }
  genEndIO(builder, loc, cookie);
}

int lowerFormat(const Pa::Format &format) {
  /// FIXME: this is a stub; process the format and return it
  return {};
  TODO();
}

L::SmallVector<M::Value, 4>
lowerBeginArgsPositionOrFlush(AbstractConverter &converter, M::Location loc,
                              const std::list<Pa::PositionOrFlushSpec> &specs) {
  L::SmallVector<M::Value, 4> args;
  // 1. find the unit number expression and append it
  for (auto &sp : specs)
    if (auto *un = std::get_if<Pa::FileUnitNumber>(&sp.u)) {
      auto *expr{semantics::GetExpr(un->v)};
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
                      loc, getModelForCharPtr(builder.getContext()), varAddr)
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
  TODO();
  return {};
}

void genPrintStatement(Br::AbstractConverter &converter,
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
  lowerPrintStatement(converter.getOpBuilder(), converter.getCurrentLocation(),
                      lowerFormat(std::get<Pa::Format>(stmt.t)), args);
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
