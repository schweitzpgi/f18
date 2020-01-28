//===-- lower/IO.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/lower/IO.h"
#include "../../runtime/io-api.h"
#include "../parser/parse-tree.h"
#include "../semantics/tools.h"
#include "RTBuilder.h"
#include "flang/lower/Bridge.h"
#include "flang/lower/OpBuilder.h"
#include "flang/lower/Runtime.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"

#define NAMIFY_HELPER(X) #X
#define NAMIFY(X) NAMIFY_HELPER(IONAME(X))

namespace Br = Fortran::lower;
namespace M = mlir;
namespace Pa = Fortran::parser;

using namespace Fortran;
using namespace Fortran::lower;

#define TODO() assert(false && "not yet implemented")

namespace {

#define mkIOKey(X) mkKey(IONAME(X))

/// Static table of IO runtime calls
///
/// This logical map contains the name and type builder function for each IO
/// runtime function listed in the tuple. This table is fully constructed at
/// compile-time. Use the `mkIOKey` macro to access the table.
static constexpr std::tuple<
    mkIOKey(BeginExternalListOutput), mkIOKey(BeginExternalListInput),
    mkIOKey(BeginExternalFormattedOutput), mkIOKey(BeginExternalFormattedInput),
    mkIOKey(BeginUnformattedOutput), mkIOKey(BeginUnformattedInput),
    mkIOKey(EndIoStatement), mkIOKey(OutputInteger64), mkIOKey(InputInteger64),
    mkIOKey(OutputReal64), mkIOKey(InputReal64), mkIOKey(OutputReal32),
    mkIOKey(InputReal64), mkIOKey(OutputComplex64), mkIOKey(OutputComplex32),
    mkIOKey(BeginClose), mkIOKey(OutputAscii), mkIOKey(InputAscii),
    mkIOKey(OutputLogical), mkIOKey(InputLogical), mkIOKey(BeginFlush),
    mkIOKey(BeginBackspace), mkIOKey(BeginEndfile), mkIOKey(BeginRewind),
    mkIOKey(BeginOpenUnit), mkIOKey(BeginOpenNewUnit),
    mkIOKey(BeginInquireUnit), mkIOKey(BeginInquireFile),
    mkIOKey(BeginInquireIoLength), mkIOKey(EnableHandlers), mkIOKey(SetAdvance),
    mkIOKey(SetBlank), mkIOKey(SetDecimal), mkIOKey(SetDelim), mkIOKey(SetPad),
    mkIOKey(SetPos), mkIOKey(SetRec), mkIOKey(SetRound), mkIOKey(SetSign),
    mkIOKey(SetAccess), mkIOKey(SetAction), mkIOKey(SetAsynchronous),
    mkIOKey(SetEncoding), mkIOKey(SetEncoding), mkIOKey(SetForm),
    mkIOKey(SetPosition), mkIOKey(SetRecl), mkIOKey(SetStatus),
    mkIOKey(SetFile), mkIOKey(GetNewUnit), mkIOKey(GetSize),
    mkIOKey(GetIoLength), mkIOKey(GetIoMsg), mkIOKey(InquireCharacter),
    mkIOKey(InquireLogical), mkIOKey(InquirePendingId),
    mkIOKey(InquireInteger64)>
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
  llvm::SmallVector<M::Type, 2> argTypes{
      IORuntimeDescription::getIOCookieType(context), type};

  M::FunctionType seekedType{M::FunctionType::get(argTypes, {}, context)};
  for (const auto &description : descriptionRange) {
    if (description.getMLIRFunctionType(context) == seekedType) {
      return description.getFuncOp(builder);
    }
  }
  assert(false && "IO output runtime function not defined for this type");
  return {};
}

/// Get (or generate) the MLIR FuncOp for a given IO runtime function. This
/// replaces getIORuntimeFunction.
template <typename E>
M::FuncOp getIORuntimeFunc(M::OpBuilder &builder) {
  auto module = getModule(&builder);
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
void genEndIO(M::OpBuilder &builder, M::Location loc, M::Value cookie) {
  // Terminate IO
  M::FuncOp endIOFunc{getIORuntimeFunc<mkIOKey(EndIoStatement)>(builder)};
  llvm::SmallVector<M::Value, 1> endArgs{cookie};
  builder.create<M::CallOp>(loc, endIOFunc, endArgs);
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
  llvm::SmallVector<M::Value, 1> beginArgs{defaultUnit};
  M::Value cookie{
      builder.create<M::CallOp>(loc, beginFunc, beginArgs).getResult(0)};

  // Call data transfer runtime function
  for (M::Value arg : args) {
    // FIXME: this loop is still using the older table
    llvm::SmallVector<M::Value, 1> operands{cookie, arg};
    M::FuncOp outputFunc{getOutputRuntimeFunction(builder, arg.getType())};
    builder.create<M::CallOp>(loc, outputFunc, operands);
  }
  endIO(builder, loc, cookie);
}

int lowerFormat(const Pa::Format &format) {
  /// FIXME: this is a stub; process the format and return it
  return {};
  TODO();
}

M::Value lowerPositionOrFlush(M::OpBuilder &builder, M::Location loc,
                              const std::list<Pa::PositionOrFlushSpec> &specs) {
  TODO();
  return {};
}

/// Generate IO calls for any of the "position or flush" like IO statements.
/// This is templatized with a statement type `S` and a key `K` for genericity.
template <typename S, typename K>
void genPosOrFlushLikeStmt(AbstractConverter &converter, const S &stmt) {
  auto builder = converter.getOpBuilder();
  auto loc = converter.getCurrentLocation();
  auto beginFunc = getIORuntimeFunc<K>(builder);
  auto cookie = lowerPositionOrFlush(builder, loc, stmt.v);
  genEndIO(builder, converter.getCurrentLocation(), cookie);
}

} // namespace

void Br::genBackspaceStatement(AbstractConverter &converter,
                               const Pa::BackspaceStmt &stmt) {
  genPosOrFlushLikeStmt<mkIOKey(BeginBackspace)>(converter, stmt);
}

void Br::genEndfileStatement(AbstractConverter &converter,
                             const Pa::EndfileStmt &stmt) {
  genPosOrFlushLikeStmt<mkIOKey(BeginEndfile)>(converter, stmt);
}

void Br::genFlushStatement(AbstractConverter &converter,
                           const Pa::FlushStmt &stmt) {
  genPosOrFlushLikeStmt<mkIOKey(BeginFlush)>(converter, stmt);
}

void Br::genRewindStatement(AbstractConverter &converter,
                            const Pa::RewindStmt &stmt) {
  genPosOrFlushLikeStmt<mkIOKey(BeginRewind)>(converter, stmt);
}

void Br::genOpenStatement(AbstractConverter &converter, const Pa::OpenStmt &) {
  auto builder = converter.getOpBuilder();
  M::FuncOp beginFunc;
  // if (...
  beginFunc = getIORuntimeFunc<mkIOKey(BeginOpenUnit)>(builder);
  // else
  beginFunc = getIORuntimeFunc<mkIOKey(BeginOpenNewUnit)>(builder);
  TODO();
}

void Br::genCloseStatement(AbstractConverter &converter,
                           const Pa::CloseStmt &) {
  auto builder = converter.getOpBuilder();
  M::FuncOp beginFunc{getIORuntimeFunc<mkIOKey(BeginClose)>(builder)};
  TODO();
}

void Br::genPrintStatement(Br::AbstractConverter &converter,
                           const Pa::PrintStmt &stmt) {
  llvm::SmallVector<M::Value, 4> args;
  for (auto &item : std::get<std::list<Pa::OutputItem>>(stmt.t)) {
    if (auto *pe{std::get_if<Pa::Expr>(&item.u)}) {
      auto loc{converter.genLocation(pe->source)};
      args.push_back(converter.genExprValue(*semantics::GetExpr(*pe), &loc));
    } else {
      TODO(); // TODO implied do
    }
  }
  lowerPrintStatement(converter.getOpBuilder(), converter.getCurrentLocation(),
                      lowerFormat(std::get<Pa::Format>(stmt.t)), args);
}

void Br::genReadStatement(AbstractConverter &converter, const Pa::ReadStmt &) {
  auto builder = converter.getOpBuilder();
  M::FuncOp beginFunc;
  // if (...
  beginFunc = getIORuntimeFunc<mkIOKey(BeginExternalListInput)>(builder);
  // else if (...
  TODO();
}

void Br::genWriteStatement(AbstractConverter &converter,
                           const Pa::WriteStmt &) {
  auto builder = converter.getOpBuilder();
  M::FuncOp beginFunc;
  // if (...
  beginFunc = getIORuntimeFunc<mkIOKey(BeginExternalListOutput)>(builder);
  // else if (...
  TODO();
}

void Br::genInquireStatement(AbstractConverter &converter,
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
}
