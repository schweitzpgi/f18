//===-- Runtime.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/lower/Runtime.h"
#include "NSAliases.h"
#include "flang/lower/OpBuilder.h"
#include "flang/optimizer/Dialect/FIRType.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/SmallVector.h"
#include <cassert>

namespace Fortran::lower {

M::Type RuntimeStaticDescription::getMLIRType(TypeCode t,
                                              M::MLIRContext *context) {
  switch (t) {
  case TypeCode::i32:
    return M::IntegerType::get(32, context);
  case TypeCode::i64:
    return M::IntegerType::get(64, context);
  case TypeCode::f32:
    return M::FloatType::getF32(context);
  case TypeCode::f64:
    return M::FloatType::getF64(context);
  // TODO need to access mapping between fe/target
  case TypeCode::c32:
    return fir::CplxType::get(context, 4);
  case TypeCode::c64:
    return fir::CplxType::get(context, 8);
  case TypeCode::boolean:
    return M::IntegerType::get(8, context);
  case TypeCode::charPtr:
    return fir::ReferenceType::get(fir::CharacterType::get(context, 1));
  // ! IOCookie is experimental only so far
  case TypeCode::IOCookie:
    return fir::ReferenceType::get(M::IntegerType::get(64, context));
  }
  assert(false && "bug");
  return {};
}

M::FunctionType
RuntimeStaticDescription::getMLIRFunctionType(M::MLIRContext *context) const {
  L::SmallVector<M::Type, 2> argMLIRTypes;
  for (const TypeCode &t : argumentTypeCodes) {
    argMLIRTypes.push_back(getMLIRType(t, context));
  }
  if (resultTypeCode.has_value()) {
    M::Type resMLIRType{getMLIRType(*resultTypeCode, context)};
    return M::FunctionType::get(argMLIRTypes, resMLIRType, context);
  }
  return M::FunctionType::get(argMLIRTypes, {}, context);
}

M::FuncOp RuntimeStaticDescription::getFuncOp(M::OpBuilder &builder) const {
  M::ModuleOp module{getModule(&builder)};
  M::FunctionType funTy{getMLIRFunctionType(module.getContext())};
  auto function{getNamedFunction(module, symbol)};
  if (!function) {
    function = createFunction(module, symbol, funTy);
    function.setAttr("fir.runtime", builder.getUnitAttr());
  } else {
    assert(funTy == function.getType() && "conflicting runtime declaration");
  }
  return function;
}

class RuntimeEntryDescription : public RuntimeStaticDescription {
public:
  using Key = RuntimeEntryCode;
  constexpr RuntimeEntryDescription(Key k, const char *s, MaybeTypeCode r,
                                    TypeCodeVector a)
      : RuntimeStaticDescription{s, r, a}, key{k} {}
  Key key;
};

using RT = RuntimeStaticDescription;
using RType = typename RT::TypeCode;
using Args = typename RT::TypeCodeVector;
using RTC = RuntimeEntryCode;

static constexpr RuntimeEntryDescription runtimeTable[]{
    {RTC::StopStatement, "StopStatement", RT::voidTy,
     Args::create<RType::i32, RType::boolean, RType::boolean>()},
    {RTC::StopStatementText, "StopStatementText", RT::voidTy,
     Args::create<RType::charPtr, RType::i32, RType::boolean,
                  RType::boolean>()},
    {RTC::FailImageStatement, "StopStatementText", RT::voidTy,
     Args::create<>()},
};

static constexpr StaticMultimapView<RuntimeEntryDescription> runtimeMap{
    runtimeTable};

M::FuncOp genRuntimeFunction(RuntimeEntryCode code, M::OpBuilder &builder) {
  auto description{runtimeMap.find(code)};
  assert(description != runtimeMap.end());
  return description->getFuncOp(builder);
}

} // namespace Fortran::lower
