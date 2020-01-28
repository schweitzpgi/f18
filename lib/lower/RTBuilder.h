//===-- lib/lower/cfg-builder.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines some C++17 template classes that are used to convert the
/// signatures of plain old C functions into a model that can be used to
/// generate MLIR calls to those functions. This can be used to autogenerate
/// tables at compiler compile-time to call runtime support code.
///
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_RT_BUILDER_H_
#define FORTRAN_LOWER_RT_BUILDER_H_

#include "fir/Dialect/FIRType.h"
#include "flang/lower/ConvertType.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/ADT/SmallVector.h"
#include <cstddef>
#include <functional>

// List the runtime headers we want to be able to dissect
#include "../../runtime/io-api.h"

namespace Fortran::lower {

using TypeBuilderFunc = mlir::Type (*)(mlir::MLIRContext *);
using FuncTypeBuilderFunc = mlir::FunctionType (*)(mlir::MLIRContext *);

template <typename>
struct errorNoBuilderForType;

/// Return a function that returns the type signature model for the type `T`
/// when provided an MLIRContext*. This allows one to translate C(++) function
/// signatures from runtime header files to MLIR signatures into a static table
/// at compile-time.
///
/// For example, when `T` is `int`, return a function that returns the MLIR
/// standard type `i32` when `sizeof(int)` is 4.
template <typename T>
static constexpr TypeBuilderFunc getModel() {
  using namespace std::placeholders;
  if constexpr (std::is_same_v<T, int>) {
    return getModelForInt;
  } else if constexpr (std::is_same_v<T, int &>) {
    return getModelForIntRef;
  } else if constexpr (std::is_same_v<T, std::int64_t>) {
    return getModelForInt64;
  } else if constexpr (std::is_same_v<T, std::int64_t &>) {
    return getModelForInt64Ref;
  } else if constexpr (std::is_same_v<T, std::size_t>) {
    return getModelForSize;
  } else if constexpr (std::is_same_v<T, double>) {
    return getModelForDouble;
  } else if constexpr (std::is_same_v<T, double &>) {
    return getModelForDoubleRef;
  } else if constexpr (std::is_same_v<T, float>) {
    return getModelForFloat;
  } else if constexpr (std::is_same_v<T, float &>) {
    return getModelForFloatRef;
  } else if constexpr (std::is_same_v<T, runtime::io::Iostat>) {
    return getModelForIostat;
  } else if constexpr (std::is_same_v<T, bool>) {
    return getModelForBool;
  } else if constexpr (std::is_same_v<T, bool &>) {
    return getModelForBoolRef;
  } else if constexpr (std::is_same_v<T, runtime::io::IoStatementState *>) {
    return getModelForCookie;
  } else if constexpr (std::is_same_v<T, char *>) {
    return getModelForCharPtr;
  } else if constexpr (std::is_same_v<T, const char *>) {
    return getModelForConstCharPtr;
  } else if constexpr (std::is_same_v<T, void>) {
    return getModelForVoid;
  } else if constexpr (std::is_same_v<T, const runtime::Descriptor &>) {
    return getModelForDescriptor;
  } else if constexpr (std::is_same_v<T, const runtime::NamelistGroup &>) {
    return getModelForNamelistGroup;
  } else {
    return errorNoBuilderForType<T>{}; // intentionally force compile-time error
  }
}

template <typename...>
struct RuntimeTableKey;
template <typename RT, typename... ATs>
struct RuntimeTableKey<RT(ATs...)> {
  static constexpr FuncTypeBuilderFunc getTypeModel() {
    return [](mlir::MLIRContext *ctxt) {
      TypeBuilderFunc ret = getModel<RT>();
      std::array<TypeBuilderFunc, sizeof...(ATs)> args = {getModel<ATs>()...};
      mlir::Type retTy = ret(ctxt);
      llvm::SmallVector<mlir::Type, sizeof...(ATs)> argTys;
      for (auto f : args)
        argTys.push_back(f(ctxt));
      return mlir::FunctionType::get(argTys, {retTy}, ctxt);
    };
  }
};

template <char... Cs>
using RuntimeIdentifier = std::integer_sequence<char, Cs...>;
template <typename T, T... Cs>
static constexpr RuntimeIdentifier<Cs...> operator""_rt_ident() {
  return {};
}

template <typename...>
struct RuntimeTableEntry;
template <typename KT, char... Cs>
struct RuntimeTableEntry<RuntimeTableKey<KT>, RuntimeIdentifier<Cs...>> {
  static constexpr FuncTypeBuilderFunc getTypeModel() {
    return RuntimeTableKey<KT>::getTypeModel();
  }
  static constexpr const char name[sizeof...(Cs) + 1] = {Cs..., '\0'};
};

#define QuoteKey(X) #X##_rt_ident
#define ExpandKey(X) QuoteKey(X)
#define mkKey(X)                                                               \
  Br::RuntimeTableEntry<Br::RuntimeTableKey<decltype(X)>,                      \
                        decltype(ExpandKey(X))>

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_RT_BUILDER_H_
