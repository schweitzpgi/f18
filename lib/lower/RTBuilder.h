//===-- lib/lower/cfg-builder.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_RT_BUILDER_H_
#define FORTRAN_LOWER_RT_BUILDER_H_

#include "fir/Dialect/FIRType.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/StandardTypes.h"
#include <cstddef>
#include <functional>

// List the runtime headers we want to be able to dissect
#include "../../runtime/io-api.h"

namespace Fortran::lower {

using TypeBuilderFunc = std::function<mlir::Type(mlir::MLIRContext *)>;
using FuncTypeBuilderFunc =
    std::function<mlir::FunctionType(mlir::MLIRContext *)>;

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
  using Iostat = typename runtime::io::Iostat;
  if constexpr (std::is_same_v<T, int>) {
    return [](mlir::MLIRContext *c) {
      return mlir::IntegerType::get(8 * sizeof(int), c);
    };
  } else if constexpr (std::is_same_v<T, int &>) {
    return [](mlir::MLIRContext *c) {
      return fir::ReferenceType::get(
          mlir::IntegerType::get(8 * sizeof(int), c));
    };
  } else if constexpr (std::is_same_v<T, std::int64_t>) {
    return [](mlir::MLIRContext *c) { return mlir::IntegerType::get(64, c); };
  } else if constexpr (std::is_same_v<T, std::int64_t &>) {
    return [](mlir::MLIRContext *c) {
      return fir::ReferenceType::get(mlir::IntegerType::get(64, c));
    };
  } else if constexpr (std::is_same_v<std::decay_t<T>, std::size_t>) {
    return [](mlir::MLIRContext *c) {
      return mlir::IntegerType::get(8 * sizeof(std::size_t), c);
    };
  } else if constexpr (std::is_same_v<T, double>) {
    return [](mlir::MLIRContext *c) { return mlir::FloatType::getF64(c); };
  } else if constexpr (std::is_same_v<T, double &>) {
    return [](mlir::MLIRContext *c) {
      return fir::ReferenceType::get(mlir::FloatType::getF64(c));
    };
  } else if constexpr (std::is_same_v<T, float>) {
    return [](mlir::MLIRContext *c) { return mlir::FloatType::getF32(c); };
  } else if constexpr (std::is_same_v<T, float &>) {
    return [](mlir::MLIRContext *c) {
      return fir::ReferenceType::get(mlir::FloatType::getF32(c));
    };
  } else if constexpr (std::is_same_v<std::decay_t<T>, Iostat>) {
    return [](mlir::MLIRContext *c) {
      return mlir::IntegerType::get(8 * sizeof(Iostat), c);
    };
  } else if constexpr (std::is_same_v<T, bool>) {
    return [](mlir::MLIRContext *c) { return mlir::IntegerType::get(1, c); };
  } else if constexpr (std::is_same_v<T, bool &>) {
    return [](mlir::MLIRContext *c) {
      return fir::ReferenceType::get(mlir::IntegerType::get(1, c));
    };
  } else if constexpr (std::is_same_v<std::decay_t<T>,
                                      runtime::io::IoStatementState *>) {
    return [](mlir::MLIRContext *c) {
      return fir::ReferenceType::get(mlir::IntegerType::get(8, c));
    };
  } else if constexpr (std::is_same_v<std::decay_t<T>, const char *>) {
    return [](mlir::MLIRContext *c) {
      return fir::ReferenceType::get(mlir::IntegerType::get(8, c));
    };
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
  RuntimeTableEntry<RuntimeTableKey<decltype(X)>, decltype(ExpandKey(X))>

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_RT_BUILDER_H_
