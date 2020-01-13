//===-- lib/lower/builder.cc ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/lower/OpBuilder.h"
#include "fir/Dialect/FIROpsSupport.h"
#include "fir/Dialect/FIRType.h"
#include "flang/lower/Bridge.h"
#include "flang/lower/ConvertType.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/StringRef.h"

namespace B = Fortran::lower;
namespace Ev = Fortran::evaluate;
namespace L = llvm;
namespace M = mlir;
namespace Se = Fortran::semantics;

using namespace Fortran;
using namespace Fortran::lower;

M::FuncOp B::createFunction(B::AbstractConverter &converter,
                            llvm::StringRef name, M::FunctionType funcTy) {
  return fir::createFuncOp(converter.getCurrentLocation(),
                           converter.getModuleOp(), name, funcTy);
}

M::FuncOp B::createFunction(M::ModuleOp module, llvm::StringRef name,
                            M::FunctionType funcTy) {
  return fir::createFuncOp(M::UnknownLoc::get(module.getContext()), module,
                           name, funcTy);
}

M::FuncOp B::getNamedFunction(M::ModuleOp module, llvm::StringRef name) {
  return module.lookupSymbol<M::FuncOp>(name);
}

void B::SymMap::addSymbol(Se::SymbolRef symbol, M::Value value) {
  symbolMap.try_emplace(&*symbol, value);
}

M::Value B::SymMap::lookupSymbol(Se::SymbolRef symbol) {
  auto iter{symbolMap.find(&*symbol)};
  return (iter == symbolMap.end()) ? nullptr : iter->second;
}

void B::SymMap::pushShadowSymbol(Se::SymbolRef symbol, M::Value value) {
  // find any existing mapping for symbol
  auto iter{symbolMap.find(&*symbol)};
  const Se::Symbol *sym{nullptr};
  M::Value val;
  // if mapping exists, save it on the shadow stack
  if (iter != symbolMap.end()) {
    sym = iter->first;
    val = iter->second;
    symbolMap.erase(iter);
  }
  shadowStack.emplace_back(sym, val);
  // insert new shadow mapping
  auto r{symbolMap.try_emplace(&*symbol, value)};
  assert(r.second && "unexpected insertion failure");
  (void)r;
}

M::Value B::OpBuilderHandler::getIntegerConstant(M::Type integerType,
                                                 std::int64_t cst) {
  return create<M::ConstantOp>(integerType,
                               builder.getIntegerAttr(integerType, cst));
}
void B::LoopCreator::genLoop(M::Value lb, M::Value ub, M::Value step,
                             const BodyGenerator &bodyGenerator) {
  auto lbi{convertToIndexType(lb)};
  auto ubi{convertToIndexType(ub)};
  auto stepi{convertToIndexType(step)};
  L::SmallVector<M::Value, 1> steps;
  steps.emplace_back(stepi);
  auto loop{create<fir::LoopOp>(lbi, ubi, steps)};
  auto *insPt{builder.getInsertionBlock()};
  builder.setInsertionPointToStart(loop.getBody());
  auto index{loop.getInductionVar()};
  bodyGenerator(*this, index);
  builder.setInsertionPointToEnd(insPt);
}

void B::LoopCreator::genLoop(M::Value lb, M::Value ub,
                             const BodyGenerator &bodyGenerator) {
  auto one{getIntegerConstant(getIndexType(), 1)};
  genLoop(lb, ub, one, bodyGenerator);
}

void B::LoopCreator::genLoop(M::Value count,
                             const BodyGenerator &bodyGenerator) {
  auto indexType{getIndexType()};
  auto zero{getIntegerConstant(indexType, 0)};
  auto one{getIntegerConstant(indexType, 1)};
  genLoop(zero, count, one, bodyGenerator);
}

M::Type B::LoopCreator::getIndexType() {
  return M::IndexType::get(builder.getContext());
}

M::Value B::LoopCreator::convertToIndexType(M::Value integer) {
  auto type{integer.getType()};
  if (type.isa<M::IndexType>()) {
    return integer;
  }
  assert((type.isa<M::IntegerType>() || type.isa<fir::IntType>()) &&
         "expected integer");
  return create<fir::ConvertOp>(getIndexType(), integer);
}

void B::CharacterOpsCreator::genCopy(CharValue &dest, CharValue &src,
                                     M::Value count) {
  auto refType{dest.getReferenceType()};
  // Cast to character sequence reference type for fir::CoordinateOp.
  auto sequenceType{getSequenceRefType(refType)};
  auto destRef{create<fir::ConvertOp>(sequenceType, dest.reference)};
  auto srcRef{create<fir::ConvertOp>(sequenceType, src.reference)};

  LoopCreator{*this}.genLoop(count, [&](OpBuilderHandler &handler,
                                        M::Value index) {
    auto destAddr{handler.create<fir::CoordinateOp>(refType, destRef, index)};
    auto srcAddr{handler.create<fir::CoordinateOp>(refType, srcRef, index)};
    auto val{handler.create<fir::LoadOp>(srcAddr)};
    handler.create<fir::StoreOp>(val, destAddr);
  });
}

void B::CharacterOpsCreator::genPadding(CharValue &str, M::Value from,
                                        M::Value to) {
  auto refType{str.getReferenceType()};
  auto sequenceType{getSequenceRefType(refType)};
  auto strRef{create<fir::ConvertOp>(sequenceType, str.reference)};
  auto blank{getBlankConstant(str.getCharacterType())};

  LoopCreator{*this}.genLoop(
      from, to, [&](OpBuilderHandler &handler, M::Value index) {
        auto strAddr{handler.create<fir::CoordinateOp>(refType, strRef, index)};
        handler.create<fir::StoreOp>(blank, strAddr);
      });
}

M::Value B::CharacterOpsCreator::getBlankConstant(fir::CharacterType type) {
  auto byteTy{M::IntegerType::get(8, builder.getContext())};
  auto asciiSpace{getIntegerConstant(byteTy, 0x20)};
  return create<fir::ConvertOp>(type, asciiSpace);
}

B::CharacterOpsCreator::CharValue
B::CharacterOpsCreator::createTemp(fir::CharacterType type, M::Value len) {
  // FIXME Does this need to be emitted somewhere safe ?
  // convert-expr.cc generates alloca at the beginning of the mlir block.
  return CharValue{create<fir::AllocaOp>(type, len), len};
}

fir::ReferenceType B::CharacterOpsCreator::CharValue::getReferenceType() {
  auto type{reference.getType().dyn_cast<fir::ReferenceType>()};
  assert(type && "expected reference type");
  return type;
}

fir::CharacterType B::CharacterOpsCreator::CharValue::getCharacterType() {
  auto type{getReferenceType().getEleTy().dyn_cast<fir::CharacterType>()};
  assert(type && "expected character type");
  return type;
}
