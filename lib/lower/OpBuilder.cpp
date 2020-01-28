//===-- OpBuilder.cc ------------------------------------------------------===//
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

namespace Br = Fortran::lower;
namespace Ev = Fortran::evaluate;
namespace Se = Fortran::semantics;

using namespace Fortran;
using namespace Fortran::lower;

mlir::FuncOp Br::createFunction(Br::AbstractConverter &converter,
                                llvm::StringRef name,
                                mlir::FunctionType funcTy) {
  return fir::createFuncOp(converter.getCurrentLocation(),
                           converter.getModuleOp(), name, funcTy);
}

mlir::FuncOp Br::createFunction(mlir::ModuleOp module, llvm::StringRef name,
                                mlir::FunctionType funcTy) {
  return fir::createFuncOp(mlir::UnknownLoc::get(module.getContext()), module,
                           name, funcTy);
}

mlir::FuncOp Br::getNamedFunction(mlir::ModuleOp module, llvm::StringRef name) {
  return module.lookupSymbol<mlir::FuncOp>(name);
}

void Br::SymMap::addSymbol(Se::SymbolRef symbol, mlir::Value value) {
  symbolMap.try_emplace(&*symbol, value);
}

mlir::Value Br::SymMap::lookupSymbol(Se::SymbolRef symbol) {
  auto iter{symbolMap.find(&*symbol)};
  return (iter == symbolMap.end()) ? nullptr : iter->second;
}

void Br::SymMap::pushShadowSymbol(Se::SymbolRef symbol, mlir::Value value) {
  // find any existing mapping for symbol
  auto iter{symbolMap.find(&*symbol)};
  const Se::Symbol *sym{nullptr};
  mlir::Value val;
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

mlir::Value Br::OpBuilderWrapper::createIntegerConstant(mlir::Type integerType,
                                                        std::int64_t cst) {
  return create<mlir::ConstantOp>(integerType,
                                  builder.getIntegerAttr(integerType, cst));
}

// LoopBuilder implementation

void Br::LoopBuilder::createLoop(mlir::Value lb, mlir::Value ub,
                                 mlir::Value step,
                                 const BodyGenerator &bodyGenerator) {
  auto lbi{convertToIndexType(lb)};
  auto ubi{convertToIndexType(ub)};
  llvm::SmallVector<mlir::Value, 1> steps;
  if (step) {
    auto stepi{convertToIndexType(step)};
    steps.emplace_back(stepi);
  }
  auto loop{create<fir::LoopOp>(lbi, ubi, steps)};
  auto *insPt{builder.getInsertionBlock()};
  builder.setInsertionPointToStart(loop.getBody());
  auto index{loop.getInductionVar()};
  bodyGenerator(*this, index);
  builder.setInsertionPointToEnd(insPt);
}

void Br::LoopBuilder::createLoop(mlir::Value lb, mlir::Value ub,
                                 const BodyGenerator &bodyGenerator) {
  auto one{createIntegerConstant(getIndexType(), 1)};
  createLoop(lb, ub, one, bodyGenerator);
}

void Br::LoopBuilder::createLoop(mlir::Value count,
                                 const BodyGenerator &bodyGenerator) {
  auto indexType{getIndexType()};
  auto zero{createIntegerConstant(indexType, 0)};
  auto one{createIntegerConstant(indexType, 1)};
  createLoop(zero, count, one, bodyGenerator);
}

mlir::Type Br::LoopBuilder::getIndexType() {
  return mlir::IndexType::get(builder.getContext());
}

mlir::Value Br::LoopBuilder::convertToIndexType(mlir::Value integer) {
  auto type{integer.getType()};
  if (type.isa<mlir::IndexType>()) {
    return integer;
  }
  assert((type.isa<mlir::IntegerType>() || type.isa<fir::IntType>()) &&
         "expected integer");
  return create<fir::ConvertOp>(getIndexType(), integer);
}

// CharacterOpsBuilder implementation

void Br::CharacterOpsBuilder::createCopy(CharValue &dest, CharValue &src,
                                         mlir::Value count) {
  auto refType{dest.getReferenceType()};
  // Cast to character sequence reference type for fir::CoordinateOp.
  auto sequenceType{getSequenceRefType(refType)};
  auto destRef{create<fir::ConvertOp>(sequenceType, dest.reference)};
  auto srcRef{create<fir::ConvertOp>(sequenceType, src.reference)};

  LoopBuilder{*this}.createLoop(count, [&](OpBuilderWrapper &handler,
                                           mlir::Value index) {
    auto destAddr{handler.create<fir::CoordinateOp>(refType, destRef, index)};
    auto srcAddr{handler.create<fir::CoordinateOp>(refType, srcRef, index)};
    auto val{handler.create<fir::LoadOp>(srcAddr)};
    handler.create<fir::StoreOp>(val, destAddr);
  });
}

void Br::CharacterOpsBuilder::createPadding(CharValue &str, mlir::Value lower,
                                            mlir::Value upper) {
  auto refType{str.getReferenceType()};
  auto sequenceType{getSequenceRefType(refType)};
  auto strRef{create<fir::ConvertOp>(sequenceType, str.reference)};
  auto blank{createBlankConstant(str.getCharacterType())};

  LoopBuilder{*this}.createLoop(
      lower, upper, [&](OpBuilderWrapper &handler, mlir::Value index) {
        auto strAddr{handler.create<fir::CoordinateOp>(refType, strRef, index)};
        handler.create<fir::StoreOp>(blank, strAddr);
      });
}

mlir::Value
Br::CharacterOpsBuilder::createBlankConstant(fir::CharacterType type) {
  auto byteTy{mlir::IntegerType::get(8, builder.getContext())};
  auto asciiSpace{createIntegerConstant(byteTy, 0x20)};
  return create<fir::ConvertOp>(type, asciiSpace);
}

Br::CharacterOpsBuilder::CharValue
Br::CharacterOpsBuilder::createTemp(fir::CharacterType type, mlir::Value len) {
  // FIXME Does this need to be emitted somewhere safe ?
  // convert-expr.cc generates alloca at the beginning of the mlir block.
  return CharValue{create<fir::AllocaOp>(type, len), len};
}

fir::ReferenceType Br::CharacterOpsBuilder::CharValue::getReferenceType() {
  auto type{reference.getType().dyn_cast<fir::ReferenceType>()};
  assert(type && "expected reference type");
  return type;
}

fir::CharacterType Br::CharacterOpsBuilder::CharValue::getCharacterType() {
  auto type{getReferenceType().getEleTy().dyn_cast<fir::CharacterType>()};
  assert(type && "expected character type");
  return type;
}

// ComplexOpsBuilder implementation

mlir::Type Br::ComplexOpsBuilder::getComplexPartType(fir::KindTy complexKind) {
  return convertReal(builder.getContext(), complexKind);
}
mlir::Type Br::ComplexOpsBuilder::getComplexPartType(mlir::Type complexType) {
  return getComplexPartType(complexType.cast<fir::CplxType>().getFKind());
}
mlir::Type Br::ComplexOpsBuilder::getComplexPartType(mlir::Value cplx) {
  assert(cplx != nullptr);
  return getComplexPartType(cplx.getType());
}

mlir::Value Br::ComplexOpsBuilder::createComplex(fir::KindTy kind,
                                                 mlir::Value real,
                                                 mlir::Value imag) {
  mlir::Type complexTy{fir::CplxType::get(builder.getContext(), kind)};
  mlir::Value und{create<fir::UndefOp>(complexTy)};
  return insert<Part::Imag>(insert<Part::Real>(und, real), imag);
}

using CplxPart = Br::ComplexOpsBuilder::Part;
template <CplxPart partId>
mlir::Value Br::ComplexOpsBuilder::createPartId() {
  auto type{mlir::IntegerType::get(32, builder.getContext())};
  return createIntegerConstant(type, static_cast<int>(partId));
}

template <CplxPart partId>
mlir::Value Br::ComplexOpsBuilder::extract(mlir::Value cplx) {
  return create<fir::ExtractValueOp>(getComplexPartType(cplx), cplx,
                                     createPartId<partId>());
}
template mlir::Value
    Br::ComplexOpsBuilder::extract<CplxPart::Real>(mlir::Value);
template mlir::Value
    Br::ComplexOpsBuilder::extract<CplxPart::Imag>(mlir::Value);

template <CplxPart partId>
mlir::Value Br::ComplexOpsBuilder::insert(mlir::Value cplx, mlir::Value part) {
  assert(cplx != nullptr);
  return create<fir::InsertValueOp>(cplx.getType(), cplx, part,
                                    createPartId<partId>());
}
template mlir::Value Br::ComplexOpsBuilder::insert<CplxPart::Real>(mlir::Value,
                                                                   mlir::Value);
template mlir::Value Br::ComplexOpsBuilder::insert<CplxPart::Imag>(mlir::Value,
                                                                   mlir::Value);

mlir::Value Br::ComplexOpsBuilder::extractComplexPart(mlir::Value cplx,
                                                      bool isImagPart) {
  return isImagPart ? extract<Part::Imag>(cplx) : extract<Part::Real>(cplx);
}

mlir::Value Br::ComplexOpsBuilder::insertComplexPart(mlir::Value cplx,
                                                     mlir::Value part,
                                                     bool isImagPart) {
  return isImagPart ? insert<Part::Imag>(cplx, part)
                    : insert<Part::Real>(cplx, part);
}

mlir::Value Br::ComplexOpsBuilder::createComplexCompare(mlir::Value cplx1,
                                                        mlir::Value cplx2,
                                                        bool eq) {
  mlir::Value real1{extract<Part::Real>(cplx1)};
  mlir::Value real2{extract<Part::Real>(cplx2)};
  mlir::Value imag1{extract<Part::Imag>(cplx1)};
  mlir::Value imag2{extract<Part::Imag>(cplx2)};

  mlir::CmpFPredicate predicate{eq ? mlir::CmpFPredicate::UEQ
                                   : mlir::CmpFPredicate::UNE};
  mlir::Value realCmp{create<mlir::CmpFOp>(predicate, real1, real2)};
  mlir::Value imagCmp{create<mlir::CmpFOp>(predicate, imag1, imag2)};

  return eq ? create<mlir::AndOp>(realCmp, imagCmp).getResult()
            : create<mlir::OrOp>(realCmp, imagCmp).getResult();
}
