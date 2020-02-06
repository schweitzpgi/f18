//===-- OpBuilder.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/lower/OpBuilder.h"
#include "NSAliases.h"
#include "flang/lower/Bridge.h"
#include "flang/lower/ConvertType.h"
#include "flang/optimizer/Dialect/FIROpsSupport.h"
#include "flang/optimizer/Dialect/FIRType.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/StringRef.h"

using namespace Fortran;
using namespace Fortran::lower;

M::FuncOp Br::createFunction(Br::AbstractConverter &converter,
                             L::StringRef name, M::FunctionType funcTy) {
  return fir::createFuncOp(converter.getCurrentLocation(),
                           converter.getModuleOp(), name, funcTy);
}

M::FuncOp Br::createFunction(M::ModuleOp module, L::StringRef name,
                             M::FunctionType funcTy) {
  return fir::createFuncOp(M::UnknownLoc::get(module.getContext()), module,
                           name, funcTy);
}

M::FuncOp Br::getNamedFunction(M::ModuleOp module, L::StringRef name) {
  return module.lookupSymbol<M::FuncOp>(name);
}

void Br::SymMap::addSymbol(Se::SymbolRef symbol, M::Value value) {
  symbolMap.try_emplace(&*symbol, value);
}

M::Value Br::SymMap::lookupSymbol(Se::SymbolRef symbol) {
  auto iter{symbolMap.find(&*symbol)};
  return (iter == symbolMap.end()) ? nullptr : iter->second;
}

void Br::SymMap::pushShadowSymbol(Se::SymbolRef symbol, M::Value value) {
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

M::Value Br::OpBuilderWrapper::createIntegerConstant(M::Type integerType,
                                                     std::int64_t cst) {
  return create<M::ConstantOp>(integerType,
                               builder.getIntegerAttr(integerType, cst));
}

// LoopBuilder implementation

void Br::LoopBuilder::createLoop(M::Value lb, M::Value ub, M::Value step,
                                 const BodyGenerator &bodyGenerator) {
  auto lbi{convertToIndexType(lb)};
  auto ubi{convertToIndexType(ub)};
  L::SmallVector<M::Value, 1> steps;
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

void Br::LoopBuilder::createLoop(M::Value lb, M::Value ub,
                                 const BodyGenerator &bodyGenerator) {
  auto one{createIntegerConstant(getIndexType(), 1)};
  createLoop(lb, ub, one, bodyGenerator);
}

void Br::LoopBuilder::createLoop(M::Value count,
                                 const BodyGenerator &bodyGenerator) {
  auto indexType{getIndexType()};
  auto zero{createIntegerConstant(indexType, 0)};
  auto one{createIntegerConstant(indexType, 1)};
  createLoop(zero, count, one, bodyGenerator);
}

M::Type Br::LoopBuilder::getIndexType() {
  return M::IndexType::get(builder.getContext());
}

M::Value Br::LoopBuilder::convertToIndexType(M::Value integer) {
  auto type{integer.getType()};
  if (type.isa<M::IndexType>()) {
    return integer;
  }
  assert((type.isa<M::IntegerType>() || type.isa<fir::IntType>()) &&
         "expected integer");
  return create<fir::ConvertOp>(getIndexType(), integer);
}

// CharacterOpsBuilder implementation

mlir::Value Br::CharacterOpsBuilder::createExtractCharAt(AddressableValue &str,
                                                         mlir::Value index) {
  if (str.isRef()) {
    auto addr{
        create<fir::CoordinateOp>(str.getReferenceType(), str.value, index)};
    return create<fir::LoadOp>(addr);
  }
  return create<fir::ExtractValueOp>(str.getCharacterType(), str.value, index);
}

void Br::CharacterOpsBuilder::createStoreCharAt(AddressableRef &str,
                                                mlir::Value index,
                                                mlir::Value c) {
  auto addr{
      create<fir::CoordinateOp>(str.getReferenceType(), str.value, index)};
  create<fir::StoreOp>(c, addr);
}

Br::CharacterOpsBuilder::AddressableValue
Br::CharacterOpsBuilder::convertToAddressableValue(CharValue &str) {
  auto newValue{str.value};
  mlir::Type seqType{};
  if (str.isRef()) {
    seqType = str.getSequenceRefType();
  } else {
    seqType = str.getSequenceType();
  }
  if (newValue.getType() != seqType) {
    newValue = create<fir::ConvertOp>(seqType, newValue);
  }
  return AddressableValue{newValue, str.len};
}

Br::CharacterOpsBuilder::AddressableRef
Br::CharacterOpsBuilder::convertToAddressableRef(CharRef &str) {
  auto newValue{create<fir::ConvertOp>(str.getSequenceRefType(), str.value)};
  return AddressableRef{newValue, str.len};
}

void Br::CharacterOpsBuilder::createCopy(CharRef &dest, CharValue &src,
                                         M::Value count) {
  auto destArray{convertToAddressableRef(dest)};
  auto srcArray{convertToAddressableValue(src)};

  LoopBuilder{*this}.createLoop(
      count, [&](OpBuilderWrapper &handler, M::Value index) {
        CharacterOpsBuilder charHanlder{handler};
        auto charVal{charHanlder.createExtractCharAt(srcArray, index)};
        charHanlder.createStoreCharAt(destArray, index, charVal);
      });
}

void Br::CharacterOpsBuilder::createPadding(CharRef &str, M::Value lower,
                                            M::Value upper) {
  auto strArray{convertToAddressableRef(str)};
  auto blank{createBlankConstant(str.getCharacterType())};
  LoopBuilder{*this}.createLoop(
      lower, upper, [&](OpBuilderWrapper &handler, M::Value index) {
        CharacterOpsBuilder{handler}.createStoreCharAt(strArray, index, blank);
      });
}

void Br::CharacterOpsBuilder::createAssign(CharRef &lhs, CharValue &rhs) {
  CharValue safe_rhs{rhs};
  if (rhs.isRef()) {
    // If rhs is in memory, always assumes rhs might overlap with lhs
    // in a way that require a temp for the copy. That can be optimize later.
    auto temp{createTemp(rhs.getCharacterType(), rhs.len)};
    createCopy(temp, rhs, rhs.len);
    safe_rhs = temp;
  }

  // Copy the minimum of the lhs and rhs lengths and pad the lhs remainder
  auto cmpLen{create<M::CmpIOp>(M::CmpIPredicate::slt, lhs.len, rhs.len)};
  auto copyCount{create<M::SelectOp>(cmpLen, lhs.len, rhs.len)};
  createCopy(lhs, safe_rhs, copyCount);
  createPadding(lhs, copyCount, lhs.len);
}

M::Value Br::CharacterOpsBuilder::createBlankConstant(fir::CharacterType type) {
  auto byteTy{M::IntegerType::get(8, builder.getContext())};
  auto asciiSpace{createIntegerConstant(byteTy, 0x20)};
  return create<fir::ConvertOp>(type, asciiSpace);
}

Br::CharacterOpsBuilder::CharRef
Br::CharacterOpsBuilder::createTemp(fir::CharacterType type, M::Value len) {
  // FIXME Does this need to be emitted somewhere safe ?
  // convert-expr.cc generates alloca at the beginning of the mlir block.
  return CharRef{create<fir::AllocaOp>(type, len), len};
}

fir::CharacterType
Br::CharacterOpsBuilder::CharValue::getCharacterType() const {
  auto type{value.getType()};
  if (auto refType{type.dyn_cast<fir::ReferenceType>()}) {
    type = refType.getEleTy();
  }
  if (auto seqType{type.dyn_cast<fir::SequenceType>()}) {
    type = seqType.getEleTy();
  }
  if (auto charType{type.dyn_cast<fir::CharacterType>()}) {
    return charType;
  }
  llvm_unreachable("Invalid character value type");
  return mlir::Type{}.dyn_cast<fir::CharacterType>();
}

// If the CharValue is already a sequence type, we do not want to
// lose the shape that might be important, plus lowering to LLVM,
// there would be errors if converting between arrays of different
// sizes.
fir::SequenceType
extractSequenceType(const Br::CharacterOpsBuilder::CharValue &str) {
  auto type{str.value.getType()};
  if (auto refType{type.dyn_cast<fir::ReferenceType>()}) {
    type = refType.getEleTy();
  }
  if (auto seqType{type.dyn_cast<fir::SequenceType>()}) {
    return seqType;
  }
  // Return fir.array<? x ...> if str is not a sequence type.
  // TODO: Could be improved if len is a constant and we can get its value.
  fir::SequenceType::Shape shape{fir::SequenceType::getUnknownExtent()};
  return fir::SequenceType::get(shape, str.getCharacterType());
}

fir::SequenceType Br::CharacterOpsBuilder::CharValue::getSequenceType() const {
  return extractSequenceType(*this);
}

fir::ReferenceType
Br::CharacterOpsBuilder::CharValue::getReferenceType() const {
  return fir::ReferenceType::get(getCharacterType());
}

fir::ReferenceType
Br::CharacterOpsBuilder::CharValue::getSequenceRefType() const {
  return fir::ReferenceType::get(extractSequenceType(*this));
}

bool Br::CharacterOpsBuilder::CharValue::isRef() const {
  return value.getType().isa<fir::ReferenceType>();
}

// ComplexOpsBuilder implementation

M::Type Br::ComplexOpsBuilder::getComplexPartType(fir::KindTy complexKind) {
  return convertReal(builder.getContext(), complexKind);
}
M::Type Br::ComplexOpsBuilder::getComplexPartType(M::Type complexType) {
  return getComplexPartType(complexType.cast<fir::CplxType>().getFKind());
}
M::Type Br::ComplexOpsBuilder::getComplexPartType(M::Value cplx) {
  assert(cplx != nullptr);
  return getComplexPartType(cplx.getType());
}

M::Value Br::ComplexOpsBuilder::createComplex(fir::KindTy kind, M::Value real,
                                              M::Value imag) {
  M::Type complexTy{fir::CplxType::get(builder.getContext(), kind)};
  M::Value und{create<fir::UndefOp>(complexTy)};
  return insert<Part::Imag>(insert<Part::Real>(und, real), imag);
}

using CplxPart = Br::ComplexOpsBuilder::Part;
template <CplxPart partId>
M::Value Br::ComplexOpsBuilder::createPartId() {
  auto type{M::IntegerType::get(32, builder.getContext())};
  return createIntegerConstant(type, static_cast<int>(partId));
}

template <CplxPart partId>
M::Value Br::ComplexOpsBuilder::extract(M::Value cplx) {
  return create<fir::ExtractValueOp>(getComplexPartType(cplx), cplx,
                                     createPartId<partId>());
}
template M::Value Br::ComplexOpsBuilder::extract<CplxPart::Real>(M::Value);
template M::Value Br::ComplexOpsBuilder::extract<CplxPart::Imag>(M::Value);

template <CplxPart partId>
M::Value Br::ComplexOpsBuilder::insert(M::Value cplx, M::Value part) {
  assert(cplx != nullptr);
  return create<fir::InsertValueOp>(cplx.getType(), cplx, part,
                                    createPartId<partId>());
}
template M::Value Br::ComplexOpsBuilder::insert<CplxPart::Real>(M::Value,
                                                                M::Value);
template M::Value Br::ComplexOpsBuilder::insert<CplxPart::Imag>(M::Value,
                                                                M::Value);

M::Value Br::ComplexOpsBuilder::extractComplexPart(M::Value cplx,
                                                   bool isImagPart) {
  return isImagPart ? extract<Part::Imag>(cplx) : extract<Part::Real>(cplx);
}

M::Value Br::ComplexOpsBuilder::insertComplexPart(M::Value cplx, M::Value part,
                                                  bool isImagPart) {
  return isImagPart ? insert<Part::Imag>(cplx, part)
                    : insert<Part::Real>(cplx, part);
}

M::Value Br::ComplexOpsBuilder::createComplexCompare(M::Value cplx1,
                                                     M::Value cplx2, bool eq) {
  M::Value real1{extract<Part::Real>(cplx1)};
  M::Value real2{extract<Part::Real>(cplx2)};
  M::Value imag1{extract<Part::Imag>(cplx1)};
  M::Value imag2{extract<Part::Imag>(cplx2)};

  M::CmpFPredicate predicate{eq ? M::CmpFPredicate::UEQ
                                : M::CmpFPredicate::UNE};
  M::Value realCmp{create<M::CmpFOp>(predicate, real1, real2)};
  M::Value imagCmp{create<M::CmpFOp>(predicate, imag1, imag2)};

  return eq ? create<M::AndOp>(realCmp, imagCmp).getResult()
            : create<M::OrOp>(realCmp, imagCmp).getResult();
}
