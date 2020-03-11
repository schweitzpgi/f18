//===-- OpBuilder.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/FIRBuilder.h"
#include "flang/Lower/Bridge.h"
#include "flang/Lower/ConvertType.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Semantics/symbol.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

void Fortran::lower::SymMap::addSymbol(Fortran::semantics::SymbolRef symbol,
                                       mlir::Value value) {
  symbolMap.try_emplace(&*symbol, value);
}

mlir::Value
Fortran::lower::SymMap::lookupSymbol(Fortran::semantics::SymbolRef symbol) {
  auto iter{symbolMap.find(&*symbol)};
  return (iter == symbolMap.end()) ? nullptr : iter->second;
}

mlir::FuncOp Fortran::lower::FirOpBuilder::createFunction(
    mlir::Location loc, mlir::ModuleOp module, llvm::StringRef name,
    mlir::FunctionType ty) {
  return fir::createFuncOp(loc, module, name, ty);
}

mlir::FuncOp
Fortran::lower::FirOpBuilder::getNamedFunction(mlir::ModuleOp modOp,
                                               llvm::StringRef name) {
  return modOp.lookupSymbol<mlir::FuncOp>(name);
}

mlir::Value
Fortran::lower::FirOpBuilder::createIntegerConstant(mlir::Type intType,
                                                    std::int64_t cst) {
  return createHere<mlir::ConstantOp>(intType, getIntegerAttr(intType, cst));
}

/// Create a temporary variable
/// `symbol` will be nullptr for an anonymous temporary
mlir::Value Fortran::lower::FirOpBuilder::createTemporary(
    mlir::Location loc, Fortran::lower::SymMap &symMap, mlir::Type type,
    llvm::ArrayRef<mlir::Value> shape, const Fortran::semantics::Symbol *symbol,
    llvm::StringRef name) {
  if (symbol)
    if (auto val = symMap.lookupSymbol(*symbol)) {
      if (auto op = val.getDefiningOp())
        return op->getResult(0);
      return val;
    }

  auto insPt = saveInsertionPoint();
  setInsertionPointToStart(getEntryBlock());
  fir::AllocaOp ae;
  assert(!type.isa<fir::ReferenceType>() && "cannot be a reference");
  if (symbol) {
    ae = create<fir::AllocaOp>(loc, type, symbol->name().ToString(), llvm::None,
                               shape);
    symMap.addSymbol(*symbol, ae);
  } else {
    ae = create<fir::AllocaOp>(loc, type, name, llvm::None, shape);
  }
  restoreInsertionPoint(insPt);
  return ae;
}

//===----------------------------------------------------------------------===//
// LoopOp builder
//===----------------------------------------------------------------------===//

void Fortran::lower::FirOpBuilder::createLoop(
    mlir::Value lb, mlir::Value ub, mlir::Value step,
    const BodyGenerator &bodyGenerator) {
  auto lbi = convertToIndexType(lb);
  auto ubi = convertToIndexType(ub);
  assert(step && "step must be an actual Value");
  auto inc = convertToIndexType(step);
  auto loop = createHere<fir::LoopOp>(lbi, ubi, inc);
  auto *insPt = getInsertionBlock();
  setInsertionPointToStart(loop.getBody());
  auto index = loop.getInductionVar();
  bodyGenerator(*this, index);
  setInsertionPointToEnd(insPt);
}

void Fortran::lower::FirOpBuilder::createLoop(
    mlir::Value lb, mlir::Value ub, const BodyGenerator &bodyGenerator) {
  createLoop(lb, ub, createIntegerConstant(getIndexType(), 1), bodyGenerator);
}

void Fortran::lower::FirOpBuilder::createLoop(
    mlir::Value count, const BodyGenerator &bodyGenerator) {
  auto indexType = getIndexType();
  auto zero = createIntegerConstant(indexType, 0);
  auto one = createIntegerConstant(indexType, 1);
  createLoop(zero, count, one, bodyGenerator);
}

mlir::Value
Fortran::lower::FirOpBuilder::convertToIndexType(mlir::Value integer) {
  // abort now if not an integral type
  fir::verifyIntegralType(integer.getType());
  return createHere<fir::ConvertOp>(getIndexType(), integer);
}

//===----------------------------------------------------------------------===//
// CharacterOpsBuilder implementation
//===----------------------------------------------------------------------===//

template <typename T>
void Fortran::lower::CharacterOpsBuilder<T>::createCopy(
    Fortran::lower::CharValue &dest, const Fortran::lower::CharValue &src,
    mlir::Value count) {
  auto &b = impl();
  auto refType = b.getReferenceType(dest);
  // Cast to character sequence reference type for fir::CoordinateOp.
  auto seqTy = getSequenceRefType(refType);
  auto destRef = b.template createHere<fir::ConvertOp>(seqTy, dest.buffer);
  auto srcRef = b.template createHere<fir::ConvertOp>(seqTy, src.buffer);

  b.createLoop(count, [&](Fortran::lower::FirOpBuilder &b, mlir::Value index) {
    auto destAddr = b.createHere<fir::CoordinateOp>(refType, destRef, index);
    auto srcAddr = b.createHere<fir::CoordinateOp>(refType, srcRef, index);
    auto val = b.createHere<fir::LoadOp>(srcAddr);
    b.createHere<fir::StoreOp>(val, destAddr);
  });
}
template void
Fortran::lower::CharacterOpsBuilder<Fortran::lower::FirOpBuilder>::createCopy(
    Fortran::lower::CharValue &, const Fortran::lower::CharValue &,
    mlir::Value);

template <typename T>
void Fortran::lower::CharacterOpsBuilder<T>::createPadding(
    Fortran::lower::CharValue &str, mlir::Value lower, mlir::Value upper) {
  auto &b = impl();
  auto loc = b.getLoc();
  auto refType = b.getReferenceType(str);
  auto seqTy = getSequenceRefType(refType);
  auto strRef = b.template createHere<fir::ConvertOp>(seqTy, str.buffer);
  auto blank = createBlankConstant(b.getCharacterType(str));

  b.createLoop(lower, upper, [&](FirOpBuilder &b, mlir::Value index) {
    auto strAddr = b.create<fir::CoordinateOp>(loc, refType, strRef, index);
    b.create<fir::StoreOp>(loc, blank, strAddr);
  });
}
template void Fortran::lower::CharacterOpsBuilder<
    Fortran::lower::FirOpBuilder>::createPadding(Fortran::lower::CharValue &,
                                                 mlir::Value, mlir::Value);

template <typename T>
mlir::Value Fortran::lower::CharacterOpsBuilder<T>::createBlankConstant(
    fir::CharacterType type) {
  auto &b = impl();
  auto byteTy = b.getIntegerType(8);
  auto asciiSpace = b.createIntegerConstant(byteTy, ' ');
  return b.template createHere<fir::ConvertOp>(type, asciiSpace);
}
template mlir::Value Fortran::lower::CharacterOpsBuilder<
    Fortran::lower::FirOpBuilder>::createBlankConstant(fir::CharacterType);

template <typename T>
Fortran::lower::CharValue
Fortran::lower::CharacterOpsBuilder<T>::createCharacterTemp(
    fir::CharacterType type, mlir::Value len) {
  auto buffer = impl().createTemporary(type, {len});
  return CharValue{buffer, len};
}
template Fortran::lower::CharValue Fortran::lower::CharacterOpsBuilder<
    Fortran::lower::FirOpBuilder>::createCharacterTemp(fir::CharacterType,
                                                       mlir::Value);

template <typename T>
fir::ReferenceType Fortran::lower::CharacterOpsBuilder<T>::getReferenceType(
    const Fortran::lower::CharValue &pair) {
  auto refTy = pair.buffer.getType();
  if (auto type = refTy.template dyn_cast<fir::ReferenceType>())
    return type;
  llvm_unreachable("expected reference type");
  return {};
}
template fir::ReferenceType
Fortran::lower::CharacterOpsBuilder<Fortran::lower::FirOpBuilder>::
    getReferenceType(const Fortran::lower::CharValue &);

template <typename T>
fir::CharacterType Fortran::lower::CharacterOpsBuilder<T>::getCharacterType(
    const Fortran::lower::CharValue &pair) {
  auto eleTy = getReferenceType(pair).getEleTy();
  if (auto type = eleTy.template dyn_cast<fir::CharacterType>())
    return type;
  llvm_unreachable("expected character type");
  return {};
}
template fir::CharacterType
Fortran::lower::CharacterOpsBuilder<Fortran::lower::FirOpBuilder>::
    getCharacterType(const Fortran::lower::CharValue &);

//===----------------------------------------------------------------------===//
// ComplexOpsBuilder implementation
//===----------------------------------------------------------------------===//

template <typename T>
mlir::Type Fortran::lower::ComplexOpsBuilder<T>::getComplexPartType(
    fir::KindTy complexKind) {
  return convertReal(impl().getContext(), complexKind);
}
template mlir::Type Fortran::lower::ComplexOpsBuilder<
    Fortran::lower::FirOpBuilder>::getComplexPartType(fir::KindTy);

template <typename T>
mlir::Type Fortran::lower::ComplexOpsBuilder<T>::getComplexPartType(
    mlir::Type complexType) {
  return getComplexPartType(complexType.cast<fir::CplxType>().getFKind());
}
template mlir::Type Fortran::lower::ComplexOpsBuilder<
    Fortran::lower::FirOpBuilder>::getComplexPartType(mlir::Type);

template <typename T>
mlir::Type
Fortran::lower::ComplexOpsBuilder<T>::getComplexPartType(mlir::Value cplx) {
  return getComplexPartType(cplx.getType());
}
template mlir::Type Fortran::lower::ComplexOpsBuilder<
    Fortran::lower::FirOpBuilder>::getComplexPartType(mlir::Value);

template <typename T>
mlir::Value Fortran::lower::ComplexOpsBuilder<T>::createComplex(
    fir::KindTy kind, mlir::Value real, mlir::Value imag) {
  auto complexTy = fir::CplxType::get(impl().getContext(), kind);
  mlir::Value und = impl().template createHere<fir::UndefOp>(complexTy);
  return insert<Part::Imag>(insert<Part::Real>(und, real), imag);
}
template mlir::Value Fortran::lower::ComplexOpsBuilder<
    Fortran::lower::FirOpBuilder>::createComplex(fir::KindTy, mlir::Value,
                                                 mlir::Value);

template <typename T>
mlir::Value Fortran::lower::ComplexOpsBuilder<T>::createComplexCompare(
    mlir::Value cplx1, mlir::Value cplx2, bool eq) {
  auto real1 = extract<Part::Real>(cplx1);
  auto real2 = extract<Part::Real>(cplx2);
  auto imag1 = extract<Part::Imag>(cplx1);
  auto imag2 = extract<Part::Imag>(cplx2);

  mlir::CmpFPredicate predicate =
      eq ? mlir::CmpFPredicate::UEQ : mlir::CmpFPredicate::UNE;
  auto &b = impl();
  mlir::Value realCmp =
      b.template createHere<mlir::CmpFOp>(predicate, real1, real2);
  mlir::Value imagCmp =
      b.template createHere<mlir::CmpFOp>(predicate, imag1, imag2);

  return eq ? b.template createHere<mlir::AndOp>(realCmp, imagCmp).getResult()
            : b.template createHere<mlir::OrOp>(realCmp, imagCmp).getResult();
}
