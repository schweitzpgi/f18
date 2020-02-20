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

namespace {
/// CharacterOpsBuilder implementation
struct CharacterOpsBuilderImpl : public OpBuilderWrapper {
  /// Interchange format to avoid inserting unbox/embox everywhere while
  /// evaluating character expressions.

  /// Opened char box
  struct Char {
    /// Get fir.char<kind> type with the same kind as inside ref.
    fir::CharacterType getCharacterType() const {
      auto type{ref.getType().cast<fir::ReferenceType>().getEleTy()};
      if (auto seqType{type.dyn_cast<fir::SequenceType>()}) {
        type = seqType.getEleTy();
      }
      if (auto charType{type.dyn_cast<fir::CharacterType>()}) {
        return charType;
      }
      llvm_unreachable("Invalid character value type");
      return mlir::Type{}.dyn_cast<fir::CharacterType>();
    }
    /// Get fir.ref<fir.char<kind>> type.
    fir::ReferenceType getReferenceType() const {
      return fir::ReferenceType::get(getCharacterType());
    }
    /// Get fir.ref<fir.array<len x fir.char<kind>>> type.
    fir::ReferenceType getSequenceRefType() const {
      // If the Char is already a sequence type, we do not want to
      // lose the shape that might be important, plus lowering to LLVM,
      // there would be errors if converting between arrays of different
      // sizes.
      auto refType{ref.getType().cast<fir::ReferenceType>()};
      if (refType.getEleTy().isa<fir::SequenceType>()) {
        return refType;
      }
      // Return fir.array<? x ...> if str is not a sequence type.
      // TODO: Could be improved if len is a constant and we can get its value.
      fir::SequenceType::Shape shape{fir::SequenceType::getUnknownExtent()};
      auto seqType{fir::SequenceType::get(shape, getCharacterType())};
      return fir::ReferenceType::get(seqType);
    }

    /// Value is fir.ref<fir.array<len x fir.char<kind>>> type.
    /// It can be used fir.coordinate_of.
    bool isAddressable() const {
      auto refType{ref.getType().cast<fir::ReferenceType>()};
      return refType.getEleTy().isa<fir::SequenceType>();
    };

    /// ref must be of type:
    /// - fir.ref<fir.char<kind>>
    /// - fir.ref<fir.array<len x fir.char<kind>>> type.
    mlir::Value ref;
    mlir::Value len;
  };

  /// Just a helper type to avoid forgetting casting char ref to the right type
  /// before fir loops.
  struct AddressableChar : Char {
    /// ref must be of type: fir.ref<fir.array<len x fir.char<kind>>>.
    bool isAddressable() const { return true; }
  };

  AddressableChar convertToAddressableChar(Char str) {
    if (str.isAddressable()) {
      return {str};
    }
    auto newRef{create<fir::ConvertOp>(str.getSequenceRefType(), str.ref)};
    return {newRef, str.len};
  }

  Char createUnboxChar(mlir::Value boxChar) {
    auto boxCharType{boxChar.getType().cast<fir::BoxCharType>()};
    auto refType{fir::ReferenceType::get(boxCharType.getEleTy())};
    auto lenType{M::IntegerType::get(64, builder.getContext())};
    auto unboxed{create<fir::UnboxCharOp>(refType, lenType, boxChar)};
    return {unboxed.getResult(0), unboxed.getResult(1)};
  }

  mlir::Value createEmbox(Char str) {
    auto kind{str.getCharacterType().getFKind()};
    auto boxCharType{fir::BoxCharType::get(builder.getContext(), kind)};
    auto refType{str.getReferenceType()};
    // So far, fir.emboxChar fails lowering to llvm when it is given
    // fir.ref<fir.array<lenxfir.char<kind>>> types, so convert to
    // fir.ref<fir.char<kind>> if needed.
    if (refType != str.ref.getType())
      str.ref = create<fir::ConvertOp>(refType, str.ref);
    // BoxChar length is i64, convert in case the provided length is not.
    auto lenType{M::IntegerType::get(64, builder.getContext())};
    if (str.len.getType() != lenType)
      str.len = create<fir::ConvertOp>(lenType, str.len);
    return create<fir::EmboxCharOp>(boxCharType, str.ref, str.len);
  }

  mlir::Value createLoadCharAt(AddressableChar str, mlir::Value index) {
    auto addr{
        create<fir::CoordinateOp>(str.getReferenceType(), str.ref, index)};
    return create<fir::LoadOp>(addr);
  }
  void createStoreCharAt(AddressableChar str, mlir::Value index,
                         mlir::Value c) {
    auto addr{
        create<fir::CoordinateOp>(str.getReferenceType(), str.ref, index)};
    create<fir::StoreOp>(c, addr);
  }

  void createCopy(Char dest, Char src, M::Value count) {
    auto destArray{convertToAddressableChar(dest)};
    auto srcArray{convertToAddressableChar(src)};

    LoopBuilder{*this}.createLoop(
        count, [&](OpBuilderWrapper &handler, M::Value index) {
          CharacterOpsBuilderImpl charHanlder{handler};
          auto charVal{charHanlder.createLoadCharAt(srcArray, index)};
          charHanlder.createStoreCharAt(destArray, index, charVal);
        });
  }

  void createPadding(Char str, M::Value lower, M::Value upper) {
    auto strArray{convertToAddressableChar(str)};
    auto blank{createBlankConstant(str.getCharacterType())};
    LoopBuilder{*this}.createLoop(
        lower, upper, [&](OpBuilderWrapper &handler, M::Value index) {
          CharacterOpsBuilderImpl{handler}.createStoreCharAt(strArray, index,
                                                             blank);
        });
  }

  Char createTemp(fir::CharacterType type, mlir::Value len) {
    // FIXME Does this need to be emitted somewhere safe ?
    // convert-expr.cc generates alloca at the beginning of the mlir block.
    auto indexType{M::IndexType::get(builder.getContext())};
    auto ten{createIntegerConstant(indexType, 10)};
    llvm::SmallVector<mlir::Value, 0> lengths;
    llvm::SmallVector<mlir::Value, 3> sizes{len};
    auto ref{builder.create<fir::AllocaOp>(loc, type, lengths, sizes)};
    return {ref, ten};
  }

  void createAssign(Char lhs, Char rhs) {
    Char safe_rhs{rhs};
    // TODO disable copy for easy stuff like constant rhs ?.
    // we need to know a boxchar holds constant data then.
    // Add an attribute to such box ?
    if (true) {
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

  mlir::Value createBlankConstant(fir::CharacterType type) {
    auto byteTy{M::IntegerType::get(8, builder.getContext())};
    auto asciiSpace{createIntegerConstant(byteTy, 0x20)};
    return create<fir::ConvertOp>(type, asciiSpace);
  }

  Char createSubstring(Char str, llvm::ArrayRef<mlir::Value> bounds) {

    auto nbounds{bounds.size()};
    if (nbounds < 1 || nbounds > 2) {
      M::emitError(loc, "Incorrect number of bounds in substring");
      return {mlir::Value{}, mlir::Value{}};
    }
    auto indexedChar{convertToAddressableChar(str)};
    auto indexType{M::IndexType::get(builder.getContext())};
    auto lowerBound{create<fir::ConvertOp>(indexType, bounds[0])};
    // FIR CoordinateOp is zero based but Fortran substring are one based.
    auto oneIndex{createIntegerConstant(indexType, 1)};
    auto offsetIndex{create<M::SubIOp>(lowerBound, oneIndex).getResult()};
    auto substringRef{create<fir::CoordinateOp>(str.getReferenceType(),
                                                indexedChar.ref, offsetIndex)};

    // Compute the length.
    mlir::Value substringLen{};
    if (nbounds < 2) {
      substringLen = create<M::SubIOp>(str.len, bounds[0]);
    } else {
      substringLen = create<M::SubIOp>(bounds[1], bounds[0]);
    }
    auto one{createIntegerConstant(substringLen.getType(), 1)};
    substringLen = create<M::AddIOp>(substringLen, one);

    // TODO: Do we need to set the length to zero in case the
    // Fortran user gave wrong bounds and the length is negative ? YES: 9.4.1
    // (2). Expose integer max/min somewhere to avoid reimplementing it
    // everywhere.
    return {substringRef, substringLen};
  }
};
} // namespace

void Br::CharacterOpsBuilder::createCopy(mlir::Value dest, mlir::Value src,
                                         mlir::Value count) {
  CharacterOpsBuilderImpl impl{*this};
  impl.createCopy(impl.createUnboxChar(dest), impl.createUnboxChar(src), count);
}

void Br::CharacterOpsBuilder::createPadding(mlir::Value str, mlir::Value lower,
                                            mlir::Value upper) {
  CharacterOpsBuilderImpl impl{*this};
  impl.createPadding(impl.createUnboxChar(str), lower, upper);
}

mlir::Value
Br::CharacterOpsBuilder::createSubstring(mlir::Value str,
                                         llvm::ArrayRef<mlir::Value> bounds) {
  CharacterOpsBuilderImpl impl{*this};
  return impl.createEmbox(
      impl.createSubstring(impl.createUnboxChar(str), bounds));
}

mlir::Value Br::CharacterOpsBuilder::createTemp(fir::CharacterType type,
                                                mlir::Value len) {
  CharacterOpsBuilderImpl impl{*this};
  return impl.createEmbox(impl.createTemp(type, len));
}

void Br::CharacterOpsBuilder::createAssign(mlir::Value lhs, mlir::Value rhs) {
  CharacterOpsBuilderImpl impl{*this};
  impl.createAssign(impl.createUnboxChar(lhs), impl.createUnboxChar(rhs));
}

mlir::Value Br::CharacterOpsBuilder::createEmbox(mlir::Value addr,
                                                 mlir::Value len) {
  return CharacterOpsBuilderImpl{*this}.createEmbox(
      CharacterOpsBuilderImpl::Char{addr, len});
}

std::pair<mlir::Value, mlir::Value>
Br::CharacterOpsBuilder::createUnbox(mlir::Value boxChar) {
  auto c{CharacterOpsBuilderImpl{*this}.createUnboxChar(boxChar)};
  return {c.ref, c.len};
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
