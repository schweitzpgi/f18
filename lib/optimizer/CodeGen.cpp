//===-- CodeGen.cpp -- bridge to lower to LLVM ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/optimizer/CodeGen/CodeGen.h"
#include "flang/optimizer/Dialect/FIRAttr.h"
#include "flang/optimizer/Dialect/FIRDialect.h"
#include "flang/optimizer/Dialect/FIROps.h"
#include "flang/optimizer/Dialect/FIRType.h"
#include "flang/optimizer/Support/InternalNames.h"
#include "flang/optimizer/Support/KindMapping.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Config/abi-breaking.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

/// The Tilikum bridge performs the conversion of operations from both the FIR
/// and standard dialects to the LLVM-IR dialect.
///
/// Some FIR operations may be lowered to other dialects, such as standard, but
/// some FIR operations will pass through to the Tilikum bridge.  This may be
/// necessary to preserve the semantics of the Fortran program.

#undef TODO
#define TODO(X)                                                                \
  (void)X;                                                                     \
  assert(false && "not yet implemented")

using namespace llvm;
using namespace fir;

using OperandTy = ArrayRef<mlir::Value>;

static cl::opt<bool>
    ClDisableFirToLLVMIR("disable-fir2llvmir",
                         cl::desc("disable FIR to LLVM-IR dialect pass"),
                         cl::init(false), cl::Hidden);

static cl::opt<bool> ClDisableLLVM("disable-llvm",
                                   cl::desc("disable LLVM pass"),
                                   cl::init(false), cl::Hidden);

namespace fir {
/// return true if all `Value`s in `operands` are `ConstantOp`s
bool allConstants(OperandTy operands) {
  for (auto opnd : operands) {
    if (auto defop = opnd.getDefiningOp())
      if (dyn_cast<mlir::LLVM::ConstantOp>(defop) ||
          dyn_cast<mlir::ConstantOp>(defop))
        continue;
    return false;
  }
  return true;
}
} // namespace fir

namespace {

using SmallVecResult = SmallVector<mlir::Value, 4>;
using AttributeTy = ArrayRef<mlir::NamedAttribute>;

const unsigned defaultAlign{8};

/// FIR type converter
/// This converts FIR types to LLVM types (for now)
class FIRToLLVMTypeConverter : public mlir::LLVMTypeConverter {
  KindMapping kindMapping;
  static StringMap<mlir::LLVM::LLVMType> identStructCache;

public:
  FIRToLLVMTypeConverter(mlir::MLIRContext *context, NameUniquer &uniquer)
      : LLVMTypeConverter(context), kindMapping(context), uniquer(uniquer) {
    addConversion([&](BoxType box) { return convertBoxType(box); });
    addConversion(
        [&](BoxCharType boxchar) { return convertBoxCharType(boxchar); });
    addConversion(
        [&](BoxProcType boxproc) { return convertBoxProcType(boxproc); });
    addConversion(
        [&](CharacterType charTy) { return convertCharType(charTy); });
    addConversion(
        [&](CplxType cplx) { return convertComplexType(cplx.getFKind()); });
    addConversion(
        [&](RecordType derived) { return convertRecordType(derived); });
    addConversion([&](DimsType dims) {
      return mlir::LLVM::LLVMType::getArrayTy(dimsType(), dims.getRank());
    });
    addConversion([&](FieldType field) {
      return mlir::LLVM::LLVMType::getInt32Ty(llvmDialect);
    });
    addConversion([&](HeapType heap) { return convertPointerLike(heap); });
    addConversion([&](IntType intr) { return convertIntegerType(intr); });
    addConversion([&](LenType field) {
      return mlir::LLVM::LLVMType::getInt32Ty(llvmDialect);
    });
    addConversion(
        [&](LogicalType logical) { return convertLogicalType(logical); });
    addConversion(
        [&](fir::PointerType pointer) { return convertPointerLike(pointer); });
    addConversion(
        [&](RealType real) { return convertRealType(real.getFKind()); });
    addConversion([&](ReferenceType ref) { return convertPointerLike(ref); });
    addConversion(
        [&](SequenceType sequence) { return convertSequenceType(sequence); });
    addConversion([&](TypeDescType tdesc) {
      return convertTypeDescType(tdesc.getContext());
    });
    addConversion(
        [&](mlir::TupleType tuple) { return convertTupleType(tuple); });
    addConversion(
        [&](mlir::ComplexType cmplx) { return convertComplexType(cmplx); });
    addConversion([&](mlir::NoneType none) {
      return mlir::LLVM::LLVMType::getStructTy(llvmDialect, {});
    });
  }

  // This returns the type of a single column. Rows are added by the caller.
  // fir.dims<r>  -->  llvm<"[r x [3 x i64]]">
  mlir::LLVM::LLVMType dimsType() {
    auto i64Ty{mlir::LLVM::LLVMType::getInt64Ty(llvmDialect)};
    return mlir::LLVM::LLVMType::getArrayTy(i64Ty, 3);
  }

  // i32 is used here because LLVM wants i32 constants when indexing into struct
  // types. Indexing into other aggregate types is more flexible.
  mlir::LLVM::LLVMType offsetType() {
    return mlir::LLVM::LLVMType::getInt32Ty(llvmDialect);
  }

  // i64 can be used to index into aggregates like arrays
  mlir::LLVM::LLVMType indexType() {
    return mlir::LLVM::LLVMType::getInt64Ty(llvmDialect);
  }

  // This corresponds to the descriptor as defined ISO_Fortran_binding.h and the
  // addendum defined in descriptor.h.
  // FIXME: This code should be generated and follow SPOT
  mlir::LLVM::LLVMType convertBoxType(BoxType box) {
    // (buffer*, ele-size, rank, type-descriptor, attribute, [dims])
    SmallVector<mlir::LLVM::LLVMType, 6> parts;
    mlir::Type ele = box.getEleTy();
    // auto *ctx = box.getContext();
    auto eleTy = unwrap(convertType(ele));
    // buffer*
    parts.push_back(eleTy.getPointerTo());
    // ele-size
    parts.push_back(mlir::LLVM::LLVMType::getInt64Ty(llvmDialect));
    // version
    parts.push_back(mlir::LLVM::LLVMType::getInt32Ty(llvmDialect));
    // rank
    parts.push_back(mlir::LLVM::LLVMType::getInt8Ty(llvmDialect));
    // type (code)
    parts.push_back(mlir::LLVM::LLVMType::getInt8Ty(llvmDialect));
    // attribute
    parts.push_back(mlir::LLVM::LLVMType::getInt8Ty(llvmDialect));
    // addendum
    parts.push_back(mlir::LLVM::LLVMType::getInt8Ty(llvmDialect));
    // opt-dims: [0..15 x [int,int,int]]  (see fir.dims)
    // opt-type-ptr: i8* (see fir.tdesc)
    // opt-flags: i64
    // opt-len-params: [? x i64]
    return mlir::LLVM::LLVMType::getStructTy(llvmDialect, parts).getPointerTo();
  }

  // fir.boxchar<n>  -->  llvm<"{ ix*, i64 }">   where ix is kind mapping
  mlir::LLVM::LLVMType convertBoxCharType(BoxCharType boxchar) {
    auto ptrTy = convertCharType(boxchar.getEleTy()).getPointerTo();
    auto i64Ty = mlir::LLVM::LLVMType::getInt64Ty(llvmDialect);
    SmallVector<mlir::LLVM::LLVMType, 2> tuple{ptrTy, i64Ty};
    return mlir::LLVM::LLVMType::getStructTy(llvmDialect, tuple);
  }

  // fir.boxproc<any>  -->  llvm<"{ any*, i8* }">
  mlir::LLVM::LLVMType convertBoxProcType(BoxProcType boxproc) {
    auto funcTy = convertType(boxproc.getEleTy());
    auto ptrTy = unwrap(funcTy).getPointerTo();
    auto i8Ty = mlir::LLVM::LLVMType::getInt8Ty(llvmDialect);
    SmallVector<mlir::LLVM::LLVMType, 2> tuple{ptrTy, i8Ty};
    return mlir::LLVM::LLVMType::getStructTy(llvmDialect, tuple);
  }

  unsigned characterBitsize(CharacterType charTy) {
    return kindMapping.getCharacterBitsize(charTy.getFKind());
  }

  // fir.char<n>  -->  llvm<"ix*">   where ix is scaled by kind mapping
  mlir::LLVM::LLVMType convertCharType(CharacterType charTy) {
    return mlir::LLVM::LLVMType::getIntNTy(llvmDialect,
                                           characterBitsize(charTy));
  }

  mlir::LLVM::LLVMType convertComplexPartType(KindTy kind) {
    auto realID = kindMapping.getComplexTypeID(kind);
    return fromRealTypeID(realID, kind);
  }

  // fir.complex<n>  -->  llvm<"{ anyfloat, anyfloat }">
  mlir::LLVM::LLVMType convertComplexType(KindTy kind) {
    auto realTy = convertComplexPartType(kind);
    SmallVector<mlir::LLVM::LLVMType, 2> tuple{realTy, realTy};
    return mlir::LLVM::LLVMType::getStructTy(llvmDialect, tuple);
  }

  mlir::LLVM::LLVMType getDefaultInt() {
    // FIXME: this should be tied to the front-end default
    return mlir::LLVM::LLVMType::getInt64Ty(llvmDialect);
  }

  // fir.int<n>  -->  llvm.ix   where ix is a kind mapping
  mlir::LLVM::LLVMType convertIntegerType(IntType intTy) {
    return mlir::LLVM::LLVMType::getIntNTy(
        llvmDialect, kindMapping.getIntegerBitsize(intTy.getFKind()));
  }

  // fir.logical<n>  -->  llvm.ix  where ix is a kind mapping
  mlir::LLVM::LLVMType convertLogicalType(LogicalType boolTy) {
    return mlir::LLVM::LLVMType::getIntNTy(
        llvmDialect, kindMapping.getLogicalBitsize(boolTy.getFKind()));
  }

  template <typename A>
  mlir::LLVM::LLVMType convertPointerLike(A &ty) {
    return unwrap(convertType(ty.getEleTy())).getPointerTo();
  }

  // convert a front-end kind value to either a std or LLVM IR dialect type
  // fir.real<n>  -->  llvm.anyfloat  where anyfloat is a kind mapping
  mlir::LLVM::LLVMType convertRealType(KindTy kind) {
    return fromRealTypeID(kindMapping.getRealTypeID(kind), kind);
  }

  // fir.type<name(p : TY'...){f : TY...}>  -->  llvm<"%name = { ty... }">
  mlir::LLVM::LLVMType convertRecordType(RecordType derived) {
    auto name{derived.getName()};
    // The cache is needed to keep a unique mapping from name -> StructType
    auto iter{identStructCache.find(name)};
    if (iter != identStructCache.end())
      return iter->second;
    auto st{mlir::LLVM::LLVMType::createStructTy(llvmDialect, name)};
    identStructCache[name] = st;
    SmallVector<mlir::LLVM::LLVMType, 8> members;
    for (auto mem : derived.getTypeList())
      members.push_back(convertType(mem.second).cast<mlir::LLVM::LLVMType>());
    mlir::LLVM::LLVMType::setStructTyBody(st, members);
    return st;
  }

  // fir.array<c ... :any>  -->  llvm<"[...[c x any]]">
  mlir::LLVM::LLVMType convertSequenceType(SequenceType seq) {
    auto baseTy = unwrap(convertType(seq.getEleTy()));
    auto shape = seq.getShape();
    if (shape.size()) {
      for (auto e : shape) {
        if (e < 0)
          e = 0;
        baseTy = mlir::LLVM::LLVMType::getArrayTy(baseTy, e);
      }
      return baseTy;
    }
    return baseTy.getPointerTo();
  }

  // tuple<TS...>  -->  llvm<"{ ts... }">
  mlir::LLVM::LLVMType convertTupleType(mlir::TupleType tuple) {
    SmallVector<mlir::Type, 8> inMembers;
    tuple.getFlattenedTypes(inMembers);
    SmallVector<mlir::LLVM::LLVMType, 8> members;
    for (auto mem : inMembers)
      members.push_back(convertType(mem).cast<mlir::LLVM::LLVMType>());
    return mlir::LLVM::LLVMType::getStructTy(llvmDialect, members);
  }

  // complex<T>  --> llvm<"{t,t}">
  mlir::LLVM::LLVMType convertComplexType(mlir::ComplexType complex) {
    auto eleTy = unwrap(convertType(complex.getElementType()));
    SmallVector<mlir::LLVM::LLVMType, 2> tuple{eleTy, eleTy};
    return mlir::LLVM::LLVMType::getStructTy(llvmDialect, tuple);
  }

  // fir.tdesc<any>  -->  llvm<"i8*">
  // FIXME: for now use a void*, however pointer identity is not sufficient for
  // the f18 object v. class distinction
  mlir::LLVM::LLVMType convertTypeDescType(mlir::MLIRContext *ctx) {
    return mlir::LLVM::LLVMType::getInt8PtrTy(llvmDialect);
  }

  /// Convert llvm::Type::TypeID to mlir::LLVM::LLVMType
  mlir::LLVM::LLVMType fromRealTypeID(llvm::Type::TypeID typeID, KindTy kind) {
    switch (typeID) {
    case llvm::Type::TypeID::HalfTyID:
      return mlir::LLVM::LLVMType::getHalfTy(llvmDialect);
    case llvm::Type::TypeID::FloatTyID:
      return mlir::LLVM::LLVMType::getFloatTy(llvmDialect);
    case llvm::Type::TypeID::DoubleTyID:
      return mlir::LLVM::LLVMType::getDoubleTy(llvmDialect);
    case llvm::Type::TypeID::X86_FP80TyID:
      return mlir::LLVM::LLVMType::getX86_FP80Ty(llvmDialect);
    case llvm::Type::TypeID::FP128TyID:
      return mlir::LLVM::LLVMType::getFP128Ty(llvmDialect);
    default:
      emitError(UnknownLoc::get(llvmDialect->getContext()))
          << "unsupported type: !fir.real<" << kind << ">";
      return {};
    }
  }

  /// HACK: cloned from LLVMTypeConverter since this is private there
  mlir::LLVM::LLVMType unwrap(mlir::Type type) {
    if (!type)
      return nullptr;
    auto *mlirContext = type.getContext();
    auto wrappedLLVMType = type.dyn_cast<mlir::LLVM::LLVMType>();
    if (!wrappedLLVMType)
      emitError(UnknownLoc::get(mlirContext),
                "conversion resulted in a non-LLVM type");
    return wrappedLLVMType;
  }

  NameUniquer &uniquer;
};

// instantiate static data member
StringMap<mlir::LLVM::LLVMType> FIRToLLVMTypeConverter::identStructCache;

/// remove `omitNames` (by name) from the attribute dictionary
SmallVector<mlir::NamedAttribute, 4>
pruneNamedAttrDict(ArrayRef<mlir::NamedAttribute> attrs,
                   ArrayRef<StringRef> omitNames) {
  SmallVector<mlir::NamedAttribute, 4> result;
  for (auto x : attrs) {
    bool omit = false;
    for (auto o : omitNames)
      if (x.first.strref() == o) {
        omit = true;
        break;
      }
    if (!omit)
      result.push_back(x);
  }
  return result;
}

inline mlir::LLVM::LLVMType getVoidPtrType(mlir::LLVM::LLVMDialect *dialect) {
  return mlir::LLVM::LLVMType::getInt8PtrTy(dialect);
}

/// FIR conversion pattern template
template <typename FromOp>
class FIROpConversion : public mlir::ConversionPattern {
public:
  explicit FIROpConversion(mlir::MLIRContext *ctx,
                           FIRToLLVMTypeConverter &lowering)
      : ConversionPattern(FromOp::getOperationName(), 1, ctx),
        lowering(lowering) {}

protected:
  LLVMContext &getLLVMContext() const { return lowering.getLLVMContext(); }
  mlir::LLVM::LLVMDialect *getDialect() const { return lowering.getDialect(); }
  mlir::Type convertType(mlir::Type ty) const {
    return lowering.convertType(ty);
  }
  mlir::LLVM::LLVMType unwrap(mlir::Type ty) const {
    return lowering.unwrap(ty);
  }
  mlir::LLVM::LLVMType voidPtrTy() const {
    return getVoidPtrType(getDialect());
  }

  mlir::LLVM::ConstantOp
  genConstantOffset(mlir::Location loc,
                    mlir::ConversionPatternRewriter &rewriter,
                    int offset) const {
    auto ity = lowering.offsetType();
    auto cattr = rewriter.getI32IntegerAttr(offset);
    return rewriter.create<mlir::LLVM::ConstantOp>(loc, ity, cattr);
  }

  FIRToLLVMTypeConverter &lowering;
};

/// Create an LLVM dialect global
void createGlobal(mlir::Location loc, mlir::ModuleOp mod, StringRef name,
                  mlir::LLVM::LLVMType type,
                  mlir::ConversionPatternRewriter &rewriter) {
  if (mod.lookupSymbol<mlir::LLVM::GlobalOp>(name))
    return;
  mlir::OpBuilder modBuilder(mod.getBodyRegion());
  modBuilder.create<mlir::LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                          mlir::LLVM::Linkage::Weak, name,
                                          mlir::Attribute{});
}

struct AddrOfOpConversion : public FIROpConversion<fir::AddrOfOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto addr = mlir::cast<fir::AddrOfOp>(op);
    auto ty = unwrap(convertType(addr.getType()));
    auto attrs = pruneNamedAttrDict(addr.getAttrs(), {"symbol"});
    rewriter.replaceOpWithNewOp<mlir::LLVM::AddressOfOp>(
        addr, ty, addr.symbol().getRootReference(), attrs);
    return matchSuccess();
  }
};

mlir::LLVM::ConstantOp
genConstantIndex(mlir::Location loc, mlir::LLVM::LLVMType ity,
                 mlir::ConversionPatternRewriter &rewriter, int offset) {
  auto cattr = rewriter.getI64IntegerAttr(offset);
  return rewriter.create<mlir::LLVM::ConstantOp>(loc, ity, cattr);
}

/// convert to LLVM IR dialect `alloca`
struct AllocaOpConversion : public FIROpConversion<fir::AllocaOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto alloc = mlir::cast<fir::AllocaOp>(op);
    auto loc = alloc.getLoc();
    auto ity = lowering.indexType();
    auto c1 = genConstantIndex(loc, ity, rewriter, 1);
    auto size = c1.getResult();
    for (auto opnd : operands)
      size = rewriter.create<mlir::LLVM::MulOp>(loc, ity, size, opnd);
    auto ty = convertType(alloc.getType());
    rewriter.replaceOpWithNewOp<mlir::LLVM::AllocaOp>(alloc, ty, size,
                                                      alloc.getAttrs());
    return matchSuccess();
  }
};

mlir::LLVM::LLVMFuncOp getMalloc(AllocMemOp op,
                                 mlir::ConversionPatternRewriter &rewriter,
                                 mlir::LLVM::LLVMDialect *dialect) {
  auto module = op.getParentOfType<mlir::ModuleOp>();
  if (auto mallocFunc = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("malloc"))
    return mallocFunc;
  mlir::OpBuilder moduleBuilder(
      op.getParentOfType<mlir::ModuleOp>().getBodyRegion());
  auto indexType = mlir::LLVM::LLVMType::getInt64Ty(dialect);
  return moduleBuilder.create<mlir::LLVM::LLVMFuncOp>(
      rewriter.getUnknownLoc(), "malloc",
      mlir::LLVM::LLVMType::getFunctionTy(getVoidPtrType(dialect), indexType,
                                          /*isVarArg=*/false));
}

/// convert to `call` to the runtime to `malloc` memory
struct AllocMemOpConversion : public FIROpConversion<AllocMemOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto heap = mlir::cast<AllocMemOp>(op);
    auto ty = convertType(heap.getType());
    auto dialect = getDialect();
    auto mallocFunc = getMalloc(heap, rewriter, dialect);
    auto loc = heap.getLoc();
    auto ity = lowering.indexType();
    auto c1 = genConstantIndex(loc, ity, rewriter, 1);
    auto size = c1.getResult();
    for (auto opnd : operands)
      size = rewriter.create<mlir::LLVM::MulOp>(loc, ity, size, opnd);
    heap.setAttr("callee", rewriter.getSymbolRefAttr(mallocFunc));
    SmallVector<mlir::Value, 1> args{size};
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(heap, ty, args,
                                                    heap.getAttrs());
    return matchSuccess();
  }
};

/// obtain the free() function
mlir::LLVM::LLVMFuncOp getFree(FreeMemOp op,
                               mlir::ConversionPatternRewriter &rewriter,
                               mlir::LLVM::LLVMDialect *dialect) {
  auto module = op.getParentOfType<mlir::ModuleOp>();
  if (auto freeFunc = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("free"))
    return freeFunc;
  mlir::OpBuilder moduleBuilder(module.getBodyRegion());
  auto voidType = mlir::LLVM::LLVMType::getVoidTy(dialect);
  return moduleBuilder.create<mlir::LLVM::LLVMFuncOp>(
      rewriter.getUnknownLoc(), "free",
      mlir::LLVM::LLVMType::getFunctionTy(voidType, getVoidPtrType(dialect),
                                          /*isVarArg=*/false));
}

/// lower a freemem instruction into a call to free()
struct FreeMemOpConversion : public FIROpConversion<fir::FreeMemOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto freemem = mlir::cast<fir::FreeMemOp>(op);
    auto dialect = getDialect();
    auto freeFunc = getFree(freemem, rewriter, dialect);
    auto bitcast = rewriter.create<mlir::LLVM::BitcastOp>(
        freemem.getLoc(), voidPtrTy(), operands[0]);
    freemem.setAttr("callee", rewriter.getSymbolRefAttr(freeFunc));
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
        freemem, mlir::LLVM::LLVMType::getVoidTy(dialect),
        SmallVector<mlir::Value, 1>{bitcast}, freemem.getAttrs());
    return matchSuccess();
  }
};

template <typename... ARGS>
mlir::LLVM::GEPOp genGEP(mlir::Location loc, mlir::LLVM::LLVMType ty,
                         mlir::ConversionPatternRewriter &rewriter,
                         mlir::Value base, ARGS... args) {
  SmallVector<mlir::Value, 8> cv{args...};
  return rewriter.create<mlir::LLVM::GEPOp>(loc, ty, base, cv);
}

/// convert to returning the first element of the box (any flavor)
struct BoxAddrOpConversion : public FIROpConversion<BoxAddrOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto boxaddr = mlir::cast<BoxAddrOp>(op);
    auto a = operands[0];
    auto loc = boxaddr.getLoc();
    auto ty = convertType(boxaddr.getType());
    if (auto argty = boxaddr.val().getType().dyn_cast<BoxType>()) {
      auto c0 = genConstantOffset(loc, rewriter, 0);
      auto pty = unwrap(ty).getPointerTo();
      auto p = genGEP(loc, unwrap(pty), rewriter, a, c0, c0);
      // load the pointer from the buffer
      rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(boxaddr, ty, p);
    } else {
      auto c0attr = rewriter.getI32IntegerAttr(0);
      auto c0 = mlir::ArrayAttr::get(c0attr, boxaddr.getContext());
      rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(boxaddr, ty, a,
                                                              c0);
    }
    return matchSuccess();
  }
};

/// convert to an extractvalue for the 2nd part of the boxchar
struct BoxCharLenOpConversion : public FIROpConversion<BoxCharLenOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto boxchar = mlir::cast<BoxCharLenOp>(op);
    auto a = operands[0];
    auto ty = convertType(boxchar.getType());
    auto ctx = boxchar.getContext();
    auto c1 = mlir::ArrayAttr::get(rewriter.getI32IntegerAttr(1), ctx);
    rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(boxchar, ty, a, c1);
    return matchSuccess();
  }
};

/// convert to a triple set of GEPs and loads
struct BoxDimsOpConversion : public FIROpConversion<BoxDimsOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto boxdims = mlir::cast<BoxDimsOp>(op);
    auto a = operands[0];
    auto dim = operands[1];
    auto loc = boxdims.getLoc();
    auto c0 = genConstantOffset(loc, rewriter, 0);
    auto c7 = genConstantOffset(loc, rewriter, 7);
    auto l0 = loadFromOffset(boxdims, loc, a, c0, c7, dim, 0, rewriter);
    auto l1 = loadFromOffset(boxdims, loc, a, c0, c7, dim, 1, rewriter);
    auto l2 = loadFromOffset(boxdims, loc, a, c0, c7, dim, 2, rewriter);
    rewriter.replaceOp(boxdims,
                       {l0.getResult(), l1.getResult(), l2.getResult()});
    return matchSuccess();
  }

  mlir::LLVM::LoadOp
  loadFromOffset(BoxDimsOp boxdims, mlir::Location loc, mlir::Value a,
                 mlir::LLVM::ConstantOp c0, mlir::LLVM::ConstantOp c7,
                 mlir::Value dim, int off,
                 mlir::ConversionPatternRewriter &rewriter) const {
    auto ty = convertType(boxdims.getResult(off).getType());
    auto pty = unwrap(ty).getPointerTo();
    auto c = genConstantOffset(loc, rewriter, off);
    auto p = genGEP(loc, pty, rewriter, a, c0, c7, dim, c);
    return rewriter.create<mlir::LLVM::LoadOp>(loc, ty, p);
  }
};

struct BoxEleSizeOpConversion : public FIROpConversion<BoxEleSizeOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto boxelesz = mlir::cast<BoxEleSizeOp>(op);
    auto a = operands[0];
    auto loc = boxelesz.getLoc();
    auto c0 = genConstantOffset(loc, rewriter, 0);
    auto c1 = genConstantOffset(loc, rewriter, 1);
    auto ty = convertType(boxelesz.getType());
    auto p = genGEP(loc, unwrap(ty), rewriter, a, c0, c1);
    rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(boxelesz, ty, p);
    return matchSuccess();
  }
};

struct BoxIsAllocOpConversion : public FIROpConversion<BoxIsAllocOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto boxisalloc = mlir::cast<BoxIsAllocOp>(op);
    auto a = operands[0];
    auto loc = boxisalloc.getLoc();
    auto ity = lowering.offsetType();
    auto c0 = genConstantOffset(loc, rewriter, 0);
    auto c5 = genConstantOffset(loc, rewriter, 5);
    auto ty = convertType(boxisalloc.getType());
    auto p = genGEP(loc, unwrap(ty), rewriter, a, c0, c5);
    auto ld = rewriter.create<mlir::LLVM::LoadOp>(loc, ty, p);
    auto ab = genConstantOffset(loc, rewriter, 2);
    auto bit = rewriter.create<mlir::LLVM::AndOp>(loc, ity, ld, ab);
    rewriter.replaceOpWithNewOp<mlir::LLVM::ICmpOp>(
        boxisalloc, mlir::LLVM::ICmpPredicate::ne, bit, c0);
    return matchSuccess();
  }
};

struct BoxIsArrayOpConversion : public FIROpConversion<BoxIsArrayOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto boxisarray = mlir::cast<BoxIsArrayOp>(op);
    auto a = operands[0];
    auto loc = boxisarray.getLoc();
    auto c0 = genConstantOffset(loc, rewriter, 0);
    auto c3 = genConstantOffset(loc, rewriter, 3);
    auto ty = convertType(boxisarray.getType());
    auto p = genGEP(loc, unwrap(ty), rewriter, a, c0, c3);
    auto ld = rewriter.create<mlir::LLVM::LoadOp>(loc, ty, p);
    rewriter.replaceOpWithNewOp<mlir::LLVM::ICmpOp>(
        boxisarray, mlir::LLVM::ICmpPredicate::ne, ld, c0);
    return matchSuccess();
  }
};

struct BoxIsPtrOpConversion : public FIROpConversion<BoxIsPtrOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto boxisptr = mlir::cast<BoxIsPtrOp>(op);
    auto a = operands[0];
    auto loc = boxisptr.getLoc();
    auto ty = convertType(boxisptr.getType());
    auto ity = lowering.offsetType();
    auto c0 = genConstantOffset(loc, rewriter, 0);
    auto c5 = genConstantOffset(loc, rewriter, 5);
    SmallVector<mlir::Value, 4> args{a, c0, c5};
    auto p = rewriter.create<mlir::LLVM::GEPOp>(loc, ty, args);
    auto ld = rewriter.create<mlir::LLVM::LoadOp>(loc, ty, p);
    auto ab = genConstantOffset(loc, rewriter, 1);
    auto bit = rewriter.create<mlir::LLVM::AndOp>(loc, ity, ld, ab);
    rewriter.replaceOpWithNewOp<mlir::LLVM::ICmpOp>(
        boxisptr, mlir::LLVM::ICmpPredicate::ne, bit, c0);
    return matchSuccess();
  }
};

struct BoxProcHostOpConversion : public FIROpConversion<BoxProcHostOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto boxprochost = mlir::cast<BoxProcHostOp>(op);
    auto a = operands[0];
    auto ty = convertType(boxprochost.getType());
    auto ctx = boxprochost.getContext();
    auto c1 = mlir::ArrayAttr::get(rewriter.getI32IntegerAttr(1), ctx);
    rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(boxprochost, ty, a,
                                                            c1);
    return matchSuccess();
  }
};

struct BoxRankOpConversion : public FIROpConversion<BoxRankOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto boxrank = mlir::cast<BoxRankOp>(op);
    auto a = operands[0];
    auto loc = boxrank.getLoc();
    auto ty = convertType(boxrank.getType());
    auto c0 = genConstantOffset(loc, rewriter, 0);
    auto c3 = genConstantOffset(loc, rewriter, 3);
    SmallVector<mlir::Value, 4> args{a, c0, c3};
    auto pty = unwrap(ty).getPointerTo();
    auto p = rewriter.create<mlir::LLVM::GEPOp>(loc, pty, args);
    rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(boxrank, ty, p);
    return matchSuccess();
  }
};

struct BoxTypeDescOpConversion : public FIROpConversion<BoxTypeDescOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto boxtypedesc = mlir::cast<BoxTypeDescOp>(op);
    auto a = operands[0];
    auto loc = boxtypedesc.getLoc();
    auto ty = convertType(boxtypedesc.getType());
    auto c0 = genConstantOffset(loc, rewriter, 0);
    auto c4 = genConstantOffset(loc, rewriter, 4);
    SmallVector<mlir::Value, 4> args{a, c0, c4};
    auto pty = unwrap(ty).getPointerTo();
    auto p = rewriter.create<mlir::LLVM::GEPOp>(loc, pty, args);
    auto ld = rewriter.create<mlir::LLVM::LoadOp>(loc, ty, p);
    auto i8ptr = mlir::LLVM::LLVMType::getInt8PtrTy(getDialect());
    rewriter.replaceOpWithNewOp<mlir::LLVM::IntToPtrOp>(boxtypedesc, i8ptr, ld);
    return matchSuccess();
  }
};

struct StringLitOpConversion : public FIROpConversion<fir::StringLitOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto constop = mlir::cast<fir::StringLitOp>(op);
    auto ty = convertType(constop.getType());
    auto attr = constop.getValue();
    if (attr.isa<mlir::StringAttr>()) {
      rewriter.replaceOpWithNewOp<mlir::LLVM::ConstantOp>(constop, ty, attr);
    } else {
      // convert the array attr to a dense elements attr
      // LLVMIR dialect knows how to lower the latter to LLVM IR
      auto arr = attr.cast<mlir::ArrayAttr>();
      auto size = constop.getSize().cast<mlir::IntegerAttr>().getInt();
      auto eleTy = constop.getType().cast<fir::SequenceType>().getEleTy();
      auto bits = lowering.characterBitsize(eleTy.cast<fir::CharacterType>());
      auto charTy = rewriter.getIntegerType(bits);
      auto det = mlir::VectorType::get({size}, charTy);
      // convert each character to a precise bitsize
      llvm::SmallVector<mlir::Attribute, 64> vec;
      for (auto a : arr.getValue())
        vec.push_back(mlir::IntegerAttr::get(
            charTy, a.cast<mlir::IntegerAttr>().getValue().sextOrTrunc(bits)));
      auto dea = mlir::DenseElementsAttr::get(det, vec);
      rewriter.replaceOpWithNewOp<mlir::LLVM::ConstantOp>(constop, ty, dea);
    }
    return matchSuccess();
  }
};

/// direct call LLVM function
struct CallOpConversion : public FIROpConversion<fir::CallOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto call = mlir::cast<fir::CallOp>(op);
    SmallVector<mlir::Type, 4> resultTys;
    for (auto r : call.getResults())
      resultTys.push_back(convertType(r.getType()));
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(call, resultTys, operands,
                                                    call.getAttrs());
    return matchSuccess();
  }
};

/// Compare complex values
///
/// Per 10.1, the only comparisons available are .EQ. (oeq) and .NE. (une).
///
/// For completeness, all other comparison are done on the real component only.
struct CmpcOpConversion : public FIROpConversion<fir::CmpcOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto cmp = mlir::cast<fir::CmpcOp>(op);
    auto ctxt = cmp.getContext();
    auto kind = cmp.lhs().getType().cast<fir::CplxType>().getFKind();
    auto ty = convertType(fir::RealType::get(ctxt, kind));
    auto loc = cmp.getLoc();
    auto pos0 = mlir::ArrayAttr::get(rewriter.getI32IntegerAttr(0), ctxt);
    SmallVector<mlir::Value, 2> rp{
        rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ty, operands[0], pos0),
        rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ty, operands[1],
                                                    pos0)};
    auto rcp = rewriter.create<mlir::LLVM::FCmpOp>(loc, ty, rp, cmp.getAttrs());
    auto pos1 = mlir::ArrayAttr::get(rewriter.getI32IntegerAttr(1), ctxt);
    SmallVector<mlir::Value, 2> ip{
        rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ty, operands[0], pos1),
        rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ty, operands[1],
                                                    pos1)};
    auto icp = rewriter.create<mlir::LLVM::FCmpOp>(loc, ty, ip, cmp.getAttrs());
    SmallVector<mlir::Value, 2> cp{rcp, icp};
    switch (cmp.getPredicate()) {
    case fir::CmpFPredicate::OEQ: // .EQ.
      rewriter.replaceOpWithNewOp<mlir::LLVM::AndOp>(cmp, ty, cp);
      break;
    case fir::CmpFPredicate::UNE: // .NE.
      rewriter.replaceOpWithNewOp<mlir::LLVM::OrOp>(cmp, ty, cp);
      break;
    default:
      rewriter.replaceOp(cmp, rcp.getResult());
      break;
    }
    return matchSuccess();
  }
};

struct CmpfOpConversion : public FIROpConversion<fir::CmpfOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto cmp = mlir::cast<fir::CmpfOp>(op);
    auto type = convertType(cmp.getType());
    rewriter.replaceOpWithNewOp<mlir::LLVM::FCmpOp>(cmp, type, operands,
                                                    cmp.getAttrs());
    return matchSuccess();
  }
};

struct ConstcOpConversion : public FIROpConversion<ConstcOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto conc = mlir::cast<ConstcOp>(op);
    auto loc = conc.getLoc();
    auto ctx = conc.getContext();
    auto ty = convertType(conc.getType());
    auto ct = conc.getType().cast<fir::CplxType>();
    auto ety = lowering.convertComplexPartType(ct.getFKind());
    auto ri = mlir::FloatAttr::get(ety, getValue(conc.getReal()));
    auto rp = rewriter.create<mlir::LLVM::ConstantOp>(loc, ety, ri);
    auto ii = mlir::FloatAttr::get(ety, getValue(conc.getImaginary()));
    auto ip = rewriter.create<mlir::LLVM::ConstantOp>(loc, ety, ii);
    auto c0 = mlir::ArrayAttr::get(rewriter.getI32IntegerAttr(0), ctx);
    auto c1 = mlir::ArrayAttr::get(rewriter.getI32IntegerAttr(1), ctx);
    auto r = rewriter.create<mlir::LLVM::UndefOp>(loc, ty);
    auto rr = rewriter.create<mlir::LLVM::InsertValueOp>(loc, ty, r, rp, c0);
    rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(conc, ty, rr, ip,
                                                           c1);
    return matchSuccess();
  }

  inline llvm::APFloat getValue(mlir::Attribute attr) const {
    return attr.cast<fir::RealAttr>().getValue();
  }
};

struct ConstfOpConversion : public FIROpConversion<ConstfOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto conf = mlir::cast<ConstfOp>(op);
    auto ty = convertType(conf.getType());
    auto val = conf.getValue();
    rewriter.replaceOpWithNewOp<mlir::LLVM::ConstantOp>(conf, ty, val);
    return matchSuccess();
  }
};

/// convert value of from-type to value of to-type
struct ConvertOpConversion : public FIROpConversion<ConvertOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto convert = mlir::cast<ConvertOp>(op);
    auto fromTy_ = convertType(convert.value().getType());
    auto fromTy = unwrap(fromTy_);
    auto toTy_ = convertType(convert.res().getType());
    auto toTy = unwrap(toTy_);
    auto *fromLLVMTy = fromTy.getUnderlyingType();
    auto *toLLVMTy = toTy.getUnderlyingType();
    auto &op0 = operands[0];
    if (fromLLVMTy == toLLVMTy) {
      rewriter.replaceOp(convert, op0);
      return matchSuccess();
    }
    auto loc = convert.getLoc();
    mlir::Value v;
    if (fromLLVMTy->isFloatingPointTy()) {
      if (toLLVMTy->isFloatingPointTy()) {
        std::size_t fromBits{fromLLVMTy->getPrimitiveSizeInBits()};
        std::size_t toBits{toLLVMTy->getPrimitiveSizeInBits()};
        // FIXME: what if different reps (F16, BF16) are the same size?
        assert(fromBits != toBits);
        if (fromBits > toBits)
          v = rewriter.create<mlir::LLVM::FPTruncOp>(loc, toTy, op0);
        else
          v = rewriter.create<mlir::LLVM::FPExtOp>(loc, toTy, op0);
      } else if (toLLVMTy->isIntegerTy()) {
        v = rewriter.create<mlir::LLVM::FPToSIOp>(loc, toTy, op0);
      }
    } else if (fromLLVMTy->isIntegerTy()) {
      if (toLLVMTy->isIntegerTy()) {
        std::size_t fromBits{fromLLVMTy->getIntegerBitWidth()};
        std::size_t toBits{toLLVMTy->getIntegerBitWidth()};
        assert(fromBits != toBits);
        if (fromBits > toBits)
          v = rewriter.create<mlir::LLVM::TruncOp>(loc, toTy, op0);
        else
          v = rewriter.create<mlir::LLVM::SExtOp>(loc, toTy, op0);
      } else if (toLLVMTy->isFloatingPointTy()) {
        v = rewriter.create<mlir::LLVM::SIToFPOp>(loc, toTy, op0);
      } else if (toLLVMTy->isPointerTy()) {
        v = rewriter.create<mlir::LLVM::IntToPtrOp>(loc, toTy, op0);
      }
    } else if (fromLLVMTy->isPointerTy()) {
      if (toLLVMTy->isIntegerTy()) {
        v = rewriter.create<mlir::LLVM::PtrToIntOp>(loc, toTy, op0);
      } else if (toLLVMTy->isPointerTy()) {
        v = rewriter.create<mlir::LLVM::BitcastOp>(loc, toTy, op0);
      }
    }
    if (v)
      rewriter.replaceOp(op, v);
    else
      emitError(loc) << "cannot convert " << fromTy_ << " to " << toTy_;
    return matchSuccess();
  }
};

/// virtual call to a method in a dispatch table
struct DispatchOpConversion : public FIROpConversion<DispatchOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto dispatch = mlir::cast<DispatchOp>(op);
    auto ty = convertType(dispatch.getFunctionType());
    // get the table, lookup the method, fetch the func-ptr
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(dispatch, ty, operands,
                                                    None);
    TODO(dispatch);
    return matchSuccess();
  }
};

/// dispatch table for a Fortran derived type
struct DispatchTableOpConversion : public FIROpConversion<DispatchTableOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto disptable = mlir::cast<DispatchTableOp>(op);
    TODO(disptable);
    return matchSuccess();
  }
};

/// entry in a dispatch table; binds a method-name to a function
struct DTEntryOpConversion : public FIROpConversion<DTEntryOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto dtentry = mlir::cast<DTEntryOp>(op);
    TODO(dtentry);
    return matchSuccess();
  }
};

/// create a CHARACTER box
struct EmboxCharOpConversion : public FIROpConversion<EmboxCharOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto emboxchar = mlir::cast<EmboxCharOp>(op);
    auto a = operands[0];
    auto b = operands[1];
    auto loc = emboxchar.getLoc();
    auto ctx = emboxchar.getContext();
    auto ty = convertType(emboxchar.getType());
    auto c0 = mlir::ArrayAttr::get(rewriter.getI32IntegerAttr(0), ctx);
    auto c1 = mlir::ArrayAttr::get(rewriter.getI32IntegerAttr(1), ctx);
    auto un = rewriter.create<mlir::LLVM::UndefOp>(loc, ty);
    auto r = rewriter.create<mlir::LLVM::InsertValueOp>(loc, ty, un, a, c0);
    rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(emboxchar, ty, r, b,
                                                           c1);
    return matchSuccess();
  }
};

/// create a generic box on a memory reference
struct EmboxOpConversion : public FIROpConversion<EmboxOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto embox = mlir::cast<EmboxOp>(op);
    auto loc = embox.getLoc();
    auto dialect = getDialect();
    auto ty = unwrap(convertType(embox.getType()));
    auto alloca = genAllocaWithType(loc, ty, 24, defaultAlign, rewriter);
    auto c0 = genConstantOffset(loc, rewriter, 0);
    auto rty = unwrap(operands[0].getType()).getPointerTo();
    auto f0p = genGEP(loc, rty, rewriter, alloca, c0, c0);
    auto f0p_ = rewriter.create<mlir::LLVM::BitcastOp>(loc, rty, f0p);
    rewriter.create<mlir::LLVM::StoreOp>(loc, operands[0], f0p_);
    auto i64Ty = mlir::LLVM::LLVMType::getInt64Ty(dialect);
    auto i64PtrTy = i64Ty.getPointerTo();
    auto f1p = genGEPToField(loc, i64PtrTy, rewriter, alloca, c0, 1);
    auto c0_ = rewriter.create<mlir::LLVM::SExtOp>(loc, i64Ty, c0);
    rewriter.create<mlir::LLVM::StoreOp>(loc, c0_, f1p);
    auto i32PtrTy = mlir::LLVM::LLVMType::getInt32Ty(dialect).getPointerTo();
    auto f2p = genGEPToField(loc, i32PtrTy, rewriter, alloca, c0, 2);
    rewriter.create<mlir::LLVM::StoreOp>(loc, c0, f2p);
    auto i8Ty = mlir::LLVM::LLVMType::getInt8Ty(dialect);
    auto i8PtrTy = mlir::LLVM::LLVMType::getInt8PtrTy(dialect);
    auto c0__ = rewriter.create<mlir::LLVM::TruncOp>(loc, i8Ty, c0);
    auto f3p = genGEPToField(loc, i8PtrTy, rewriter, alloca, c0, 3);
    rewriter.create<mlir::LLVM::StoreOp>(loc, c0__, f3p);
    auto f4p = genGEPToField(loc, i8PtrTy, rewriter, alloca, c0, 4);
    rewriter.create<mlir::LLVM::StoreOp>(loc, c0__, f4p);
    auto f5p = genGEPToField(loc, i8PtrTy, rewriter, alloca, c0, 5);
    rewriter.create<mlir::LLVM::StoreOp>(loc, c0__, f5p);
    auto f6p = genGEPToField(loc, i8PtrTy, rewriter, alloca, c0, 6);
    rewriter.create<mlir::LLVM::StoreOp>(loc, c0__, f6p);
    // FIXME: copy the dims info, etc.

    rewriter.replaceOp(embox, alloca.getResult());
    return matchSuccess();
  }

  /// Generate an alloca of size `size` and cast it to type `toTy`
  mlir::LLVM::BitcastOp
  genAllocaWithType(mlir::Location loc, mlir::LLVM::LLVMType toTy,
                    unsigned size, unsigned alignment,
                    mlir::ConversionPatternRewriter &rewriter) const {
    auto i8Ty = mlir::LLVM::LLVMType::getInt8PtrTy(getDialect());
    auto thisPt = rewriter.saveInsertionPoint();
    auto *thisBlock = rewriter.getInsertionBlock();
    auto func = mlir::cast<mlir::LLVM::LLVMFuncOp>(thisBlock->getParentOp());
    rewriter.setInsertionPointToStart(&func.front());
    auto size_ = genConstantOffset(loc, rewriter, size);
    auto al =
        rewriter.create<mlir::LLVM::AllocaOp>(loc, i8Ty, size_, alignment);
    rewriter.restoreInsertionPoint(thisPt);
    return rewriter.create<mlir::LLVM::BitcastOp>(loc, toTy, al);
  }

  mlir::LLVM::BitcastOp genGEPToField(mlir::Location loc,
                                      mlir::LLVM::LLVMType ty,
                                      mlir::ConversionPatternRewriter &rewriter,
                                      mlir::Value base, mlir::Value zero,
                                      int field) const {
    auto coff = genConstantOffset(loc, rewriter, field);
    auto gep = genGEP(loc, ty, rewriter, base, zero, coff);
    return rewriter.create<mlir::LLVM::BitcastOp>(loc, ty, gep);
  }
};

/// create a procedure pointer box
struct EmboxProcOpConversion : public FIROpConversion<EmboxProcOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto emboxproc = mlir::cast<EmboxProcOp>(op);
    auto a = operands[0];
    auto b = operands[1];
    auto loc = emboxproc.getLoc();
    auto ctx = emboxproc.getContext();
    auto ty = convertType(emboxproc.getType());
    auto c0 = mlir::ArrayAttr::get(rewriter.getI32IntegerAttr(0), ctx);
    auto c1 = mlir::ArrayAttr::get(rewriter.getI32IntegerAttr(1), ctx);
    auto un = rewriter.create<mlir::LLVM::UndefOp>(loc, ty);
    auto r = rewriter.create<mlir::LLVM::InsertValueOp>(loc, ty, un, a, c0);
    rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(emboxproc, ty, r, b,
                                                           c1);
    return matchSuccess();
  }
};

mlir::Attribute getValue(mlir::Value value) {
  assert(value.getDefiningOp());
  if (auto v = dyn_cast<mlir::LLVM::ConstantOp>(value.getDefiningOp()))
    return v.value();
  if (auto v = dyn_cast<mlir::ConstantOp>(value.getDefiningOp()))
    return v.value();
  assert(false && "must be a constant op");
  return {};
}

template <typename A>
inline void appendTo(SmallVectorImpl<A> &dest, ArrayRef<A> from) {
  dest.append(from.begin(), from.end());
}

/// Extract a subobject value from an ssa-value of aggregate type
struct ExtractValueOpConversion : public FIROpConversion<fir::ExtractValueOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto extractVal = mlir::cast<ExtractValueOp>(op);
    auto ty = convertType(extractVal.getType());
    assert(allConstants(operands.drop_front(1)));
    // since all indices are constants use LLVM's extractvalue instruction
    SmallVector<mlir::Attribute, 8> attrs;
    for (std::size_t i = 1, end{operands.size()}; i < end; ++i)
      attrs.push_back(getValue(operands[i]));
    auto position = mlir::ArrayAttr::get(attrs, extractVal.getContext());
    rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(
        extractVal, ty, operands[0], position);
    return matchSuccess();
  }
};

/// InsertValue is the generalized instruction for the composition of new
/// aggregate type values.
struct InsertValueOpConversion : public FIROpConversion<InsertValueOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto insertVal = cast<InsertValueOp>(op);
    auto ty = convertType(insertVal.getType());
    assert(allConstants(operands.drop_front(2)));
    // since all indices must be constants use LLVM's insertvalue instruction
    SmallVector<mlir::Attribute, 8> attrs;
    for (std::size_t i = 2, end{operands.size()}; i < end; ++i)
      attrs.push_back(getValue(operands[i]));
    auto position = mlir::ArrayAttr::get(attrs, insertVal.getContext());
    rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(
        insertVal, ty, operands[0], operands[1], position);
    return matchSuccess();
  }
};

/// return true if all `Value`s in `operands` are not `FieldIndexOp`s
bool noFieldIndexOps(mlir::Operation::operand_range operands) {
  for (auto opnd : operands) {
    if (auto defop = opnd.getDefiningOp())
      if (dyn_cast<FieldIndexOp>(defop))
        return false;
  }
  return true;
}

/// convert to reference to a reference to a subobject
struct CoordinateOpConversion : public FIROpConversion<CoordinateOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto coor = mlir::cast<CoordinateOp>(op);
    auto ty = convertType(coor.getType());
    auto loc = coor.getLoc();
    mlir::Value base = operands[0];
    auto c0 = genConstantIndex(loc, lowering.indexType(), rewriter, 0);

    // The base can be a boxed reference or a raw reference
    if (auto boxTy = coor.ref().getType().dyn_cast<BoxType>()) {
      if (coor.getNumOperands() == 2) {
        auto coorPtr = *coor.coor().begin();
        auto s = coorPtr.getDefiningOp();
        if (s && isa<LenParamIndexOp>(s)) {
          mlir::Value lenParam = operands[1]; // byte offset
          auto vty = voidPtrTy();
          auto bc = rewriter.create<mlir::LLVM::BitcastOp>(loc, vty, base);
          auto uty = unwrap(ty);
          auto gep = genGEP(loc, uty, rewriter, bc, lenParam);
          rewriter.replaceOpWithNewOp<mlir::LLVM::BitcastOp>(coor, uty, gep);
          return matchSuccess();
        }
      }
      auto c0_ = genConstantOffset(loc, rewriter, 0);
      auto pty = unwrap(convertType(boxTy.getEleTy())).getPointerTo();
      // Extract the boxed reference
      auto p = genGEP(loc, pty, rewriter, base, c0, c0_);
      // base = box->data : ptr
      base = rewriter.create<mlir::LLVM::LoadOp>(loc, pty, p);
    }

    SmallVector<mlir::Value, 8> offs{c0};
    auto indices = operands.drop_front(1);
    offs.append(indices.begin(), indices.end());
    if (noFieldIndexOps(coor.coor())) {
      // do not need to lower any field index ops, so use a GEP
      rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(coor, ty, base, offs);
      return matchSuccess();
    }

    // lower the field index ops by walking the indices
    auto bty = coor.ref().getType().cast<BoxType>();
    mlir::Type baseTy = ReferenceType::get(bty.getEleTy());
    SmallVector<mlir::Value, 8> args{c0};
    args.append(coor.coor().begin(), coor.coor().end());

    mlir::Value retval = base;
    assert(offs.size() == args.size() && "must have same arity");
    unsigned pos = 0;
    for (std::size_t i = 0, sz{offs.size()}; i != sz; ++i) {
      assert(pos <= i);
      if (auto defop = args[i].getDefiningOp())
        if (auto field = dyn_cast<FieldIndexOp>(defop)) {
          auto memTy = unwrap(convertType(baseTy)).getPointerTo();
          mlir::Value gep = retval;
          if (i - pos > 0)
            gep = genGEP(loc, memTy, rewriter, gep, arguments(offs, pos, i));
          auto bc =
              rewriter.create<mlir::LLVM::BitcastOp>(loc, voidPtrTy(), gep);
          auto gep_ = genGEP(loc, voidPtrTy(), rewriter, bc, offs[i]);
          pos = i + 1;
          baseTy = baseTy.cast<RecordType>().getType(field.field_id());
          retval = rewriter.create<mlir::LLVM::BitcastOp>(
              loc, convertType(baseTy), gep_);
          continue;
        }
      if (auto ptrTy = baseTy.dyn_cast<ReferenceType>()) {
        baseTy = ptrTy.getEleTy();
      } else if (auto ptrTy = baseTy.dyn_cast<fir::PointerType>()) {
        baseTy = ptrTy.getEleTy();
      } else if (auto ptrTy = baseTy.dyn_cast<HeapType>()) {
        baseTy = ptrTy.getEleTy();
      } else if (auto arrTy = baseTy.dyn_cast<SequenceType>()) {
        // FIXME: unchecked advance over array dims
        i += arrTy.getDimension() - 1;
        baseTy = arrTy.getEleTy();
      } else if (auto strTy = baseTy.dyn_cast<RecordType>()) {
        baseTy = strTy.getType(getIntValue(offs[i]));
      } else if (auto strTy = baseTy.dyn_cast<mlir::TupleType>()) {
        baseTy = strTy.getType(getIntValue(offs[i]));
      } else {
        assert(false && "unhandled type");
      }
    }
    if (pos < offs.size())
      retval = genGEP(loc, unwrap(ty), rewriter, retval,
                      arguments(offs, pos, offs.size()));
    rewriter.replaceOp(coor, retval);
    return matchSuccess();
  }

  SmallVector<mlir::Value, 8> arguments(ArrayRef<mlir::Value> vec, unsigned s,
                                        unsigned e) const {
    return {vec.begin() + s, vec.begin() + e};
  }

  int64_t getIntValue(mlir::Value val) const {
    if (val)
      if (auto defop = val.getDefiningOp())
        if (auto constOp = dyn_cast<mlir::ConstantIntOp>(defop))
          return constOp.getValue();
    assert(false && "must be a constant");
    return 0;
  }
};

/// convert a field index to a runtime function that computes the byte offset of
/// the dynamic field
struct FieldIndexOpConversion : public FIROpConversion<fir::FieldIndexOp> {
  using FIROpConversion::FIROpConversion;

  // NB: most field references should be resolved by this point
  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto field = mlir::cast<FieldIndexOp>(op);
    // call the compiler generated function to determine the byte offset of
    // the field at runtime
    auto symAttr =
        mlir::SymbolRefAttr::get(methodName(field), field.getContext());
    SmallVector<mlir::NamedAttribute, 1> attrs{
        rewriter.getNamedAttr("callee", symAttr)};
    auto ty = lowering.offsetType();
    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(field, ty, operands, attrs);
    return matchSuccess();
  }

  // constructing the name of the method
  inline static std::string methodName(FieldIndexOp field) {
    Twine fldName = field.field_id();
    // note: using std::string to dodge a bug in g++ 7.4.0
    std::string tyName = field.on_type().cast<RecordType>().getName().str();
    Twine methodName = "_QQOFFSETOF_" + tyName + "_" + fldName;
    return methodName.str();
  }
};

struct LenParamIndexOpConversion
    : public FIROpConversion<fir::LenParamIndexOp> {
  using FIROpConversion::FIROpConversion;

  // FIXME: this should be specialized by the runtime target
  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto lenp = mlir::cast<LenParamIndexOp>(op);
    auto ity = lowering.indexType();
    auto onty = lenp.getOnType();
    // size of portable descriptor
    const unsigned boxsize = 24; // FIXME
    unsigned offset = boxsize;
    // add the size of the rows of triples
    if (auto arr = onty.dyn_cast<SequenceType>()) {
      offset += 3 * arr.getDimension();
    }
    // advance over some addendum fields
    const unsigned addendumOffset{sizeof(void *) + sizeof(uint64_t)};
    offset += addendumOffset;
    // add the offset into the LENs
    offset += 0; // FIXME
    auto attr = rewriter.getI64IntegerAttr(offset);
    rewriter.replaceOpWithNewOp<mlir::LLVM::ConstantOp>(lenp, ity, attr);
    return matchSuccess();
  }
};

/// lower the fir.end operation to a null (erasing it)
struct FirEndOpConversion : public FIROpConversion<FirEndOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, {});
    return matchSuccess();
  }
};

/// lower a gendims operation into a sequence of writes to a temp
/// TODO: should this be returning a value or a ref? A !fir.dims object has
/// very restricted application
struct GenDimsOpConversion : public FIROpConversion<GenDimsOp> {
  using FIROpConversion::FIROpConversion;

  // gendims(args:index, ...) ==> %v = ... : [size x <3 x index>]
  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto gendims = mlir::cast<GenDimsOp>(op);
    auto loc = gendims.getLoc();
    auto ty = convertType(gendims.getType());
    auto ptrTy = unwrap(ty).getPointerTo();
    auto alloca = genAlloca(loc, ptrTy, defaultAlign, rewriter);
    unsigned offIndex = 0;
    auto c0 = genConstantOffset(loc, rewriter, 0);
    auto ipty = lowering.indexType().getPointerTo();
    for (auto opd : operands) {
      auto offset = genConstantOffset(loc, rewriter, offIndex++);
      auto gep = genGEP(loc, ipty, rewriter, alloca, c0, c0, offset);
      rewriter.create<mlir::LLVM::StoreOp>(loc, opd, gep);
    }
    rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(gendims, ptrTy, alloca);
    return matchSuccess();
  }

  // Generate an alloca of size `size` and cast it to type `toTy`
  mlir::LLVM::AllocaOp
  genAlloca(mlir::Location loc, mlir::LLVM::LLVMType toTy, unsigned alignment,
            mlir::ConversionPatternRewriter &rewriter) const {
    auto thisPt = rewriter.saveInsertionPoint();
    auto *thisBlock = rewriter.getInsertionBlock();
    auto func = mlir::cast<mlir::LLVM::LLVMFuncOp>(thisBlock->getParentOp());
    rewriter.setInsertionPointToStart(&func.front());
    auto size = genConstantOffset(loc, rewriter, 1);
    auto rv = rewriter.create<mlir::LLVM::AllocaOp>(loc, toTy, size, alignment);
    rewriter.restoreInsertionPoint(thisPt);
    return rv;
  }
};

/// lower a type descriptor to a global constant
struct GenTypeDescOpConversion : public FIROpConversion<GenTypeDescOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto gentypedesc = mlir::cast<GenTypeDescOp>(op);
    auto loc = gentypedesc.getLoc();
    auto inTy = gentypedesc.getInType();
    auto name = consName(rewriter, inTy);
    auto gty = unwrap(convertType(inTy));
    auto pty = gty.getPointerTo();
    auto module = gentypedesc.getParentOfType<mlir::ModuleOp>();
    createGlobal(loc, module, name, gty, rewriter);
    rewriter.replaceOpWithNewOp<mlir::LLVM::AddressOfOp>(gentypedesc, pty,
                                                         name);
    return matchSuccess();
  }

  std::string consName(mlir::ConversionPatternRewriter &rewriter,
                       mlir::Type type) const {
    if (auto d = type.dyn_cast<RecordType>()) {
      auto name = d.getName();
      auto pair = NameUniquer::deconstruct(name);
      return lowering.uniquer.doTypeDescriptor(
          pair.second.modules, pair.second.host, pair.second.name,
          pair.second.kinds);
    }
    assert(false);
    return {};
  }
};

struct GlobalLenOpConversion : public FIROpConversion<GlobalLenOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto globalentry = mlir::cast<GlobalLenOp>(op);
    TODO(globalentry);
    return matchSuccess();
  }
};

struct HasValueOpConversion : public FIROpConversion<fir::HasValueOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, operands, llvm::None,
                                                op->getAttrs());
    return matchSuccess();
  }
};

struct GlobalOpConversion : public FIROpConversion<fir::GlobalOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto global = mlir::cast<fir::GlobalOp>(op);
    auto tyAttr = unwrap(convertType(global.getType()));
    auto loc = global.getLoc();
    mlir::Attribute initAttr{};
    if (global.initval())
      initAttr = global.initval().getValue();
    auto g = rewriter.create<mlir::LLVM::GlobalOp>(
        loc, tyAttr, global.constant(), mlir::LLVM::Linkage::External,
        global.sym_name(), initAttr);
    auto &gr = g.getInitializerRegion();
    rewriter.inlineRegionBefore(global.region(), gr, gr.end());
    rewriter.eraseOp(global);
    return matchSuccess();
  }
};

// convert to LLVM IR dialect `load`
struct LoadOpConversion : public FIROpConversion<fir::LoadOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto load = mlir::cast<fir::LoadOp>(op);
    auto ty = convertType(load.getType());
    auto at = load.getAttrs();
    rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, ty, operands, at);
    return matchSuccess();
  }
};

// FIXME: how do we want to enforce this in LLVM-IR?
struct NoReassocOpConversion : public FIROpConversion<NoReassocOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto noreassoc = mlir::cast<NoReassocOp>(op);
    noreassoc.replaceAllUsesWith(operands[0]);
    rewriter.replaceOp(noreassoc, {});
    return matchSuccess();
  }
};

void genCaseLadderStep(mlir::Location loc, mlir::Value cmp, mlir::Block *dest,
                       OperandTy destOps,
                       mlir::ConversionPatternRewriter &rewriter) {
  auto *thisBlock = rewriter.getInsertionBlock();
  auto *newBlock = rewriter.createBlock(dest);
  rewriter.setInsertionPointToEnd(thisBlock);
  SmallVector<mlir::Block *, 2> dest_{dest, newBlock};
  SmallVector<mlir::ValueRange, 2> destOps_{destOps, {}};
  rewriter.create<mlir::LLVM::CondBrOp>(loc, mlir::ValueRange{cmp}, dest_,
                                        destOps_);
  rewriter.setInsertionPointToEnd(newBlock);
}

/// Conversion of `fir.select_case`
///
/// TODO: lowering of CHARACTER type cases
struct SelectCaseOpConversion : public FIROpConversion<SelectCaseOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  ArrayRef<mlir::Block *> destinations,
                  ArrayRef<OperandTy> destOperands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto selectcase = mlir::cast<SelectCaseOp>(op);
    auto conds = selectcase.getNumConditions();
    auto attrName = SelectCaseOp::AttrName;
    auto caseAttr = selectcase.getAttrOfType<mlir::ArrayAttr>(attrName);
    auto cases = caseAttr.getValue();
    // Type can be CHARACTER, INTEGER, or LOGICAL (C1145)
    auto ty = selectcase.getSelector().getType();
    (void)ty;
    auto &selector = operands[0];
    unsigned nextOp = 1;
    auto loc = selectcase.getLoc();
    assert(conds > 0 && "selectcase must have cases");
    for (unsigned t = 0; t != conds; ++t) {
      auto &attr = cases[t];
      if (attr.dyn_cast_or_null<fir::PointIntervalAttr>()) {
        auto cmp = rewriter.create<mlir::LLVM::ICmpOp>(
            loc, mlir::LLVM::ICmpPredicate::eq, selector, operands[nextOp++]);
        genCaseLadderStep(loc, cmp, destinations[t], destOperands[t], rewriter);
        continue;
      }
      if (attr.dyn_cast_or_null<fir::LowerBoundAttr>()) {
        auto cmp = rewriter.create<mlir::LLVM::ICmpOp>(
            loc, mlir::LLVM::ICmpPredicate::sle, operands[nextOp++], selector);
        genCaseLadderStep(loc, cmp, destinations[t], destOperands[t], rewriter);
        continue;
      }
      if (attr.dyn_cast_or_null<fir::UpperBoundAttr>()) {
        auto cmp = rewriter.create<mlir::LLVM::ICmpOp>(
            loc, mlir::LLVM::ICmpPredicate::sle, selector, operands[nextOp++]);
        genCaseLadderStep(loc, cmp, destinations[t], destOperands[t], rewriter);
        continue;
      }
      if (attr.dyn_cast_or_null<fir::ClosedIntervalAttr>()) {
        auto cmp = rewriter.create<mlir::LLVM::ICmpOp>(
            loc, mlir::LLVM::ICmpPredicate::sle, operands[nextOp++], selector);
        auto *thisBlock = rewriter.getInsertionBlock();
        auto *newBlock1 = rewriter.createBlock(destinations[t]);
        auto *newBlock2 = rewriter.createBlock(destinations[t]);
        rewriter.setInsertionPointToEnd(thisBlock);
        SmallVector<mlir::Block *, 2> dests{newBlock1, newBlock2};
        SmallVector<mlir::ValueRange, 2> destOps{{}, {}};
        rewriter.create<mlir::LLVM::CondBrOp>(loc, mlir::ValueRange{cmp}, dests,
                                              destOps);
        rewriter.setInsertionPointToEnd(newBlock1);
        auto cmp_ = rewriter.create<mlir::LLVM::ICmpOp>(
            loc, mlir::LLVM::ICmpPredicate::sle, selector, operands[nextOp++]);
        SmallVector<mlir::Block *, 2> dest2{destinations[t], newBlock2};
        SmallVector<mlir::ValueRange, 2> destOp2{destOperands[t], {}};
        rewriter.create<mlir::LLVM::CondBrOp>(loc, mlir::ValueRange{cmp_},
                                              dest2, destOp2);
        rewriter.setInsertionPointToEnd(newBlock2);
        continue;
      }
      assert(attr.dyn_cast_or_null<mlir::UnitAttr>());
      assert((t + 1 == conds) && "unit must be last");
      rewriter.replaceOpWithNewOp<mlir::LLVM::BrOp>(
          selectcase, mlir::ValueRange{}, destinations[t],
          mlir::ValueRange{destOperands[t]});
    }
    return matchSuccess();
  }
};

template <typename OP>
void selectMatchAndRewrite(FIRToLLVMTypeConverter &lowering,
                           mlir::Operation *op, OperandTy operands,
                           ArrayRef<mlir::Block *> destinations,
                           ArrayRef<OperandTy> destOperands,
                           mlir::ConversionPatternRewriter &rewriter) {
  auto select = mlir::cast<OP>(op);

  // We could target the LLVM switch instruction, but it isn't part of the
  // LLVM IR dialect.  Create an if-then-else ladder instead.
  auto conds = select.getNumConditions();
  auto attrName = OP::AttrName;
  auto caseAttr = select.template getAttrOfType<mlir::ArrayAttr>(attrName);
  auto cases = caseAttr.getValue();
  auto ty = select.getSelector().getType();
  auto ity = lowering.convertType(ty);
  auto &selector = operands[0];
  auto loc = select.getLoc();
  assert(conds > 0 && "select must have cases");
  for (unsigned t = 0, end{conds}; t != end; ++t) {
    auto &attr = cases[t];
    if (auto intAttr = attr.template dyn_cast_or_null<mlir::IntegerAttr>()) {
      auto ci = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, ity, rewriter.getIntegerAttr(ty, intAttr.getInt()));
      auto cmp = rewriter.create<mlir::LLVM::ICmpOp>(
          loc, mlir::LLVM::ICmpPredicate::eq, selector, ci);
      genCaseLadderStep(loc, cmp, destinations[t], destOperands[t], rewriter);
      continue;
    }
    assert(attr.template dyn_cast_or_null<mlir::UnitAttr>());
    assert((t + 1 == conds) && "unit must be last");
    rewriter.replaceOpWithNewOp<mlir::LLVM::BrOp>(
        select, mlir::ValueRange{}, destinations[t],
        mlir::ValueRange{destOperands[t]});
  }
}

/// conversion of fir::SelectOp to an if-then-else ladder
struct SelectOpConversion : public FIROpConversion<fir::SelectOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  ArrayRef<mlir::Block *> destinations,
                  ArrayRef<OperandTy> destOperands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    selectMatchAndRewrite<fir::SelectOp>(lowering, op, operands, destinations,
                                         destOperands, rewriter);
    return matchSuccess();
  }
};

/// conversion of fir::SelectRankOp to an if-then-else ladder
struct SelectRankOpConversion : public FIROpConversion<SelectRankOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  ArrayRef<mlir::Block *> destinations,
                  ArrayRef<OperandTy> destOperands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    selectMatchAndRewrite<fir::SelectRankOp>(
        lowering, op, operands, destinations, destOperands, rewriter);
    return matchSuccess();
  }
};

// SelectTypeOp should have already been lowered
struct SelectTypeOpConversion : public FIROpConversion<SelectTypeOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  ArrayRef<mlir::Block *> destinations,
                  ArrayRef<OperandTy> destOperands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto selecttype = mlir::cast<SelectTypeOp>(op);
    TODO(selecttype);
    return matchSuccess();
  }
};

// convert to LLVM IR dialect `store`
struct StoreOpConversion : public FIROpConversion<fir::StoreOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto store = mlir::cast<fir::StoreOp>(op);
    rewriter.replaceOpWithNewOp<mlir::LLVM::StoreOp>(store, operands[0],
                                                     operands[1]);
    return matchSuccess();
  }
};

// cons an extractvalue on a tuple value, returning value at element `x`
mlir::LLVM::ExtractValueOp genExtractValueWithIndex(
    mlir::Location loc, mlir::Value tuple, mlir::LLVM::LLVMType ty,
    mlir::ConversionPatternRewriter &rewriter, mlir::MLIRContext *ctx, int x) {
  auto cx = mlir::ArrayAttr::get(rewriter.getI32IntegerAttr(x), ctx);
  auto xty = ty.getStructElementType(x);
  return rewriter.create<mlir::LLVM::ExtractValueOp>(loc, xty, tuple, cx);
}

// unbox a CHARACTER box value, yielding its components
struct UnboxCharOpConversion : public FIROpConversion<UnboxCharOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto unboxchar = mlir::cast<UnboxCharOp>(op);
    auto *ctx = unboxchar.getContext();
    auto loc = unboxchar.getLoc();
    auto tuple = operands[0];
    auto ty = unwrap(tuple.getType());
    mlir::Value ptr =
        genExtractValueWithIndex(loc, tuple, ty, rewriter, ctx, 0);
    mlir::Value len =
        genExtractValueWithIndex(loc, tuple, ty, rewriter, ctx, 1);
    std::vector<mlir::Value> repls = {ptr, len};
    unboxchar.replaceAllUsesWith(repls);
    rewriter.eraseOp(unboxchar);
    return matchSuccess();
  }
};

// generate a GEP into a structure and load the element at position `x`
mlir::LLVM::LoadOp genLoadWithIndex(mlir::Location loc, mlir::Value tuple,
                                    mlir::LLVM::LLVMType ty,
                                    mlir::ConversionPatternRewriter &rewriter,
                                    mlir::LLVM::LLVMType oty,
                                    mlir::LLVM::ConstantOp c0, int x) {
  auto ax = rewriter.getI32IntegerAttr(x);
  auto cx = rewriter.create<mlir::LLVM::ConstantOp>(loc, oty, ax);
  auto xty = ty.getStructElementType(x);
  auto gep = genGEP(loc, xty.getPointerTo(), rewriter, tuple, c0, cx);
  return rewriter.create<mlir::LLVM::LoadOp>(loc, xty, gep);
}

// unbox a generic box reference, yielding its components
struct UnboxOpConversion : public FIROpConversion<UnboxOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto unbox = mlir::cast<UnboxOp>(op);
    auto loc = unbox.getLoc();
    auto tuple = operands[0];
    auto ty = unwrap(tuple.getType());
    auto oty = lowering.offsetType();
    auto c0 = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, oty, rewriter.getI32IntegerAttr(0));
    mlir::Value ptr = genLoadWithIndex(loc, tuple, ty, rewriter, oty, c0, 0);
    mlir::Value len = genLoadWithIndex(loc, tuple, ty, rewriter, oty, c0, 1);
    mlir::Value ver = genLoadWithIndex(loc, tuple, ty, rewriter, oty, c0, 2);
    mlir::Value rank = genLoadWithIndex(loc, tuple, ty, rewriter, oty, c0, 3);
    mlir::Value type = genLoadWithIndex(loc, tuple, ty, rewriter, oty, c0, 4);
    mlir::Value attr = genLoadWithIndex(loc, tuple, ty, rewriter, oty, c0, 5);
    mlir::Value xtra = genLoadWithIndex(loc, tuple, ty, rewriter, oty, c0, 6);
    // FIXME: add dims, etc.
    std::vector<mlir::Value> repls{ptr, len, ver, rank, type, attr, xtra};
    unbox.replaceAllUsesWith(repls);
    rewriter.eraseOp(unbox);
    return matchSuccess();
  }
};

// unbox a procedure box value, yielding its components
struct UnboxProcOpConversion : public FIROpConversion<UnboxProcOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto unboxproc = mlir::cast<UnboxProcOp>(op);
    auto *ctx = unboxproc.getContext();
    auto loc = unboxproc.getLoc();
    auto tuple = operands[0];
    auto ty = unwrap(tuple.getType());
    mlir::Value ptr =
        genExtractValueWithIndex(loc, tuple, ty, rewriter, ctx, 0);
    mlir::Value host =
        genExtractValueWithIndex(loc, tuple, ty, rewriter, ctx, 1);
    std::vector<mlir::Value> repls{ptr, host};
    unboxproc.replaceAllUsesWith(repls);
    rewriter.eraseOp(unboxproc);
    return matchSuccess();
  }
};

// convert to LLVM IR dialect `undef`
struct UndefOpConversion : public FIROpConversion<UndefOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto undef = mlir::cast<UndefOp>(op);
    rewriter.replaceOpWithNewOp<mlir::LLVM::UndefOp>(
        undef, convertType(undef.getType()));
    return matchSuccess();
  }
};

// convert to LLVM IR dialect `unreachable`
struct UnreachableOpConversion : public FIROpConversion<UnreachableOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto unreach = mlir::cast<UnreachableOp>(op);
    rewriter.replaceOpWithNewOp<mlir::LLVM::UnreachableOp>(
        unreach, operands, None, None, unreach.getAttrs());
    return matchSuccess();
  }
};

//
// Primitive operations on Real (floating-point) types
//

/// Convert a floating-point primitive
template <typename BINOP, typename LLVMOP>
void lowerRealBinaryOp(mlir::Operation *op, OperandTy operands,
                       mlir::ConversionPatternRewriter &rewriter,
                       FIRToLLVMTypeConverter &lowering) {
  auto binop = cast<BINOP>(op);
  auto ty = lowering.convertType(binop.getType());
  rewriter.replaceOpWithNewOp<LLVMOP>(binop, ty, operands);
}

struct AddfOpConversion : public FIROpConversion<fir::AddfOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    lowerRealBinaryOp<fir::AddfOp, mlir::LLVM::FAddOp>(op, operands, rewriter,
                                                       lowering);
    return matchSuccess();
  }
};
struct SubfOpConversion : public FIROpConversion<fir::SubfOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    lowerRealBinaryOp<fir::SubfOp, mlir::LLVM::FSubOp>(op, operands, rewriter,
                                                       lowering);
    return matchSuccess();
  }
};
struct MulfOpConversion : public FIROpConversion<fir::MulfOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    lowerRealBinaryOp<fir::MulfOp, mlir::LLVM::FMulOp>(op, operands, rewriter,
                                                       lowering);
    return matchSuccess();
  }
};
struct DivfOpConversion : public FIROpConversion<fir::DivfOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    lowerRealBinaryOp<fir::DivfOp, mlir::LLVM::FDivOp>(op, operands, rewriter,
                                                       lowering);
    return matchSuccess();
  }
};
struct ModfOpConversion : public FIROpConversion<fir::ModfOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    lowerRealBinaryOp<fir::ModfOp, mlir::LLVM::FRemOp>(op, operands, rewriter,
                                                       lowering);
    return matchSuccess();
  }
};

struct NegfOpConversion : public FIROpConversion<fir::NegfOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto neg = mlir::cast<fir::NegfOp>(op);
    auto ty = convertType(neg.getType());
    rewriter.replaceOpWithNewOp<mlir::LLVM::FNegOp>(neg, ty, operands);
    return matchSuccess();
  }
};

//
// Primitive operations on Complex types
//

/// Generate code for complex addition/subtraction
template <typename LLVMOP, typename OPTY>
mlir::LLVM::InsertValueOp complexSum(OPTY sumop, OperandTy opnds,
                                     mlir::ConversionPatternRewriter &rewriter,
                                     FIRToLLVMTypeConverter &lowering) {
  auto a = opnds[0];
  auto b = opnds[1];
  auto loc = sumop.getLoc();
  auto ctx = sumop.getContext();
  auto c0 = mlir::ArrayAttr::get(rewriter.getI32IntegerAttr(0), ctx);
  auto c1 = mlir::ArrayAttr::get(rewriter.getI32IntegerAttr(1), ctx);
  auto ty = lowering.convertType(sumop.getType());
  auto x = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ty, a, c0);
  auto x_ = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ty, b, c0);
  auto rx = rewriter.create<LLVMOP>(loc, ty, x, x_);
  auto y = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ty, a, c1);
  auto y_ = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ty, b, c1);
  auto ry = rewriter.create<LLVMOP>(loc, ty, y, y_);
  auto r = rewriter.create<mlir::LLVM::UndefOp>(loc, ty);
  auto r_ = rewriter.create<mlir::LLVM::InsertValueOp>(loc, ty, r, rx, c0);
  return rewriter.create<mlir::LLVM::InsertValueOp>(loc, ty, r_, ry, c1);
}

struct AddcOpConversion : public FIROpConversion<fir::AddcOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // result: (x + x') + i(y + y')
    auto addc = cast<fir::AddcOp>(op);
    auto r = complexSum<mlir::LLVM::FAddOp>(addc, operands, rewriter, lowering);
    addc.replaceAllUsesWith(r.getResult());
    rewriter.replaceOp(addc, r.getResult());
    return matchSuccess();
  }
};

struct SubcOpConversion : public FIROpConversion<fir::SubcOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // result: (x - x') + i(y - y')
    auto subc = mlir::cast<fir::SubcOp>(op);
    auto r = complexSum<mlir::LLVM::FSubOp>(subc, operands, rewriter, lowering);
    subc.replaceAllUsesWith(r.getResult());
    rewriter.replaceOp(subc, r.getResult());
    return matchSuccess();
  }
};

/// Inlined complex multiply
struct MulcOpConversion : public FIROpConversion<fir::MulcOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto mulc = mlir::cast<fir::MulcOp>(op);
    // FIXME: should this just call __muldc3 ?
    // result: (xx'-yy')+i(xy'+yx')
    auto a = operands[0];
    auto b = operands[1];
    auto loc = mulc.getLoc();
    auto ctx = mulc.getContext();
    auto c0 = mlir::ArrayAttr::get(rewriter.getI32IntegerAttr(0), ctx);
    auto c1 = mlir::ArrayAttr::get(rewriter.getI32IntegerAttr(1), ctx);
    auto ty = convertType(mulc.getType());
    auto x = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ty, a, c0);
    auto x_ = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ty, b, c0);
    auto xx_ = rewriter.create<mlir::LLVM::FMulOp>(loc, ty, x, x_);
    auto y = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ty, a, c1);
    auto yx_ = rewriter.create<mlir::LLVM::FMulOp>(loc, ty, y, x_);
    auto y_ = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ty, b, c1);
    auto xy_ = rewriter.create<mlir::LLVM::FMulOp>(loc, ty, x, y_);
    auto ri = rewriter.create<mlir::LLVM::FAddOp>(loc, ty, xy_, yx_);
    auto yy_ = rewriter.create<mlir::LLVM::FMulOp>(loc, ty, y, y_);
    auto rr = rewriter.create<mlir::LLVM::FSubOp>(loc, ty, xx_, yy_);
    auto ra = rewriter.create<mlir::LLVM::UndefOp>(loc, ty);
    auto r_ = rewriter.create<mlir::LLVM::InsertValueOp>(loc, ty, ra, rr, c0);
    auto r = rewriter.create<mlir::LLVM::InsertValueOp>(loc, ty, r_, ri, c1);
    mulc.replaceAllUsesWith(r.getResult());
    rewriter.replaceOp(mulc, r.getResult());
    return matchSuccess();
  }
};

/// Inlined complex division
struct DivcOpConversion : public FIROpConversion<fir::DivcOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto divc = mlir::cast<fir::DivcOp>(op);
    // FIXME: should this just call __divdc3 ?
    // result: ((xx'+yy')/d) + i((yx'-xy')/d) where d = x'x' + y'y'
    auto a = operands[0];
    auto b = operands[1];
    auto loc = divc.getLoc();
    auto ctx = divc.getContext();
    auto c0 = mlir::ArrayAttr::get(rewriter.getI32IntegerAttr(0), ctx);
    auto c1 = mlir::ArrayAttr::get(rewriter.getI32IntegerAttr(1), ctx);
    auto ty = convertType(divc.getType());
    auto x = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ty, a, c0);
    auto x_ = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ty, b, c0);
    auto xx_ = rewriter.create<mlir::LLVM::FMulOp>(loc, ty, x, x_);
    auto x_x_ = rewriter.create<mlir::LLVM::FMulOp>(loc, ty, x_, x_);
    auto y = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ty, a, c1);
    auto yx_ = rewriter.create<mlir::LLVM::FMulOp>(loc, ty, y, x_);
    auto y_ = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ty, b, c1);
    auto xy_ = rewriter.create<mlir::LLVM::FMulOp>(loc, ty, x, y_);
    auto yy_ = rewriter.create<mlir::LLVM::FMulOp>(loc, ty, y, y_);
    auto y_y_ = rewriter.create<mlir::LLVM::FMulOp>(loc, ty, y_, y_);
    auto d = rewriter.create<mlir::LLVM::FAddOp>(loc, ty, x_x_, y_y_);
    auto rrn = rewriter.create<mlir::LLVM::FAddOp>(loc, ty, xx_, yy_);
    auto rin = rewriter.create<mlir::LLVM::FSubOp>(loc, ty, yx_, xy_);
    auto rr = rewriter.create<mlir::LLVM::FDivOp>(loc, ty, rrn, d);
    auto ri = rewriter.create<mlir::LLVM::FDivOp>(loc, ty, rin, d);
    auto ra = rewriter.create<mlir::LLVM::UndefOp>(loc, ty);
    auto r_ = rewriter.create<mlir::LLVM::InsertValueOp>(loc, ty, ra, rr, c0);
    auto r = rewriter.create<mlir::LLVM::InsertValueOp>(loc, ty, r_, ri, c1);
    divc.replaceAllUsesWith(r.getResult());
    rewriter.replaceOp(divc, r.getResult());
    return matchSuccess();
  }
};

struct NegcOpConversion : public FIROpConversion<fir::NegcOp> {
  using FIROpConversion::FIROpConversion;

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op, OperandTy operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto neg = mlir::cast<fir::NegcOp>(op);
    auto ctxt = neg.getContext();
    auto ty = convertType(neg.getType());
    auto loc = neg.getLoc();
    auto c0 = mlir::ArrayAttr::get(rewriter.getI32IntegerAttr(0), ctxt);
    auto &o0 = operands[0];
    auto rp = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ty, o0, c0);
    auto nrp = rewriter.create<mlir::LLVM::FNegOp>(loc, ty, rp);
    auto c1 = mlir::ArrayAttr::get(rewriter.getI32IntegerAttr(1), ctxt);
    auto ip = rewriter.create<mlir::LLVM::ExtractValueOp>(loc, ty, o0, c1);
    auto nip = rewriter.create<mlir::LLVM::FNegOp>(loc, ty, ip);
    auto r = rewriter.create<mlir::LLVM::InsertValueOp>(loc, ty, o0, nrp, c0);
    rewriter.replaceOpWithNewOp<mlir::LLVM::InsertValueOp>(neg, ty, r, nip, c1);
    return matchSuccess();
  }
};

// Lower a SELECT operation into a cascade of conditional branches. The last
// case must be the `true` condition.
/// Convert FIR dialect to LLVM dialect
///
/// This pass lowers all FIR dialect operations to LLVM IR dialect.  An
/// MLIR pass is used to lower residual Std dialect to LLVM IR dialect.
struct FIRToLLVMLoweringPass : public mlir::ModulePass<FIRToLLVMLoweringPass> {
  FIRToLLVMLoweringPass(NameUniquer &uniquer) : uniquer{uniquer} {}

  void runOnModule() override {
    if (ClDisableFirToLLVMIR)
      return;

    auto *context{&getContext()};
    FIRToLLVMTypeConverter typeConverter{context, uniquer};
    mlir::OwningRewritePatternList patterns;
    patterns.insert<
        AddcOpConversion, AddfOpConversion, AddrOfOpConversion,
        AllocaOpConversion, AllocMemOpConversion, BoxAddrOpConversion,
        BoxCharLenOpConversion, BoxDimsOpConversion, BoxEleSizeOpConversion,
        BoxIsAllocOpConversion, BoxIsArrayOpConversion, BoxIsPtrOpConversion,
        BoxProcHostOpConversion, BoxRankOpConversion, BoxTypeDescOpConversion,
        CallOpConversion, CmpcOpConversion, CmpfOpConversion,
        ConstcOpConversion, ConstfOpConversion, ConvertOpConversion,
        CoordinateOpConversion, DispatchOpConversion, DispatchTableOpConversion,
        DivcOpConversion, DivfOpConversion, DTEntryOpConversion,
        EmboxCharOpConversion, EmboxOpConversion, EmboxProcOpConversion,
        FieldIndexOpConversion, FirEndOpConversion, ExtractValueOpConversion,
        FreeMemOpConversion, GenDimsOpConversion, GenTypeDescOpConversion,
        GlobalLenOpConversion, GlobalOpConversion, HasValueOpConversion,
        InsertValueOpConversion, LenParamIndexOpConversion, LoadOpConversion,
        ModfOpConversion, MulcOpConversion, MulfOpConversion, NegcOpConversion,
        NegfOpConversion, NoReassocOpConversion, SelectCaseOpConversion,
        SelectOpConversion, SelectRankOpConversion, SelectTypeOpConversion,
        StoreOpConversion, StringLitOpConversion, SubcOpConversion,
        SubfOpConversion, UnboxCharOpConversion, UnboxOpConversion,
        UnboxProcOpConversion, UndefOpConversion, UnreachableOpConversion>(
        context, typeConverter);
    mlir::populateStdToLLVMConversionPatterns(typeConverter, patterns);
    mlir::ConversionTarget target{*context};
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();

    // required NOP stubs for applying a full conversion
    target.addDynamicallyLegalOp<mlir::ModuleOp>(
        [&](mlir::ModuleOp) { return true; });
    target.addDynamicallyLegalOp<mlir::ModuleTerminatorOp>(
        [&](mlir::ModuleTerminatorOp) { return true; });

    genDispatchTableMap();

    // apply the patterns
    if (mlir::failed(mlir::applyFullConversion(
            getModule(), target, std::move(patterns), &typeConverter))) {
      mlir::emitError(mlir::UnknownLoc::get(context),
                      "error in converting to LLVM-IR dialect\n");
      signalPassFailure();
    }
  }

private:
  void genDispatchTableMap() {
    for (auto dt : getModule().getOps<DispatchTableOp>()) {
      // FIXME
      (void)dt;
    }
  }

  NameUniquer &uniquer;
};

/// Lower from LLVM IR dialect to proper LLVM-IR and dump the module
struct LLVMIRLoweringPass : public mlir::ModulePass<LLVMIRLoweringPass> {
  LLVMIRLoweringPass(raw_ostream &output) : output{output} {}

  void runOnModule() override {
    if (ClDisableLLVM)
      return;

    if (auto llvmModule{mlir::translateModuleToLLVMIR(getModule())}) {
      output << *llvmModule;
      return;
    }

    auto ctxt{getModule().getContext()};
    mlir::emitError(mlir::UnknownLoc::get(ctxt), "could not emit LLVM-IR\n");
    signalPassFailure();
  }

private:
  raw_ostream &output;
};

} // namespace

std::unique_ptr<mlir::Pass>
fir::createFIRToLLVMPass(fir::NameUniquer &nameUniquer) {
  return std::make_unique<FIRToLLVMLoweringPass>(nameUniquer);
}

std::unique_ptr<mlir::Pass>
fir::createLLVMDialectToLLVMPass(raw_ostream &output) {
  return std::make_unique<LLVMIRLoweringPass>(output);
}

// Register the FIR to LLVM-IR pass
static mlir::PassRegistration<FIRToLLVMLoweringPass>
    passLowFIR("fir-to-llvmir",
               "Conversion of the FIR dialect to the LLVM-IR dialect", [] {
                 NameUniquer dummy;
                 return std::make_unique<FIRToLLVMLoweringPass>(dummy);
               });
