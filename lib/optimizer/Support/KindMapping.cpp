//===-- KindMapping.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/optimizer/Support/KindMapping.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/CommandLine.h"

/// Allow the user to set the FIR intrinsic type kind value to LLVM type
/// mappings.  Note that these are not mappings from kind values to any
/// other MLIR dialect, only to LLVM IR. The default values follow the f18
/// front-end kind mappings.

using Bitsize = fir::KindMapping::Bitsize;
using KindTy = fir::KindMapping::KindTy;
using LLVMTypeID = fir::KindMapping::LLVMTypeID;
using MatchResult = fir::KindMapping::MatchResult;

static llvm::cl::opt<std::string> ClKindMapping(
    "kind-mapping", llvm::cl::desc("kind mapping string to set kind precision"),
    llvm::cl::value_desc("kind-mapping-string"), llvm::cl::init(""));

namespace fir {
namespace {

/// Integral types default to the kind value being the size of the value in
/// bytes. The default is to scale from bytes to bits.
Bitsize defaultScalingKind(KindTy kind) {
  const unsigned BITS_IN_BYTE = 8;
  return kind * BITS_IN_BYTE;
}

/// Floating-point types default to the kind value being the size of the value
/// in bytes. The default is to translate kinds of 2, 4, 8, 10, and 16 to a
/// valid llvm::Type::TypeID value. Otherwise, the default is FloatTyID.
LLVMTypeID defaultRealKind(KindTy kind) {
  switch (kind) {
  case 2:
    return LLVMTypeID::HalfTyID;
  case 4:
    return LLVMTypeID::FloatTyID;
  case 8:
    return LLVMTypeID::DoubleTyID;
  case 10:
    return LLVMTypeID::X86_FP80TyID;
  case 16:
    return LLVMTypeID::FP128TyID;
  default:
    return LLVMTypeID::FloatTyID;
  }
}

// lookup the kind-value given the defaults, the mappings, and a KIND key
template <typename RT, char KEY>
RT doLookup(std::function<RT(KindTy)> def,
            const std::map<char, std::map<KindTy, RT>> &map, KindTy kind) {
  auto iter = map.find(KEY);
  if (iter != map.end()) {
    auto iter2 = iter->second.find(kind);
    if (iter2 != iter->second.end())
      return iter2->second;
  }
  return def(kind);
}

// do a lookup for INTERGER, LOGICAL, or CHARACTER
template <char KEY, typename MAP>
Bitsize getIntegerLikeBitsize(KindTy kind, const MAP &map) {
  return doLookup<Bitsize, KEY>(defaultScalingKind, map, kind);
}

// do a lookup for REAL or COMPLEX
template <char KEY, typename MAP>
LLVMTypeID getFloatLikeTypeID(KindTy kind, const MAP &map) {
  return doLookup<LLVMTypeID, KEY>(defaultRealKind, map, kind);
}

MatchResult parseCode(char &code, const char *&ptr) {
  if (*ptr != 'a' && *ptr != 'c' && *ptr != 'i' && *ptr != 'l' && *ptr != 'r')
    return {};
  code = *ptr++;
  return {true};
}

template <char ch>
MatchResult parseSingleChar(const char *&ptr) {
  if (*ptr != ch)
    return {};
  ++ptr;
  return {true};
}

MatchResult parseColon(const char *&ptr) { return parseSingleChar<':'>(ptr); }

MatchResult parseComma(const char *&ptr) { return parseSingleChar<','>(ptr); }

MatchResult parseInt(unsigned &result, const char *&ptr) {
  const char *beg = ptr;
  while (*ptr >= '0' && *ptr <= '9')
    ptr++;
  if (beg == ptr)
    return {};
  llvm::StringRef ref(beg, ptr - beg);
  int temp;
  if (ref.consumeInteger(10, temp))
    return {};
  result = temp;
  return {true};
}

bool matchString(const char *&ptr, llvm::StringRef literal) {
  llvm::StringRef s(ptr);
  if (s.startswith(literal)) {
    ptr += literal.size();
    return true;
  }
  return false;
}

MatchResult parseTypeID(LLVMTypeID &result, const char *&ptr) {
  if (matchString(ptr, "Half")) {
    result = LLVMTypeID::HalfTyID;
    return {true};
  }
  if (matchString(ptr, "Float")) {
    result = LLVMTypeID::FloatTyID;
    return {true};
  }
  if (matchString(ptr, "Double")) {
    result = LLVMTypeID::DoubleTyID;
    return {true};
  }
  if (matchString(ptr, "X86_FP80")) {
    result = LLVMTypeID::X86_FP80TyID;
    return {true};
  }
  if (matchString(ptr, "FP128")) {
    result = LLVMTypeID::FP128TyID;
    return {true};
  }
  return {};
}

} // namespace

KindMapping::KindMapping(mlir::MLIRContext *context, llvm::StringRef map)
    : context{context} {
  parse(map);
}

KindMapping::KindMapping(mlir::MLIRContext *context)
    : KindMapping{context, ClKindMapping} {}

MatchResult KindMapping::badMapString(const llvm::Twine &ptr) {
  auto unknown = mlir::UnknownLoc::get(context);
  mlir::emitError(unknown, ptr);
  return {};
}

MatchResult KindMapping::parse(llvm::StringRef kindMap) {
  if (kindMap.empty())
    return {true};
  const char *srcPtr = kindMap.begin();
  while (true) {
    char code = '\0';
    KindTy kind = 0;
    if (parseCode(code, srcPtr) || parseInt(kind, srcPtr))
      return badMapString(srcPtr);
    if (code == 'a' || code == 'i' || code == 'l') {
      Bitsize bits = 0;
      if (parseColon(srcPtr) || parseInt(bits, srcPtr))
        return badMapString(srcPtr);
      intMap[code][kind] = bits;
    } else if (code == 'r' || code == 'c') {
      LLVMTypeID id{};
      if (parseColon(srcPtr) || parseTypeID(id, srcPtr))
        return badMapString(srcPtr);
      floatMap[code][kind] = id;
    } else {
      return badMapString(srcPtr);
    }
    if (parseComma(srcPtr))
      break;
  }
  if (*srcPtr)
    return badMapString(srcPtr);
  return {true};
}

Bitsize KindMapping::getCharacterBitsize(KindTy kind) {
  return getIntegerLikeBitsize<'a'>(kind, intMap);
}

Bitsize KindMapping::getIntegerBitsize(KindTy kind) {
  return getIntegerLikeBitsize<'i'>(kind, intMap);
}

Bitsize KindMapping::getLogicalBitsize(KindTy kind) {
  return getIntegerLikeBitsize<'l'>(kind, intMap);
}

LLVMTypeID KindMapping::getRealTypeID(KindTy kind) {
  return getFloatLikeTypeID<'r'>(kind, floatMap);
}

LLVMTypeID KindMapping::getComplexTypeID(KindTy kind) {
  return getFloatLikeTypeID<'c'>(kind, floatMap);
}

} // namespace fir
