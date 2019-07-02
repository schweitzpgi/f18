// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "fir-type.h"
#include "fir-dialect.h"
#include "../common/idioms.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Parser.h"

namespace Br = Fortran::mlbridge;
namespace Co = Fortran::common;
namespace L = llvm;
namespace M = mlir;

using namespace Fortran;
using namespace Fortran::mlbridge;

namespace {

// Tokens

enum class TokenKind {
  error,
  eof,
  leftang,
  rightang,
  leftparen,
  rightparen,
  leftbrace,
  rightbrace,
  leftbracket,
  rightbracket,
  colon,
  comma,
  period,
  eroteme,
  ecphoneme,
  star,
  arrow,
  ident,
  string,
  intlit,
};

struct Token {
  TokenKind kind;
  L::StringRef text;
};

// Lexer

class Lexer {
public:
  Lexer(L::StringRef source) : srcBuff{source}, srcPtr{source.begin()} {}

  // consume and return next token from the input. input is advanced to after
  // the token.
  Token lexToken();

  // peek ahead to the next non-whitespace character, leaving it on the input
  // stream
  char nextChar() {
    skipWhitespace();
    if (atEnd()) {
      return '\0';
    }
    return *srcPtr;
  }

  // advance the input stream `count` characters
  void advance(unsigned count = 1) {
    while (count--) {
      if (atEnd()) {
        break;
      }
      ++srcPtr;
    }
  }

  const char *getMarker() { return srcPtr; }

private:
  void skipWhitespace() {
    while (!atEnd()) {
      switch (*srcPtr) {
      case ' ':
      case '\f':
      case '\n':
      case '\r':
      case '\t':
      case '\v': ++srcPtr; continue;
      default: break;
      }
      break;
    }
  }

  Token formToken(TokenKind kind, const char *tokStart) {
    return Token{kind, L::StringRef(tokStart, srcPtr - tokStart)};
  }

  Token emitError(const char *loc, const L::Twine &message) {
    return formToken(TokenKind::error, loc);
  }

  bool atEnd() const { return srcPtr == srcBuff.end(); }

  Token lexIdent(const char *tokStart);
  Token lexNumber(const char *tokStart);
  Token lexString(const char *tokStart);

  L::StringRef srcBuff;
  const char *srcPtr;
};

Token Lexer::lexToken() {
  skipWhitespace();
  if (atEnd()) {
    return formToken(TokenKind::eof, "");
  }

  const char *tokStart = srcPtr;
  switch (*srcPtr++) {
  case '<': return formToken(TokenKind::leftang, tokStart);
  case '>': return formToken(TokenKind::rightang, tokStart);
  case '{': return formToken(TokenKind::leftbrace, tokStart);
  case '}': return formToken(TokenKind::rightbrace, tokStart);
  case '[': return formToken(TokenKind::leftbracket, tokStart);
  case ']': return formToken(TokenKind::rightbracket, tokStart);
  case '(': return formToken(TokenKind::leftparen, tokStart);
  case ')': return formToken(TokenKind::rightparen, tokStart);
  case ':': return formToken(TokenKind::colon, tokStart);
  case ',': return formToken(TokenKind::comma, tokStart);
  case '"': return lexString(tokStart + 1);
  case '-':
    if (*srcPtr == '>') {
      srcPtr++;
      return formToken(TokenKind::arrow, tokStart);
    }
    return lexNumber(tokStart);
  case '+': return lexNumber(tokStart + 1);
  case '!': return formToken(TokenKind::ecphoneme, tokStart);
  case '?': return formToken(TokenKind::eroteme, tokStart);
  case '*': return formToken(TokenKind::star, tokStart);
  case '.': return formToken(TokenKind::period, tokStart);
  default:
    if (std::isalpha(*tokStart)) {
      return lexIdent(tokStart);
    }
    if (std::isdigit(*tokStart)) {
      return lexNumber(tokStart);
    }
    return emitError(tokStart, "unexpected character");
  }
}

Token Lexer::lexString(const char *tokStart) {
  while (!atEnd() && *srcPtr != '"') {
    ++srcPtr;
  }
  Token token{formToken(TokenKind::string, tokStart)};
  ++srcPtr;
  return token;
}

Token Lexer::lexIdent(const char *tokStart) {
  while (!atEnd() && (std::isalpha(*srcPtr) || std::isdigit(*srcPtr))) {
    ++srcPtr;
  }
  return formToken(TokenKind::ident, tokStart);
}

Token Lexer::lexNumber(const char *tokStart) {
  while (!atEnd() && std::isdigit(*srcPtr)) {
    ++srcPtr;
  }
  return formToken(TokenKind::intlit, tokStart);
}

class auto_counter {
public:
  explicit auto_counter(int &ref) : counter(ref) { ++counter; }
  ~auto_counter() { --counter; }

private:
  auto_counter() = delete;
  auto_counter(const auto_counter &) = delete;
  int &counter;
};

/// A FIROpsDialect instance uses a FIRTypeParser object to parse and
/// instantiate all FIR types from .fir files.
class FIRTypeParser {
public:
  FIRTypeParser(FIROpsDialect *dialect, L::StringRef rawData, M::Location loc)
    : context{dialect->getContext()}, lexer{rawData}, loc{loc} {}

  M::Type parseType();

protected:
  void emitError(M::Location loc, const L::Twine &msg) {
    getContext()->emitError(loc, msg);
  }

  bool consumeToken(TokenKind tk, const L::Twine &msg) {
    auto token = lexer.lexToken();
    if (token.kind != tk) {
      emitError(loc, msg);
      return true;  // error!
    }
    return false;
  }

  bool consumeChar(char ch, const L::Twine &msg) {
    auto lexCh = lexer.nextChar();
    if (lexCh != ch) {
      emitError(loc, msg);
      return true;  // error!
    }
    lexer.advance();
    return false;
  }

  template<typename A> A parseIntLitSingleton(const char *msg) {
    if (consumeToken(TokenKind::leftang, "expected '<' in type")) {
      return {};
    }
    auto token{lexer.lexToken()};
    if (token.kind != TokenKind::intlit) {
      emitError(loc, msg);
      return {};
    }
    KindTy kind;
    if (token.text.getAsInteger(0, kind)) {
      emitError(loc, "expected integer constant");
      return {};
    }
    if (consumeToken(TokenKind::rightang, "expected '>' in type")) {
      return {};
    }
    if (checkAtEnd()) return {};
    return A::get(getContext(), kind);
  }

  // `<` kind `>`
  template<typename A> A parseKindSingleton() {
    return parseIntLitSingleton<A>("expected kind parameter");
  }

  // `<` rank `>`
  template<typename A> A parseRankSingleton() {
    return parseIntLitSingleton<A>("expected rank parameter");
  }

  // '<' type '>'
  template<typename A> A parseTypeSingleton() {
    if (consumeToken(TokenKind::leftang, "expected '<' in type")) {
      return {};
    }
    auto ofTy = parseNextType();
    if (!ofTy) {
      emitError(loc, "expected type parameter");
      return {};
    }
    if (consumeToken(TokenKind::rightang, "expected '>' in type")) {
      return {};
    }
    if (checkAtEnd()) return {};
    return A::get(ofTy);
  }

  const char *advanceFuncType();
  std::pair<bool, M::Type> advanceNonFuncType(const Token &lastTkn);
  M::Type parseNextType();

  bool checkAtEnd() {
    if (!recursiveCall) {
      auto token = lexer.lexToken();
      if (token.kind != TokenKind::eof) {
        emitError(loc, "unexpected extra characters");
        return true;
      }
    }
    return false;
  }

  // `box` `<` type `>`
  FIRBoxType parseBox() { return parseTypeSingleton<FIRBoxType>(); }

  // `boxchar` `<` fir-character-type `>`
  FIRBoxCharType parseBoxChar() {
    if (consumeToken(TokenKind::leftang, "expected '<' in type")) {
      return {};
    }
    auto ofTy = parseNextType();
    if (!ofTy) {
      emitError(loc, "expected type parameter");
      return {};
    }
    if (consumeToken(TokenKind::rightang, "expected '>' in type")) {
      return {};
    }
    if (checkAtEnd()) return {};
    if (auto eleTy{ofTy.dyn_cast<FIRCharacterType>()}) {
      return FIRBoxCharType::get(eleTy);
    }
    emitError(loc, "subtype must be !fir.char<K>");
    return {};
  }

  // `boxproc` `<` return-type `>`
  FIRBoxProcType parseBoxProc() { return parseTypeSingleton<FIRBoxProcType>(); }

  // `char` `<` kind `>`
  FIRCharacterType parseCharacter() {
    return parseKindSingleton<FIRCharacterType>();
  }

  // `dims` `<` rank `>`
  FIRDimsType parseDims() { return parseRankSingleton<FIRDimsType>(); }

  // `field`
  FIRFieldType parseField() {
    if (checkAtEnd()) return {};
    return FIRFieldType::get(getContext());
  }

  // `logical` `<` kind `>`
  FIRLogicalType parseLogical() { return parseKindSingleton<FIRLogicalType>(); }

  // `real` `<` kind `>`
  FIRRealType parseReal() { return parseKindSingleton<FIRRealType>(); }

  // `ref` `<` type `>`
  FIRReferenceType parseReference() {
    return parseTypeSingleton<FIRReferenceType>();
  }

  FIRSequenceType::Shape parseShape();
  FIRSequenceType parseSequence();

  FIRTupleType::TypeList parseTypeList();
  FIRTupleType::KindList parseKindList();
  FIRTupleType parseTuple();

  // `tdesc` `<` type `>`
  FIRTypeDescType parseTypeDesc() {
    return parseTypeSingleton<FIRTypeDescType>();
  }

  // `void` ==> `std.tuple` `<` `>`
  M::Type parseVoid() {
    if (checkAtEnd()) return {};
    return M::TupleType::get(getContext());
  }

  M::MLIRContext *getContext() const { return context; }

private:
  M::MLIRContext *context;
  Lexer lexer;
  M::Location loc;
  int recursiveCall{-1};
};

const char *FIRTypeParser::advanceFuncType() {
  Token token;
  int lparen = 0;
  while (true) {
    token = lexer.lexToken();
    if (token.kind == TokenKind::leftparen) {
      lparen++;
    } else if (token.kind == TokenKind::rightparen) {
      if (lparen == 0) break;
      lparen--;
    }
  }
  token = lexer.lexToken();
  if (token.kind != TokenKind::arrow) {
    return lexer.getMarker();
  }
  advanceNonFuncType(lexer.lexToken());
  return lexer.getMarker();
}

std::pair<bool, M::Type> FIRTypeParser::advanceNonFuncType(
    const Token &lastTkn) {
  if (lastTkn.kind == TokenKind::ecphoneme) {  // `!`
    auto token = lexer.lexToken();
    if (token.kind == TokenKind::ident) {  // `!` ident
      if (token.text == "fir") {
        token = lexer.lexToken();
        if (token.kind == TokenKind::period) {  // `!fir.`
          return {true, parseType()};
        }
      }
      char nextCh = lexer.nextChar();
      if (nextCh == '.') {  // `!` ident `.`
        lexer.lexToken();  // `.`
        lexer.lexToken();  // ident
        nextCh = lexer.nextChar();
      }
      if (nextCh == '<') {  // `!` ... `<` ... `>`
        token = lexer.lexToken();
        int lang = 0;
        while (true) {
          token = lexer.lexToken();
          if (token.kind == TokenKind::leftang) {
            lang++;
          } else if (token.kind == TokenKind::rightang) {
            if (lang == 0) break;
            lang--;
          }
        }
      }
    }
  } else if (lastTkn.kind == TokenKind::ident) {  // ident
    char nextCh = lexer.nextChar();
    if (nextCh == '<') {  // ident `<` ... `>`
      auto token = lexer.lexToken();
      int lang = 0;
      while (true) {
        token = lexer.lexToken();
        if (token.kind == TokenKind::leftang) {
          lang++;
        } else if (token.kind == TokenKind::rightang) {
          if (lang == 0) break;
          lang--;
        }
      }
    }
  } else if (lastTkn.kind == TokenKind::leftparen) {  // `(` ... `)`
    int lparen = 0;
    while (true) {
      auto token = lexer.lexToken();
      if (token.kind == TokenKind::leftparen) {
        lparen++;
      } else if (token.kind == TokenKind::rightparen) {
        if (lparen == 0) break;
        lparen--;
      }
    }
  } else {
    // This doesn't really look like a valid type. Let the standard type parser
    // deal with it.
  }
  return {false, {}};
}

// If this is a `!fir.x` type then recursively parse it now, otherwise figure
// out its extent and call into the standard type parser.
M::Type FIRTypeParser::parseNextType() {
  const char *marker = lexer.getMarker();
  Token token = lexer.lexToken();
  if (token.kind == TokenKind::leftparen) {
    auto count = advanceFuncType() - marker;
    assert(count > 0);
    return M::parseType(L::StringRef(marker, count), getContext());
  }
  auto pair = advanceNonFuncType(token);
  if (pair.first) {
    return pair.second;
  }
  auto count = lexer.getMarker() - marker;
  assert(count > 0);
  return M::parseType(L::StringRef(marker, count), getContext());
}

// Parses either `*` `:`
//            or (int int int | `?`) (`,` (int int int | `?`))* `:`
FIRSequenceType::Shape FIRTypeParser::parseShape() {
  FIRSequenceType::Bounds bounds;
  int lower, extent, stride;
  Token token = lexer.lexToken();
  if (token.kind == TokenKind::star) {
    token = lexer.lexToken();
    if (token.kind != TokenKind::colon) {
      emitError(loc, "expected '*' to be followed by ':'");
      return {};
    }
    return {FIRSequenceType::Unknown{}};
  }
  while (true) {
    if (token.kind != TokenKind::eroteme) {
      goto shape_spec;
    }
    bounds.emplace_back(FIRSequenceType::Unknown{});
    goto check_comma;
  shape_spec:
    if (token.kind != TokenKind::intlit) {
      emitError(loc, "expected an integer or '?' in shape specification");
      return {};
    }
    token.text.getAsInteger(10, lower);
    token = lexer.lexToken();
    if (token.kind != TokenKind::intlit) {
      emitError(loc, "expected a second integer");
      return {};
    }
    token.text.getAsInteger(10, extent);
    token = lexer.lexToken();
    if (token.kind != TokenKind::intlit) {
      emitError(loc, "expected a third integer");
      return {};
    }
    token.text.getAsInteger(10, stride);
    bounds.emplace_back(FIRSequenceType::BoundInfo{lower, extent, stride});
  check_comma:
    token = lexer.lexToken();
    if (token.kind == TokenKind::colon) {
      return {bounds};
    }
    if (token.kind != TokenKind::comma) {
      emitError(loc, "expected a ',' or ':' after integer-triple");
      return {};
    }
    token = lexer.lexToken();
  }
}

// bounds ::= lo extent stride | `?`
// `array` `<` bounds (`,` bounds)* `:` type `>`
FIRSequenceType FIRTypeParser::parseSequence() {
  if (consumeToken(TokenKind::leftang, "expected '<' in array type")) {
    return {};
  }
  auto shape = parseShape();
  M::Type eleTy = parseNextType();
  if (!eleTy) {
    emitError(loc, "invalid element type");
    return {};
  }
  if (consumeToken(TokenKind::rightang, "expected '>' in array type")) {
    return {};
  }
  if (checkAtEnd()) {
    return {};
  }
  return FIRSequenceType::get(shape, eleTy);
}

// Parses: string `:` type (',' string `:` type)* '}'
FIRTupleType::TypeList FIRTypeParser::parseTypeList() {
  FIRTupleType::TypeList result;
  while (true) {
    auto name{lexer.lexToken()};
    if (name.kind != TokenKind::string) {
      emitError(loc, "expected string");
      return {};
    }
    if (consumeToken(TokenKind::colon, "expected colon")) {
      return {};
    }
    auto memTy{parseNextType()};
    result.emplace_back(name.text, memTy);
    auto token{lexer.lexToken()};
    if (token.kind == TokenKind::rightbrace) {
      return result;
    }
    if (token.kind != TokenKind::comma) {
      emitError(loc, "expected ','");
      return {};
    }
  }
}

// Parses: string `:` integer (',' string `:` integer)* ']'
FIRTupleType::KindList FIRTypeParser::parseKindList() {
  FIRTupleType::KindList result;
  while (true) {
    auto name{lexer.lexToken()};
    if (name.kind != TokenKind::string) {
      emitError(loc, "expected string");
      return {};
    }
    if (consumeToken(TokenKind::colon, "expected colon")) {
      return {};
    }
    Token kind = lexer.lexToken();
    KindTy kindValue;
    if ((kind.kind != TokenKind::intlit) ||
        kind.text.getAsInteger(0, kindValue)) {
      emitError(loc, "expected integer constant");
      return {};
    }
    result.emplace_back(name.text, kindValue);
    auto token{lexer.lexToken()};
    if (token.kind == TokenKind::rightbracket) {
      return result;
    }
    if (token.kind != TokenKind::comma) {
      emitError(loc, "expected ','");
      return {};
    }
  }
}

// Fortran derived type
// `type` `<` name
//           (`,` `{` id `:` type (`,` id `:` type)* `}`)?
//           (`,` `[` id `:` kind (`,` id `:` kind)* `]`)? '>'
FIRTupleType FIRTypeParser::parseTuple() {
  if (consumeToken(TokenKind::leftang, "expected '<' in type type")) {
    return {};
  }
  auto name{lexer.lexToken()};
  if (name.kind != TokenKind::string) {
    emitError(loc, "expected a \"string\" as name of derived type");
    return {};
  }
  auto token = lexer.lexToken();
  FIRTupleType::TypeList typeList;
  FIRTupleType::KindList kindList;
  if (token.kind != TokenKind::comma) {
    // neither optional list present
    goto check_close;
  }
  token = lexer.lexToken();
  if (token.kind != TokenKind::leftbrace) {
    // {type-list} not present, must be a [kind-list]
    goto check_kind_list;
  }
  typeList = parseTypeList();
  token = lexer.lexToken();
  if (token.kind != TokenKind::comma) {
    // no [kind-list]
    goto check_close;
  }
  token = lexer.lexToken();
check_kind_list:
  if (token.kind != TokenKind::leftbracket) {
    emitError(loc, "expected {type-list} or [kind-list] after ','");
    return {};
  }
  kindList = parseKindList();
  token = lexer.lexToken();
check_close:
  if (token.kind != TokenKind::rightang) {
    emitError(loc, "expected '>' in type type");
    return {};
  }
  if (checkAtEnd()) {
    return {};
  }
  return FIRTupleType::get(getContext(), name.text, kindList, typeList);
}

M::Type FIRTypeParser::parseType() {
  auto_counter c{recursiveCall};
  auto token = lexer.lexToken();
  if (token.kind == TokenKind::ident) {
    if (token.text == "ref") return parseReference();
    if (token.text == "array") return parseSequence();
    if (token.text == "char") return parseCharacter();
    if (token.text == "logical") return parseLogical();
    if (token.text == "real") return parseReal();
    if (token.text == "type") return parseTuple();
    if (token.text == "box") return parseBox();
    if (token.text == "boxchar") return parseBoxChar();
    if (token.text == "boxproc") return parseBoxProc();
    if (token.text == "dims") return parseDims();
    if (token.text == "tdesc") return parseTypeDesc();
    if (token.text == "field") return parseField();
    if (token.text == "void") return parseVoid();
    emitError(loc, "not a known fir type");
    return {};
  }
  emitError(loc, "invalid token");
  return {};
}

}  // anonymous

namespace Fortran::mlbridge::detail {

// Type storage classes

/// `CHARACTER` storage
struct FIRCharacterTypeStorage : public M::TypeStorage {
  using KeyTy = KindTy;

  static unsigned hashKey(const KeyTy &key) { return L::hash_combine(key); }

  bool operator==(const KeyTy &key) const { return key == getFKind(); }

  static FIRCharacterTypeStorage *construct(
      M::TypeStorageAllocator &allocator, KindTy kind) {
    auto *storage = allocator.allocate<FIRCharacterTypeStorage>();
    return new (storage) FIRCharacterTypeStorage{kind};
  }

  KindTy getFKind() const { return kind; }

protected:
  KindTy kind;

private:
  FIRCharacterTypeStorage() = delete;
  explicit FIRCharacterTypeStorage(KindTy kind) : kind{kind} {}
};

struct FIRDimsTypeStorage : public M::TypeStorage {
  using KeyTy = unsigned;

  static unsigned hashKey(const KeyTy &key) { return L::hash_combine(key); }

  bool operator==(const KeyTy &key) const { return key == getRank(); }

  static FIRDimsTypeStorage *construct(
      M::TypeStorageAllocator &allocator, unsigned rank) {
    auto *storage = allocator.allocate<FIRDimsTypeStorage>();
    return new (storage) FIRDimsTypeStorage{rank};
  }

  unsigned getRank() const { return rank; }

protected:
  unsigned rank;

private:
  FIRDimsTypeStorage() = delete;
  explicit FIRDimsTypeStorage(unsigned rank) : rank{rank} {}
};

/// The type of a derived type part reference
struct FIRFieldTypeStorage : public M::TypeStorage {
  using KeyTy = KindTy;

  static unsigned hashKey(const KeyTy &) { return L::hash_combine(0); }

  bool operator==(const KeyTy &) const { return true; }

  static FIRFieldTypeStorage *construct(
      M::TypeStorageAllocator &allocator, KindTy) {
    auto *storage = allocator.allocate<FIRFieldTypeStorage>();
    return new (storage) FIRFieldTypeStorage{0};
  }

private:
  FIRFieldTypeStorage() = delete;
  explicit FIRFieldTypeStorage(KindTy) {}
};

/// `LOGICAL` storage
struct FIRLogicalTypeStorage : public M::TypeStorage {
  using KeyTy = KindTy;

  static unsigned hashKey(const KeyTy &key) { return L::hash_combine(key); }

  bool operator==(const KeyTy &key) const { return key == getFKind(); }

  static FIRLogicalTypeStorage *construct(
      M::TypeStorageAllocator &allocator, KindTy kind) {
    auto *storage = allocator.allocate<FIRLogicalTypeStorage>();
    return new (storage) FIRLogicalTypeStorage{kind};
  }

  KindTy getFKind() const { return kind; }

protected:
  KindTy kind;

private:
  FIRLogicalTypeStorage() = delete;
  explicit FIRLogicalTypeStorage(KindTy kind) : kind{kind} {}
};

/// `REAL` storage (for reals of unsupported sizes)
struct FIRRealTypeStorage : public M::TypeStorage {
  using KeyTy = KindTy;

  static unsigned hashKey(const KeyTy &key) { return L::hash_combine(key); }

  bool operator==(const KeyTy &key) const { return key == getFKind(); }

  static FIRRealTypeStorage *construct(
      M::TypeStorageAllocator &allocator, KindTy kind) {
    auto *storage = allocator.allocate<FIRRealTypeStorage>();
    return new (storage) FIRRealTypeStorage{kind};
  }

  KindTy getFKind() const { return kind; }

protected:
  KindTy kind;

private:
  FIRRealTypeStorage() = delete;
  explicit FIRRealTypeStorage(KindTy kind) : kind{kind} {}
};

/// Boxed object (a Fortran descriptor)
struct FIRBoxTypeStorage : public M::TypeStorage {
  using KeyTy = M::Type;

  static unsigned hashKey(const KeyTy &key) { return L::hash_combine(key); }

  bool operator==(const KeyTy &key) const { return key == getElementType(); }

  static FIRBoxTypeStorage *construct(
      M::TypeStorageAllocator &allocator, M::Type eleTy) {
    assert(eleTy && "element type is null");
    auto *storage = allocator.allocate<FIRBoxTypeStorage>();
    return new (storage) FIRBoxTypeStorage{eleTy};
  }

  M::Type getElementType() const { return eleTy; }

protected:
  M::Type eleTy;

private:
  FIRBoxTypeStorage() = delete;
  explicit FIRBoxTypeStorage(M::Type eleTy) : eleTy{eleTy} {}
};

/// Boxed CHARACTER object type
struct FIRBoxCharTypeStorage : public M::TypeStorage {
  using KeyTy = FIRCharacterType;

  static unsigned hashKey(const KeyTy &key) { return L::hash_combine(key); }

  bool operator==(const KeyTy &key) const { return key == getElementType(); }

  static FIRBoxCharTypeStorage *construct(
      M::TypeStorageAllocator &allocator, FIRCharacterType charTy) {
    assert(charTy && "element type is null");
    auto *storage = allocator.allocate<FIRBoxCharTypeStorage>();
    return new (storage) FIRBoxCharTypeStorage{charTy};
  }

  FIRCharacterType getElementType() const { return charTy; }

protected:
  FIRCharacterType charTy;

private:
  FIRBoxCharTypeStorage() = delete;
  explicit FIRBoxCharTypeStorage(FIRCharacterType charTy) : charTy{charTy} {}
};

/// Boxed PROCEDURE POINTER object type
struct FIRBoxProcTypeStorage : public M::TypeStorage {
  using KeyTy = M::Type;

  static unsigned hashKey(const KeyTy &key) { return L::hash_combine(key); }

  bool operator==(const KeyTy &key) const { return key == getElementType(); }

  static FIRBoxProcTypeStorage *construct(
      M::TypeStorageAllocator &allocator, M::Type eleTy) {
    assert(eleTy && "element type is null");
    auto *storage = allocator.allocate<FIRBoxProcTypeStorage>();
    return new (storage) FIRBoxProcTypeStorage{eleTy};
  }

  M::Type getElementType() const { return eleTy; }

protected:
  M::Type eleTy;

private:
  FIRBoxProcTypeStorage() = delete;
  explicit FIRBoxProcTypeStorage(M::Type eleTy) : eleTy{eleTy} {}
};

/// Pointer-like object storage
struct FIRReferenceTypeStorage : public M::TypeStorage {
  using KeyTy = M::Type;

  static unsigned hashKey(const KeyTy &key) { return L::hash_combine(key); }

  bool operator==(const KeyTy &key) const { return key == getElementType(); }

  static FIRReferenceTypeStorage *construct(
      M::TypeStorageAllocator &allocator, M::Type eleTy) {
    assert(eleTy && "element type is null");
    auto *storage = allocator.allocate<FIRReferenceTypeStorage>();
    return new (storage) FIRReferenceTypeStorage{eleTy};
  }

  M::Type getElementType() const { return eleTy; }

protected:
  M::Type eleTy;

private:
  FIRReferenceTypeStorage() = delete;
  explicit FIRReferenceTypeStorage(M::Type eleTy) : eleTy{eleTy} {}
};

/// Sequence-like object storage
struct FIRSequenceTypeStorage : public M::TypeStorage {
  using KeyTy = std::pair<FIRSequenceType::Shape, M::Type>;

  static unsigned hashKey(const KeyTy &key) {
    auto shapeHash{hash_value(std::get<FIRSequenceType::Shape>(key))};
    return L::hash_combine(shapeHash, std::get<M::Type>(key));
  }

  bool operator==(const KeyTy &key) const {
    return key == KeyTy{getShape(), getElementType()};
  }

  static FIRSequenceTypeStorage *construct(
      M::TypeStorageAllocator &allocator, const KeyTy &key) {
    auto *storage = allocator.allocate<FIRSequenceTypeStorage>();
    return new (storage) FIRSequenceTypeStorage{key.first, key.second};
  }

  FIRSequenceType::Shape getShape() const { return shape; }
  M::Type getElementType() const { return eleTy; }

protected:
  FIRSequenceType::Shape shape;
  M::Type eleTy;

private:
  FIRSequenceTypeStorage() = delete;
  explicit FIRSequenceTypeStorage(
      const FIRSequenceType::Shape &shape, M::Type eleTy)
    : shape{shape}, eleTy{eleTy} {}
};

/// Derived type storage
struct FIRTupleTypeStorage : public M::TypeStorage {
  using KeyTy = std::tuple<L::StringRef, L::ArrayRef<FIRTupleType::KindPair>,
      L::ArrayRef<FIRTupleType::TypePair>>;

  static unsigned hashKey(const KeyTy &key) {
    const L::ArrayRef<FIRTupleType::KindPair> &vec =
        std::get<L::ArrayRef<FIRTupleType::KindPair>>(key);
    return L::hash_combine(
        std::get<0>(key).str(), L::hash_combine_range(vec.begin(), vec.end()));
  }

  bool operator==(const KeyTy &key) const {
    return std::get<0>(key) == getName() &&
        std::get<L::ArrayRef<FIRTupleType::KindPair>>(key) == getKindList();
  }

  static FIRTupleTypeStorage *construct(
      M::TypeStorageAllocator &allocator, const KeyTy &key) {
    auto *storage = allocator.allocate<FIRTupleTypeStorage>();
    auto &name = std::get<0>(key);
    auto &kinds = std::get<L::ArrayRef<FIRTupleType::KindPair>>(key);
    auto &members = std::get<L::ArrayRef<FIRTupleType::TypePair>>(key);
    return new (storage) FIRTupleTypeStorage{name, kinds, members};
  }

  L::StringRef getName() const { return name; }

  // The KindList must be provided at construction for correct hash-consing
  L::ArrayRef<FIRTupleType::KindPair> getKindList() const { return kinds; }

  void setTypeList(L::ArrayRef<FIRTupleType::TypePair> list) { types = list; }
  L::ArrayRef<FIRTupleType::TypePair> getTypeList() const { return types; }

protected:
  std::string name;
  std::vector<FIRTupleType::KindPair> kinds;
  std::vector<FIRTupleType::TypePair> types;

private:
  FIRTupleTypeStorage() = delete;
  explicit FIRTupleTypeStorage(L::StringRef name,
      L::ArrayRef<FIRTupleType::KindPair> kinds,
      L::ArrayRef<FIRTupleType::TypePair> types)
    : name{name}, kinds{kinds}, types{types} {}
};

/// Type descriptor type storage
struct FIRTypeDescTypeStorage : public M::TypeStorage {
  using KeyTy = M::Type;

  static unsigned hashKey(const KeyTy &key) { return L::hash_combine(key); }

  bool operator==(const KeyTy &key) const { return key == getOfType(); }

  static FIRTypeDescTypeStorage *construct(
      M::TypeStorageAllocator &allocator, M::Type ofTy) {
    assert(ofTy && "descriptor type is null");
    auto *storage = allocator.allocate<FIRTypeDescTypeStorage>();
    return new (storage) FIRTypeDescTypeStorage{ofTy};
  }

  // The type described by this type descriptor instance
  M::Type getOfType() const { return ofTy; }

protected:
  M::Type ofTy;

private:
  FIRTypeDescTypeStorage() = delete;
  explicit FIRTypeDescTypeStorage(M::Type ofTy) : ofTy{ofTy} {}
};

}  // detail

// CHARACTER

FIRCharacterType Br::FIRCharacterType::get(M::MLIRContext *ctxt, KindTy kind) {
  return Base::get(ctxt, FIR_CHARACTER, kind * 8);
}

int Br::FIRCharacterType::getSizeInBits() const {
  return getImpl()->getFKind();
}

KindTy Br::FIRCharacterType::getFKind() const { return getSizeInBits() / 8; }

// Dims

FIRDimsType Br::FIRDimsType::get(M::MLIRContext *ctxt, unsigned rank) {
  return Base::get(ctxt, FIR_DIMS, rank);
}

unsigned Br::FIRDimsType::getRank() const { return getImpl()->getRank(); }

// Field

FIRFieldType Br::FIRFieldType::get(M::MLIRContext *ctxt, KindTy) {
  return Base::get(ctxt, FIR_FIELD, 0);
}

// LOGICAL

FIRLogicalType Br::FIRLogicalType::get(M::MLIRContext *ctxt, KindTy kind) {
  return Base::get(ctxt, FIR_LOGICAL, kind * 8);
}

int Br::FIRLogicalType::getSizeInBits() const { return getImpl()->getFKind(); }

KindTy Br::FIRLogicalType::getFKind() const { return getSizeInBits() / 8; }

// REAL

FIRRealType Br::FIRRealType::get(M::MLIRContext *ctxt, KindTy kind) {
  return Base::get(ctxt, FIR_REAL, kind * 8);
}

int Br::FIRRealType::getSizeInBits() const { return getImpl()->getFKind(); }

KindTy Br::FIRRealType::getFKind() const { return getSizeInBits() / 8; }

// Box<T>

FIRBoxType Br::FIRBoxType::get(M::Type elementType) {
  return Base::get(elementType.getContext(), FIR_BOX, elementType);
}

M::Type Br::FIRBoxType::getEleTy() const { return getImpl()->getElementType(); }

// BoxChar<C>

FIRBoxCharType Br::FIRBoxCharType::get(FIRCharacterType elementType) {
  return Base::get(elementType.getContext(), FIR_BOXCHAR, elementType);
}

FIRCharacterType Br::FIRBoxCharType::getEleTy() const {
  return getImpl()->getElementType();
}

// BoxProc<T>

FIRBoxProcType Br::FIRBoxProcType::get(M::Type elementType) {
  return Base::get(elementType.getContext(), FIR_BOXPROC, elementType);
}

M::Type Br::FIRBoxProcType::getEleTy() const {
  return getImpl()->getElementType();
}

// Reference<T>

FIRReferenceType Br::FIRReferenceType::get(M::Type elementType) {
  return Base::get(elementType.getContext(), FIR_REFERENCE, elementType);
}

M::Type Br::FIRReferenceType::getEleTy() const {
  return getImpl()->getElementType();
}

// Sequence<T>

FIRSequenceType Br::FIRSequenceType::get(
    const Shape &shape, M::Type elementType) {
  auto *ctxt = elementType.getContext();
  return Base::get(ctxt, FIR_SEQUENCE, shape, elementType);
}

M::Type Br::FIRSequenceType::getEleTy() const {
  return getImpl()->getElementType();
}

FIRSequenceType::Shape Br::FIRSequenceType::getShape() const {
  return getImpl()->getShape();
}

// compare if two shapes are equivalent
bool Br::operator==(
    const FIRSequenceType::Shape &sh_1, const FIRSequenceType::Shape &sh_2) {
  return std::visit(
      Co::visitors{
          [](const FIRSequenceType::Unknown &,
              const FIRSequenceType::Unknown &) { return true; },
          [](const FIRSequenceType::Bounds &bnd_1,
              const FIRSequenceType::Bounds &bnd_2) {
            if (bnd_1.size() != bnd_2.size()) return false;
            for (std::size_t i = 0, end = bnd_1.size(); i != end; ++i) {
              if (!std::visit(
                      Co::visitors{
                          [](const FIRSequenceType::Unknown &,
                              const FIRSequenceType::Unknown &) {
                            return true;
                          },
                          [](const FIRSequenceType::BoundInfo &info_1,
                              const FIRSequenceType::BoundInfo &info_2) {
                            return info_1.lower == info_2.lower &&
                                info_1.count == info_2.count &&
                                info_1.stride == info_2.stride;
                          },
                          [](auto &, auto &) { return false; },
                      },
                      bnd_1[i], bnd_2[i])) {
                return false;
              }
            }
            return true;
          },
          [](auto, auto) { return false; },
      },
      sh_1, sh_2);
}

// compute the hash of an extent (variant)
L::hash_code Br::hash_value(const FIRSequenceType::Extent &ext) {
  return std::visit(
      Co::visitors{
          [](const FIRSequenceType::Unknown &) { return L::hash_combine(0); },
          [](const FIRSequenceType::BoundInfo &info) {
            return L::hash_combine(info.lower, info.count, info.stride);
          },
      },
      ext);
}

// compute the hash of a shapes (variant)
L::hash_code Br::hash_value(const FIRSequenceType::Shape &sh) {
  return std::visit(
      Co::visitors{
          [](const FIRSequenceType::Unknown &) { return L::hash_combine(0); },
          [](const FIRSequenceType::Bounds &bnd) {
            return L::hash_combine_range(bnd.begin(), bnd.end());
          },
      },
      sh);
}

// Tuple<Ts...>

FIRTupleType Br::FIRTupleType::get(M::MLIRContext *ctxt, L::StringRef name,
    L::ArrayRef<KindPair> kindList, L::ArrayRef<TypePair> typeList) {
  return Base::get(ctxt, FIR_TUPLE, name, kindList, typeList);
}

L::StringRef Br::FIRTupleType::getName() { return getImpl()->getName(); }

FIRTupleType::TypeList Br::FIRTupleType::getTypeList() {
  return getImpl()->getTypeList();
}

FIRTupleType::KindList Br::FIRTupleType::getKindList() {
  return getImpl()->getKindList();
}

// Type descriptor

FIRTypeDescType Br::FIRTypeDescType::get(M::Type ofType) {
  return Base::get(ofType.getContext(), FIR_TYPEDESC, ofType);
}

M::Type Br::FIRTypeDescType::getOfTy() const { return getImpl()->getOfType(); }

// Implementation of the thin interface from dialect to type parser

M::Type Br::parseFirType(
    FIROpsDialect *dialect, L::StringRef rawData, M::Location loc) {
  FIRTypeParser parser{dialect, rawData, loc};
  return parser.parseType();
}
