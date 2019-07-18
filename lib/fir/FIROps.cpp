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

#include "fir/FIROps.h"
#include "fir/Attribute.h"
#include "fir/Type.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"

namespace L = llvm;
namespace M = mlir;

namespace fir {

// AllocaExpr

M::Type AllocaExpr::getAllocatedType() {
  return getType().cast<ReferenceType>().getEleTy();
}

// DispatchTableOp

void DispatchTableOp::build(M::Builder *builder, M::OperationState *result,
    L::StringRef name, M::Type type, L::ArrayRef<M::NamedAttribute> attrs) {
  result->addAttribute(
      M::SymbolTable::getSymbolAttrName(), builder->getStringAttr(name));
  for (const auto &pair : attrs) {
    result->addAttribute(pair.first, pair.second);
  }
}

M::ParseResult DispatchTableOp::parse(
    M::OpAsmParser *parser, M::OperationState *result) {
  // Parse the name as a symbol reference attribute.
  SymbolRefAttr nameAttr;
  if (parser->parseAttribute(
          nameAttr, M::SymbolTable::getSymbolAttrName(), result->attributes))
    return failure();

  // Convert the parsed name attr into a string attr.
  result->attributes.back().second =
      parser->getBuilder().getStringAttr(nameAttr.getValue());

  // Parse the optional table body.
  M::Region *body = result->addRegion();
  if (parser->parseOptionalRegion(*body,
          L::ArrayRef<M::OpAsmParser::OperandType>{}, L::ArrayRef<M::Type>{}))
    return M::failure();

  ensureTerminator(*body, parser->getBuilder(), result->location);
  return M::success();
}

void DispatchTableOp::print(M::OpAsmPrinter *p) {
  auto tableName =
      getAttrOfType<StringAttr>(M::SymbolTable::getSymbolAttrName()).getValue();
  *p << getOperationName() << " @" << tableName;

  Region &body = getOperation()->getRegion(0);
  if (!body.empty())
    p->printRegion(body, /*printEntryBlockArgs=*/false,
        /*printBlockTerminators=*/false);
}

M::LogicalResult DispatchTableOp::verify() { return M::success(); }

M::Region &DispatchTableOp::front() {
  return this->getOperation()->getRegion(0);
}

void DispatchTableOp::appendTableEntry(M::Operation *op) {
  assert(M::isa<fir::DTEntryOp>(*op) && "operation must be a DTEntryOp");
  front().front().push_back(op);
}

M::ParseResult parseCallOp(M::OpAsmParser *parser, M::OperationState *result) {
  M::FunctionType calleeType;
  L::SmallVector<M::OpAsmParser::OperandType, 4> operands;
  M::OpAsmParser::OperandType callee;
  auto calleeLoc = parser->getNameLoc();
  if (parser->parseOperand(callee)) {
    M::SymbolRefAttr calleeAttr;
    if (parser->parseAttribute(calleeAttr, "proc", result->attributes) ||
        parser->parseOperandList(operands, M::OpAsmParser::Delimiter::Paren) ||
        parser->parseOptionalAttributeDict(result->attributes) ||
        parser->parseColonType(calleeType) ||
        parser->addTypesToList(calleeType.getResults(), result->types) ||
        parser->resolveOperands(
            operands, calleeType.getInputs(), calleeLoc, result->operands))
      return M::failure();
  } else {
    result->attributes.push_back(parser->getBuilder().getNamedAttr(
        "proc", parser->getBuilder().getSymbolRefAttr("")));
    if (parser->getCurrentLocation(&calleeLoc) ||
        parser->parseOperandList(operands, M::OpAsmParser::Delimiter::Paren) ||
        parser->parseOptionalAttributeDict(result->attributes) ||
        parser->parseColonType(calleeType) ||
        parser->resolveOperand(callee, calleeType, result->operands) ||
        parser->resolveOperands(
            operands, calleeType.getInputs(), calleeLoc, result->operands) ||
        parser->addTypesToList(calleeType.getResults(), result->types))
      return M::failure();
  }
  return M::success();
}

mlir::ParseResult parseDispatchOp(
    mlir::OpAsmParser *parser, mlir::OperationState *result) {
  M::FunctionType calleeType;
  L::SmallVector<M::OpAsmParser::OperandType, 4> operands;
  auto calleeLoc = parser->getNameLoc();
  M::StringAttr calleeAttr;
  if (parser->parseAttribute(calleeAttr, "proc", result->attributes) ||
      parser->parseOperandList(operands, M::OpAsmParser::Delimiter::Paren) ||
      parser->parseOptionalAttributeDict(result->attributes) ||
      parser->parseColonType(calleeType) ||
      parser->addTypesToList(calleeType.getResults(), result->types) ||
      parser->resolveOperands(
          operands, calleeType.getInputs(), calleeLoc, result->operands))
    return M::failure();
  return M::success();
}

// DTEntryOp

mlir::ParseResult parseDTEntryOp(
    mlir::OpAsmParser *parser, mlir::OperationState *result) {
  M::SymbolRefAttr calleeAttr;
  M::StringAttr methodNameAttr;
  if (parser->parseBareName(methodNameAttr, "method", result->attributes) ||
      parser->parseComma() ||
      parser->parseAttribute(calleeAttr, "proc", result->attributes))
    return M::failure();
  return M::success();
}

// GlobalOp

void GlobalOp::build(M::Builder *builder, M::OperationState *result,
    L::StringRef name, M::Type type, L::ArrayRef<M::NamedAttribute> attrs) {
  result->addAttribute(getTypeAttrName(), builder->getTypeAttr(type));
  result->addAttribute(
      M::SymbolTable::getSymbolAttrName(), builder->getStringAttr(name));
  for (const auto &pair : attrs) {
    result->addAttribute(pair.first, pair.second);
  }
}

M::ParseResult GlobalOp::parse(
    M::OpAsmParser *parser, M::OperationState *result) {
  // Parse the name as a symbol reference attribute.
  SymbolRefAttr nameAttr;
  if (parser->parseAttribute(
          nameAttr, M::SymbolTable::getSymbolAttrName(), result->attributes))
    return failure();

  auto &builder = parser->getBuilder();
  result->attributes.back().second = builder.getStringAttr(nameAttr.getValue());

  M::Type globalType;
  if (parser->parseColonType(globalType)) {
    return M::failure();
  }
  result->addAttribute(getTypeAttrName(), builder.getTypeAttr(globalType));

  // Parse the optional initializer body.
  M::Region *body = result->addRegion();
  if (parser->parseOptionalRegion(*body,
          L::ArrayRef<M::OpAsmParser::OperandType>{}, L::ArrayRef<M::Type>{}))
    return M::failure();

  ensureTerminator(*body, builder, result->location);
  return M::success();
}

void GlobalOp::print(M::OpAsmPrinter *p) {
  auto varName =
      getAttrOfType<StringAttr>(M::SymbolTable::getSymbolAttrName()).getValue();
  *p << getOperationName() << " @" << varName << " : ";
  p->printType(getType());
  Region &body = getOperation()->getRegion(0);
  if (!body.empty())
    p->printRegion(body, /*printEntryBlockArgs=*/false,
        /*printBlockTerminators=*/false);
}

M::LogicalResult GlobalOp::verify() { return M::success(); }

void GlobalOp::appendInitialValue(M::Operation *op) {
  front().front().push_back(op);
}

M::Region &GlobalOp::front() { return this->getOperation()->getRegion(0); }

// LoopOp

void LoopOp::build(M::Builder *builder, M::OperationState *result, M::Value *lb,
    M::Value *ub, L::ArrayRef<M::Value *> step) {
  if (step.empty())
    result->addOperands({lb, ub});
  else
    result->addOperands({lb, ub, step[0]});
  M::Region *bodyRegion = result->addRegion();
  LoopOp::ensureTerminator(*bodyRegion, *builder, result->location);
  bodyRegion->front().addArgument(builder->getIndexType());
}

M::ParseResult parseLoopOp(M::OpAsmParser *parser, M::OperationState *result) {
  auto &builder = parser->getBuilder();
  M::OpAsmParser::OperandType inductionVariable, lb, ub, step;
  // Parse the induction variable followed by '='.
  if (parser->parseRegionArgument(inductionVariable) || parser->parseEqual())
    return M::failure();

  // Parse loop bounds.
  M::Type indexType = builder.getIndexType();
  if (parser->parseOperand(lb) ||
      parser->resolveOperand(lb, indexType, result->operands) ||
      parser->parseKeyword("to") || parser->parseOperand(ub) ||
      parser->resolveOperand(ub, indexType, result->operands))
    return M::failure();
  if (parser->parseOptionalKeyword(LoopOp::getStepKeyword())) {
    result->addAttribute(LoopOp::getStepKeyword(),
        builder.getIntegerAttr(builder.getIndexType(), 1));
  } else if (parser->parseOperand(step) ||
      parser->resolveOperand(step, indexType, result->operands)) {
    return M::failure();
  }

  if (parser->parseOptionalKeyword("unordered")) {
    // ok
  } else {
    result->addAttribute("unordered", builder.getBoolAttr(true));
  }

  // Parse the body region.
  M::Region *body = result->addRegion();
  if (parser->parseRegion(*body, inductionVariable, indexType))
    return M::failure();

  fir::LoopOp::ensureTerminator(*body, builder, result->location);

  // Parse the optional attribute list.
  if (parser->parseOptionalAttributeDict(result->attributes)) {
    return M::failure();
  }
  return M::success();
}

fir::LoopOp getForInductionVarOwner(M::Value *val) {
  auto *ivArg = dyn_cast<M::BlockArgument>(val);
  if (!ivArg) {
    return fir::LoopOp();
  }
  assert(ivArg->getOwner() && "unlinked block argument");
  auto *containingInst = ivArg->getOwner()->getParentOp();
  return dyn_cast_or_null<fir::LoopOp>(containingInst);
}

// StoreExpr

M::Type StoreExpr::elementType(M::Type refType) {
  if (auto ref = refType.dyn_cast<ReferenceType>()) return ref.getEleTy();
  if (auto ref = refType.dyn_cast<PointerType>()) return ref.getEleTy();
  if (auto ref = refType.dyn_cast<HeapType>()) return ref.getEleTy();
  return {};
}

// WhereOp

void WhereOp::build(M::Builder *builder, M::OperationState *result,
    M::Value *cond, bool withElseRegion) {
  result->addOperands(cond);
  M::Region *thenRegion = result->addRegion();
  M::Region *elseRegion = result->addRegion();
  WhereOp::ensureTerminator(*thenRegion, *builder, result->location);
  if (withElseRegion)
    WhereOp::ensureTerminator(*elseRegion, *builder, result->location);
}

M::ParseResult parseWhereOp(M::OpAsmParser *parser, M::OperationState *result) {
  // Create the regions for 'then'.
  result->regions.reserve(2);
  M::Region *thenRegion = result->addRegion();
  M::Region *elseRegion = result->addRegion();

  auto &builder = parser->getBuilder();
  M::OpAsmParser::OperandType cond;
  M::Type i1Type = builder.getIntegerType(1);
  if (parser->parseOperand(cond) ||
      parser->resolveOperand(cond, i1Type, result->operands))
    return M::failure();

  if (parser->parseRegion(*thenRegion, {}, {})) {
    return M::failure();
  }
  WhereOp::ensureTerminator(
      *thenRegion, parser->getBuilder(), result->location);

  if (!parser->parseOptionalKeyword("otherwise")) {
    if (parser->parseRegion(*elseRegion, {}, {})) {
      return M::failure();
    }
    WhereOp::ensureTerminator(
        *elseRegion, parser->getBuilder(), result->location);
  }

  // Parse the optional attribute list.
  if (parser->parseOptionalAttributeDict(result->attributes)) {
    return M::failure();
  }

  return M::success();
}

M::ParseResult isValidCaseAttr(M::Attribute attr) {
  if (attr.dyn_cast_or_null<M::UnitAttr>() ||
      attr.dyn_cast_or_null<ClosedIntervalAttr>() ||
      attr.dyn_cast_or_null<PointIntervalAttr>() ||
      attr.dyn_cast_or_null<LowerBoundAttr>() ||
      attr.dyn_cast_or_null<UpperBoundAttr>())
    return M::success();
  return M::failure();
}

unsigned getCaseArgumentOffset(L::ArrayRef<M::Attribute> cases, unsigned dest) {
  unsigned o = 0;
  for (unsigned i = 0; i < dest; ++i) {
    auto &attr = cases[i];
    if (!attr.dyn_cast_or_null<M::UnitAttr>()) {
      ++o;
      if (attr.dyn_cast_or_null<ClosedIntervalAttr>()) {
        ++o;
      }
    }
  }
  return o;
}

// Tablegen operators

#define GET_OP_CLASSES
#include "fir/FIROps.cpp.inc"

}  // fir
