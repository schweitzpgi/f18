//===-- include/fir/Attribute.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//----------------------------------------------------------------------------//

#ifndef DIALECT_FIR_FIRATTRIBUTE_H
#define DIALECT_FIR_FIRATTRIBUTE_H

#include "mlir/IR/Attributes.h"

namespace mlir {
class DialectAsmParser;
class DialectAsmPrinter;
} // namespace mlir

namespace fir {

class FIROpsDialect;

namespace detail {
struct TypeAttributeStorage;
}

enum AttributeKind {
  FIR_ATTR = mlir::Attribute::FIRST_FIR_ATTR,
  FIR_EXACTTYPE, // instance_of, precise type relation
  FIR_SUBCLASS,  // subsumed_by, is-a (subclass) relation
  FIR_POINT,
  FIR_CLOSEDCLOSED_INTERVAL,
  FIR_OPENCLOSED_INTERVAL,
  FIR_CLOSEDOPEN_INTERVAL,
  FIR_REAL_ATTR,
};

class ExactTypeAttr
    : public mlir::Attribute::AttrBase<ExactTypeAttr, mlir::Attribute,
                                       detail::TypeAttributeStorage> {
public:
  using Base::Base;
  using ValueType = mlir::Type;

  static llvm::StringRef getAttrName() { return "instance"; }
  static ExactTypeAttr get(mlir::Type value);

  mlir::Type getType() const;

  constexpr static bool kindof(unsigned kind) { return kind == getId(); }
  constexpr static unsigned getId() { return AttributeKind::FIR_EXACTTYPE; }
};

class SubclassAttr
    : public mlir::Attribute::AttrBase<SubclassAttr, mlir::Attribute,
                                       detail::TypeAttributeStorage> {
public:
  using Base::Base;
  using ValueType = mlir::Type;

  static llvm::StringRef getAttrName() { return "subsumed"; }
  static SubclassAttr get(mlir::Type value);

  mlir::Type getType() const;

  constexpr static bool kindof(unsigned kind) { return kind == getId(); }
  constexpr static unsigned getId() { return AttributeKind::FIR_SUBCLASS; }
};

class ClosedIntervalAttr
    : public mlir::Attribute::AttrBase<ClosedIntervalAttr> {
public:
  using Base::Base;

  static llvm::StringRef getAttrName() { return "interval"; }
  static ClosedIntervalAttr get(mlir::MLIRContext *ctxt);
  constexpr static bool kindof(unsigned kind) { return kind == getId(); }
  constexpr static unsigned getId() {
    return AttributeKind::FIR_CLOSEDCLOSED_INTERVAL;
  }
};

class UpperBoundAttr : public mlir::Attribute::AttrBase<UpperBoundAttr> {
public:
  using Base::Base;

  static llvm::StringRef getAttrName() { return "upper"; }
  static UpperBoundAttr get(mlir::MLIRContext *ctxt);
  constexpr static bool kindof(unsigned kind) { return kind == getId(); }
  constexpr static unsigned getId() {
    return AttributeKind::FIR_OPENCLOSED_INTERVAL;
  }
};

class LowerBoundAttr : public mlir::Attribute::AttrBase<LowerBoundAttr> {
public:
  using Base::Base;

  static llvm::StringRef getAttrName() { return "lower"; }
  static LowerBoundAttr get(mlir::MLIRContext *ctxt);
  constexpr static bool kindof(unsigned kind) { return kind == getId(); }
  constexpr static unsigned getId() {
    return AttributeKind::FIR_CLOSEDOPEN_INTERVAL;
  }
};

class PointIntervalAttr : public mlir::Attribute::AttrBase<PointIntervalAttr> {
public:
  using Base::Base;

  static llvm::StringRef getAttrName() { return "point"; }
  static PointIntervalAttr get(mlir::MLIRContext *ctxt);
  constexpr static bool kindof(unsigned kind) { return kind == getId(); }
  constexpr static unsigned getId() { return AttributeKind::FIR_POINT; }
};

class RealAttr : public mlir::Attribute::AttrBase<RealAttr, mlir::Attribute> {
public:
  using Base::Base;

  static llvm::StringRef getAttrName() { return "real"; }
  static RealAttr get(mlir::MLIRContext *ctxt, const llvm::APFloat &flt);
  constexpr static bool kindof(unsigned kind) { return kind == getId(); }
  constexpr static unsigned getId() { return AttributeKind::FIR_REAL_ATTR; }
};

mlir::Attribute parseFirAttribute(FIROpsDialect *dialect,
                                  mlir::DialectAsmParser &parser,
                                  mlir::Type type);

void printFirAttribute(FIROpsDialect *dialect, mlir::Attribute attr,
                       mlir::DialectAsmPrinter &p);

} // namespace fir

#endif // DIALECT_FIR_FIRATTRIBUTE_H
