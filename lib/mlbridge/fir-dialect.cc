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

#include "fir-dialect.h"
#include "fir-type.h"
#include "../evaluate/expression.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/LLVMIR/LLVMDialect.h"

namespace Br = Fortran::mlbridge;
namespace Co = Fortran::common;
namespace M = mlir;

using namespace Fortran;
using namespace Fortran::mlbridge;

namespace {
std::string someExprToString(const SomeExpr *expr) {
  std::stringstream ss;
  expr->AsFortran(ss);
  return ss.str();
}

template<typename A>
void selectBuild(M::OpBuilder *builder, M::OperationState *result,
    M::Value *condition, llvm::ArrayRef<typename A::BranchTuple> tuples) {
  result->addOperands(condition);
  for (auto &tup : tuples) {
    auto *cond{std::get<typename A::Conditions>(tup)};
    result->addOperands(cond);
  }
  // Note: succs must be added *after* operands
  for (auto &tup : tuples) {
    auto *block{std::get<M::Block *>(tup)};
    assert(block);
    auto blkArgs{std::get<llvm::ArrayRef<M::Value *>>(tup)};
    result->addSuccessor(block, blkArgs);
  }
}
}  // namespace

Br::FIROpsDialect::FIROpsDialect(M::MLIRContext *ctx) : M::Dialect("fir", ctx) {
  addTypes<FIRBoxType, FIRBoxCharType, FIRBoxProcType, FIRCharacterType,
      FIRDimsType, FIRFieldType, FIRLogicalType, FIRRealType, FIRReferenceType,
      FIRSequenceType, FIRTupleType, FIRTypeDescType>();
  addOperations<AllocaExpr, AllocMemOp, ApplyExpr, BoxAddrOp, BoxCharLenOp,
      BoxDimsOp, BoxEleSizeOp, BoxIsAllocOp, BoxIsPtrOp, BoxProcHostOp,
      BoxRankOp, BoxTDescOp, ConvertOp, CoordinateOp, EmboxOp, EmboxCharOp,
      EmboxProcOp, ExtractValueOp, FieldValueOp, FreeMemOp, GenDimsOp,
      GlobalExpr, InsertValueOp, LoadExpr, LocateExpr, SelectOp, SelectCaseOp,
      SelectRankOp, SelectTypeOp, StoreExpr, UnboxOp, UnboxCharOp, UnboxProcOp,
      UndefOp, UnreachableOp>();
}

// anchor the class vtable
Br::FIROpsDialect::~FIROpsDialect() {}

M::Type Br::FIROpsDialect::parseType(
    llvm::StringRef rawData, M::Location loc) const {
  return parseFirType(const_cast<FIROpsDialect *>(this), rawData, loc);
}

void printBounds(llvm::raw_ostream &os, const FIRSequenceType::Bounds &bounds) {
  char ch = ' ';
  for (auto &b : bounds) {
    std::visit(Co::visitors{
                   [&](const FIRSequenceType::Unknown &) { os << ch << '?'; },
                   [&](const FIRSequenceType::BoundInfo &info) {
                     os << ch << info.lower << ' ' << info.count << ' '
                        << info.stride;
                   },
               },
        b);
    ch = ',';
  }
}

void Br::FIROpsDialect::printType(M::Type ty, llvm::raw_ostream &os) const {
  if (auto type = ty.dyn_cast<FIRReferenceType>()) {
    os << "ref<";
    type.getEleTy().print(os);
    os << '>';
  } else if (auto type = ty.dyn_cast<FIRLogicalType>()) {
    os << "logical<" << type.getFKind() << '>';
  } else if (auto type = ty.dyn_cast<FIRRealType>()) {
    os << "real<" << type.getFKind() << '>';
  } else if (auto type = ty.dyn_cast<FIRCharacterType>()) {
    os << "char<" << type.getFKind() << '>';
  } else if (auto type = ty.dyn_cast<FIRTypeDescType>()) {
    os << "tdesc<";
    type.getOfTy().print(os);
    os << '>';
  } else if (auto type = ty.dyn_cast<FIRFieldType>()) {
    os << "field";
  } else if (auto type = ty.dyn_cast<FIRBoxType>()) {
    os << "box<";
    type.getEleTy().print(os);
    os << '>';
  } else if (auto type = ty.dyn_cast<FIRBoxCharType>()) {
    os << "boxchar<";
    type.getEleTy().print(os);
    os << '>';
  } else if (auto type = ty.dyn_cast<FIRBoxProcType>()) {
    os << "boxproc<";
    type.getEleTy().print(os);
    os << '>';
  } else if (auto type = ty.dyn_cast<FIRDimsType>()) {
    os << "dims<" << type.getRank() << '>';
  } else if (auto type = ty.dyn_cast<FIRSequenceType>()) {
    os << "array<";
    std::visit(Co::visitors{
                   [&](const FIRSequenceType::Unknown &) { os << '*'; },
                   [&](const FIRSequenceType::Bounds &bounds) {
                     printBounds(os, bounds);
                   },
               },
        type.getShape());
    os << ':';
    type.getEleTy().print(os);
    os << '>';
  } else if (auto type = ty.dyn_cast<FIRTupleType>()) {
    os << "type<\"" << type.getName() << '"';
    if (type.getTypeList().size()) {
      os << ",{";
      char ch = ' ';
      for (auto p : type.getTypeList()) {
        os << ch << '"' << p.first << "\": ";
        p.second.print(os);
        ch = ',';
      }
      os << '}';
    }
    if (type.getKindList().size()) {
      os << ",[";
      char ch = ' ';
      for (auto p : type.getKindList()) {
        os << ch << '"' << p.first << "\": " << p.second;
        ch = ',';
      }
      os << ']';
    }
    os << '>';
  } else {
    assert(false);
  }
}

M::Attribute Br::FIROpsDialect::parseAttribute(
    llvm::StringRef attrData, M::Location loc) const {
  return M::SerializableAttr::get(nullptr, getContext());
}

void Br::FIROpsDialect::printAttribute(
    M::Attribute attr, llvm::raw_ostream &os) const {
  os << '?';
}

/// AllocaExpr
void Br::AllocaExpr::build(
    M::OpBuilder *builder, M::OperationState *result, M::Type opTy) {
  result->addTypes(FIRReferenceType::get(opTy));
}

void Br::AllocaExpr::build(M::OpBuilder *builder, M::OperationState *result,
    llvm::StringRef name, M::Type opTy) {
  result->addAttribute("name", builder->getStringAttr(name));
  result->addTypes(FIRReferenceType::get(opTy));
}

M::Type Br::AllocaExpr::getAllocatedType() {
  auto ty{*getOperation()->result_type_begin()};
  return ty.cast<FIRReferenceType>().getEleTy();
}

M::LogicalResult Br::AllocaExpr::verify() { return M::success(); }

/// AllocMem
M::LogicalResult Br::AllocMemOp::verify() { return M::success(); }

/// ApplyExpr
void Br::ApplyExpr::build(M::OpBuilder *builder, M::OperationState *result,
    const SomeExpr *expr, const std::map<unsigned, void *> &dict,
    llvm::ArrayRef<M::Value *> operands, M::Type opTy) {
  result->addOperands(operands);
  result->addAttribute(
      "expr", builder->getSerializableAttr(const_cast<SomeExpr *>(expr)));
  void *dictCopy = new std::map<unsigned, void *>{dict};  // FIXME
  result->addAttribute("dict", builder->getSerializableAttr(dictCopy));
  result->addTypes(opTy);
}

M::LogicalResult Br::ApplyExpr::verify() { return M::success(); }

SomeExpr *Br::ApplyExpr::getRawExpr() {
  void *barePtr{getAttrOfType<M::SerializableAttr>("expr").getValue()};
  return reinterpret_cast<SomeExpr *>(barePtr);
}

/// string representation of the expression
llvm::StringRef Br::ApplyExpr::getExpr() {
  return someExprToString(getRawExpr());
}

std::map<unsigned, void *> *Br::ApplyExpr::getDict() {
  void *barePtr{getAttrOfType<M::SerializableAttr>("dict").getValue()};
  return reinterpret_cast<std::map<unsigned, void *> *>(barePtr);
}

// FIXME
inline llvm::ArrayRef<M::Value *> getODSOperands(int) { return {}; }

/// BoxAddr
void Br::BoxAddrOp::build(
    M::OpBuilder *builder, M::OperationState *result, M::Value *operand) {
  result->addOperands(operand);
}
M::LogicalResult Br::BoxAddrOp::verify() {
  for (M::Value *v : getODSOperands(0)) {
    if (!(v->getType().isa<FIRBoxType>() ||
            v->getType().isa<FIRBoxCharType>() ||
            v->getType().isa<FIRBoxProcType>())) {
      return emitOpError("operand #0 must be a FIR box type");
    }
  }
  return M::success();
}

/// BoxCharLen
void Br::BoxCharLenOp::build(
    M::OpBuilder *builder, M::OperationState *result, M::Value *operand) {
  result->addOperands(operand);
  result->addTypes(M::IntegerType::get(64, builder->getContext()));
}
M::LogicalResult Br::BoxCharLenOp::verify() {
  for (M::Value *v : getODSOperands(0)) {
    if (!(v->getType().isa<FIRBoxCharType>())) {
      return emitOpError("operand #0 must be a FIR boxchar type");
    }
  }
  return M::success();
}

/// BoxDims
void Br::BoxDimsOp::build(M::OpBuilder *builder, M::OperationState *result,
    llvm::ArrayRef<M::Value *> operand) {
  result->addOperands(operand);
}
M::LogicalResult Br::BoxDimsOp::verify() {
  for (M::Value *v : getODSOperands(0)) {
    if (!(v->getType().isa<FIRBoxType>())) {
      return emitOpError("operand #0 must be a FIR box type");
    }
  }
  return M::success();
}

/// BoxEleSize
void Br::BoxEleSizeOp::build(
    M::OpBuilder *builder, M::OperationState *result, M::Value *operand) {
  result->addOperands(operand);
  result->addTypes(M::IntegerType::get(64, builder->getContext()));
}
M::LogicalResult Br::BoxEleSizeOp::verify() {
  for (M::Value *v : getODSOperands(0)) {
    if (!(v->getType().isa<FIRBoxType>())) {
      return emitOpError("operand #0 must be a FIR box type");
    }
  }
  return M::success();
}

/// BoxIsAlloc
void Br::BoxIsAllocOp::build(
    M::OpBuilder *builder, M::OperationState *result, M::Value *operand) {
  result->addOperands(operand);
  result->addTypes(M::IntegerType::get(1, builder->getContext()));
}
M::LogicalResult Br::BoxIsAllocOp::verify() {
  for (M::Value *v : getODSOperands(0)) {
    if (!(v->getType().isa<FIRBoxType>())) {
      return emitOpError("operand #0 must be a FIR box type");
    }
  }
  return M::success();
}

/// BoxIsPtr
void Br::BoxIsPtrOp::build(
    M::OpBuilder *builder, M::OperationState *result, M::Value *operand) {
  result->addOperands(operand);
  result->addTypes(M::IntegerType::get(1, builder->getContext()));
}
M::LogicalResult Br::BoxIsPtrOp::verify() {
  for (M::Value *v : getODSOperands(0)) {
    if (!(v->getType().isa<FIRBoxType>())) {
      return emitOpError("operand #0 must be a FIR box type");
    }
  }
  return M::success();
}

/// BoxProcHost
void Br::BoxProcHostOp::build(
    M::OpBuilder *builder, M::OperationState *result, M::Value *operand) {
  result->addOperands(operand);
}
M::LogicalResult Br::BoxProcHostOp::verify() {
  for (M::Value *v : getODSOperands(0)) {
    if (!(v->getType().isa<FIRBoxProcType>())) {
      return emitOpError("operand #0 must be a FIR boxproc type");
    }
  }
  return M::success();
}

/// BoxRank
void Br::BoxRankOp::build(
    M::OpBuilder *builder, M::OperationState *result, M::Value *operand) {
  result->addOperands(operand);
  result->addTypes(M::IntegerType::get(16, builder->getContext()));
}
M::LogicalResult Br::BoxRankOp::verify() {
  for (M::Value *v : getODSOperands(0)) {
    if (!(v->getType().isa<FIRBoxType>())) {
      return emitOpError("operand #0 must be a FIR box type");
    }
  }
  return M::success();
}

/// BoxTDesc
void Br::BoxTDescOp::build(
    M::OpBuilder *builder, M::OperationState *result, M::Value *operand) {
  result->addOperands(operand);
  FIRBoxType boxTy{operand->getType().cast<FIRBoxType>()};
  result->addTypes(FIRTypeDescType::get(boxTy.getEleTy()));
}
M::LogicalResult Br::BoxTDescOp::verify() {
  for (M::Value *v : getODSOperands(0)) {
    if (!(v->getType().isa<FIRBoxType>())) {
      return emitOpError("operand #0 must be a FIR box type");
    }
  }
  return M::success();
}

/// ConvertOp
M::LogicalResult Br::ConvertOp::verify() { return M::success(); }

void Br::ConvertOp::build(mlir::OpBuilder *builder,
    mlir::OperationState *result, mlir::Value *val, mlir::Type toType) {
  result->addOperands({val});
  result->addTypes(toType);
}

/// CoordinateOp
M::LogicalResult Br::CoordinateOp::verify() { return M::success(); }

void Br::CoordinateOp::build(mlir::OpBuilder *builder,
    mlir::OperationState *result, llvm::ArrayRef<mlir::Value *> operands,
    M::Type opTy) {
  result->addOperands(operands);
  result->addTypes(FIRReferenceType::get(opTy));
}

/// Embox
void Br::EmboxOp::build(M::OpBuilder *builder, M::OperationState *result,
    llvm::ArrayRef<M::Value *> operands) {
  result->addOperands(operands);
}
M::LogicalResult Br::EmboxOp::verify() { return M::success(); }

/// EmboxChar
void Br::EmboxCharOp::build(M::OpBuilder *builder, M::OperationState *result,
    llvm::ArrayRef<M::Value *> operands) {
  result->addOperands(operands);
}
M::LogicalResult Br::EmboxCharOp::verify() { return M::success(); }

/// EmboxProc
void Br::EmboxProcOp::build(M::OpBuilder *builder, M::OperationState *result,
    llvm::ArrayRef<M::Value *> operands) {
  result->addOperands(operands);
}
M::LogicalResult Br::EmboxProcOp::verify() { return M::success(); }

/// ExtractValue
M::LogicalResult Br::ExtractValueOp::verify() { return M::success(); }

void Br::ExtractValueOp::build(mlir::OpBuilder *builder,
    mlir::OperationState *result, llvm::ArrayRef<mlir::Value *> operands,
    mlir::Type opTy) {
  result->addOperands(operands);
  result->addTypes(opTy);
}

/// FieldValue
M::LogicalResult Br::FieldValueOp::verify() { return M::success(); }

void Br::FieldValueOp::build(mlir::OpBuilder *builder,
    mlir::OperationState *result, llvm::StringRef name) {
  result->addAttribute("part", builder->getStringAttr(name));
  result->addTypes(FIRFieldType::get(builder->getContext()));
}

/// FreeMem
M::LogicalResult Br::FreeMemOp::verify() { return M::success(); }

/// GenDims
M::LogicalResult Br::GenDimsOp::verify() { return M::success(); }

void Br::GenDimsOp::build(M::OpBuilder *builder, M::OperationState *result,
    llvm::ArrayRef<M::Value *> args) {
  assert(args.size() % 3 == 0);
  unsigned rank = args.size() / 3;
  result->addOperands(args);
  result->addTypes(FIRDimsType::get(builder->getContext(), rank));
}

llvm::ArrayRef<M::Value *> Br::GenDimsOp::getDim(unsigned dim) {
  M::Operation *op = getOperation();
  llvm::SmallVector<M::Value *, 3> result;
  assert(dim < op->getNumOperands() / 3);
  for (unsigned i = dim * 3, end = dim * 3 + 2; i < end; ++i) {
    result.push_back(op->getOperand(i));
  }
  return result;
}

/// GlobalExpr
M::LogicalResult Br::GlobalExpr::verify() { return M::success(); }

void Br::GlobalExpr::build(M::OpBuilder *builder, M::OperationState *result,
    llvm::StringRef name, M::Attribute value, M::Type opTy) {
  result->addAttribute("name", builder->getStringAttr(name));
  result->addAttribute("value", value);
  result->addTypes(FIRReferenceType::get(opTy));
}

/// InsertValue
M::LogicalResult Br::InsertValueOp::verify() { return M::success(); }

void Br::InsertValueOp::build(mlir::OpBuilder *builder,
    mlir::OperationState *result, llvm::ArrayRef<mlir::Value *> operands,
    mlir::Type opTy) {
  result->addOperands(operands);
  result->addTypes(opTy);
}

/// LoadExpr
M::LogicalResult Br::LoadExpr::verify() { return M::success(); }

void Br::LoadExpr::build(M::OpBuilder *builder, M::OperationState *result,
    llvm::ArrayRef<M::Value *> operands, M::Type opTy) {
  result->addOperands(operands);
  result->addTypes(opTy);
}

void Br::LoadExpr::build(
    M::OpBuilder *builder, M::OperationState *result, M::Value *store) {
  auto opTy{store->getType().cast<FIRReferenceType>().getEleTy()};
  llvm::SmallVector<M::Value *, 1> loadArg{store};
  result->addOperands(std::move(loadArg));
  result->addTypes(opTy);
}

/// LocateExpr
M::LogicalResult Br::LocateExpr::verify() { return M::success(); }

void Br::LocateExpr::build(M::OpBuilder *builder, M::OperationState *result,
    const SomeExpr *expr, const std::map<unsigned, void *> &dict,
    llvm::ArrayRef<M::Value *> operands, M::Type opTy) {
  result->addOperands(operands);
  result->addAttribute(
      "addr", builder->getSerializableAttr(const_cast<SomeExpr *>(expr)));
  void *dictCopy = new std::map<unsigned, void *>{dict};  // FIXME
  result->addAttribute("dict", builder->getSerializableAttr(dictCopy));
  result->addTypes(opTy);
}

SomeExpr *Br::LocateExpr::getRawExpr() {
  void *barePtr{getAttrOfType<M::SerializableAttr>("addr").getValue()};
  return reinterpret_cast<SomeExpr *>(barePtr);
}

/// string representation of the expression
llvm::StringRef Br::LocateExpr::getExpr() {
  return someExprToString(getRawExpr());
}

std::map<unsigned, void *> *Br::LocateExpr::getDict() {
  void *barePtr{getAttrOfType<M::SerializableAttr>("dict").getValue()};
  return reinterpret_cast<std::map<unsigned, void *> *>(barePtr);
}

/// SelectCase
M::LogicalResult Br::SelectCaseOp::verify() { return M::success(); }

void Br::SelectCaseOp::build(M::OpBuilder *builder, M::OperationState *result,
    M::Value *condition, llvm::ArrayRef<BranchTuple> tuples) {
  selectBuild<SelectCaseOp>(builder, result, condition, tuples);
}

/// SelectRank
M::LogicalResult Br::SelectRankOp::verify() { return M::success(); }

void Br::SelectRankOp::build(M::OpBuilder *builder, M::OperationState *result,
    M::Value *condition, llvm::ArrayRef<BranchTuple> tuples) {
  selectBuild<SelectRankOp>(builder, result, condition, tuples);
}

/// SelectType
M::LogicalResult Br::SelectTypeOp::verify() { return M::success(); }

void Br::SelectTypeOp::build(M::OpBuilder *builder, M::OperationState *result,
    M::Value *condition, llvm::ArrayRef<BranchTuple> tuples) {
  selectBuild<SelectTypeOp>(builder, result, condition, tuples);
}

/// Select
M::LogicalResult Br::SelectOp::verify() { return M::success(); }

/// StoreExpr
void Br::StoreExpr::build(M::OpBuilder *builder, M::OperationState *result,
    llvm::ArrayRef<M::Value *> operands) {
  result->addOperands(operands);
}

void Br::StoreExpr::build(M::OpBuilder *builder, M::OperationState *result,
    M::Value *value, M::Value *store) {
  llvm::SmallVector<M::Value *, 2> storeArgs{value, store};
  result->addOperands(std::move(storeArgs));
}

M::LogicalResult Br::StoreExpr::verify() { return M::success(); }

/// Unbox
void Br::UnboxOp::build(
    M::OpBuilder *builder, M::OperationState *result, M::Value *operand) {
  result->addOperands(operand);
}
M::LogicalResult Br::UnboxOp::verify() { return M::success(); }

/// UnboxChar
void Br::UnboxCharOp::build(
    M::OpBuilder *builder, M::OperationState *result, M::Value *operand) {
  result->addOperands(operand);
}
M::LogicalResult Br::UnboxCharOp::verify() { return M::success(); }

/// UnboxProc
void Br::UnboxProcOp::build(
    M::OpBuilder *builder, M::OperationState *result, M::Value *operand) {
  result->addOperands(operand);
}
M::LogicalResult Br::UnboxProcOp::verify() { return M::success(); }

/// Undef
M::LogicalResult Br::UndefOp::verify() { return M::success(); }

void Br::UndefOp::build(
    M::OpBuilder *builder, M::OperationState *result, M::Type opTy) {
  result->addTypes(opTy);
}

/// Unreachable
/// an unreachable takes no operands or attributes and has no type
void Br::UnreachableOp::build(
    M::OpBuilder *builder, M::OperationState *result) {
  // do nothing
}

M::LogicalResult Br::UnreachableOp::verify() { return M::success(); }
