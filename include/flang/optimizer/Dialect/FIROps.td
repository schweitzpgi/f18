//===-- FIROps.td - FIR operation definitions --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Definition of the FIR dialect operations
///
//===----------------------------------------------------------------------===//

#ifndef FIR_DIALECT_FIR_OPS
#define FIR_DIALECT_FIR_OPS

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif

def fir_Dialect : Dialect {
  let name = "fir";
}

def fir_Type : Type<CPred<"fir::isa_fir_or_std_type($_self)">,
    "FIR dialect type">;

def fir_CharacterType : Type<CPred<"$_self.isa<fir::CharacterType>()">,
    "FIR character type">;
def fir_ComplexType : Type<CPred<"$_self.isa<fir::CplxType>()">,
    "FIR complex type">;
def fir_IntegerType : Type<CPred<"$_self.isa<fir::IntType>()">,
    "FIR integer type">;
def fir_LogicalType : Type<CPred<"$_self.isa<fir::LogicalType>()">,
    "FIR logical type">;
def fir_RealType : Type<CPred<"$_self.isa<fir::RealType>()">,
    "FIR real type">;

def AnyIntegerLike : TypeConstraint<Or<[IntegerLike.predicate,
    fir_IntegerType.predicate]>, "any integer">;
def AnyLogicalLike : TypeConstraint<Or<[BoolLike.predicate,
    fir_LogicalType.predicate]>, "any logical">;
def AnyRealLike : TypeConstraint<Or<[FloatLike.predicate,
    fir_RealType.predicate]>, "any real">;
def AnyIntegerType : Type<AnyIntegerLike.predicate, "any integer">;

def fir_RecordType : Type<CPred<"$_self.isa<fir::RecordType>()">,
    "FIR derived type">;
def fir_SequenceType : Type<CPred<"$_self.isa<fir::SequenceType>()">,
    "array type">;

def AnyCompositeLike : TypeConstraint<Or<[fir_RecordType.predicate,
    fir_SequenceType.predicate, fir_ComplexType.predicate,
    IsTupleTypePred]>, "any composite">;

def fir_ReferenceType : Type<CPred<"$_self.isa<fir::ReferenceType>()">,
    "reference type">;
def fir_HeapType : Type<CPred<"$_self.isa<fir::HeapType>()">,
    "allocatable type">;
def fir_PointerType : Type<CPred<"$_self.isa<fir::PointerType>()">,
    "pointer type">;

def AnyReferenceLike : TypeConstraint<Or<[fir_ReferenceType.predicate,
    fir_HeapType.predicate, fir_PointerType.predicate]>, "any reference">;

def fir_BoxType : Type<CPred<"$_self.isa<fir::BoxType>()">, "box type">;
def fir_BoxCharType : Type<CPred<"$_self.isa<fir::BoxCharType>()">,
    "box character type">;
def fir_BoxProcType : Type<CPred<"$_self.isa<fir::BoxProcType>()">,
    "box procedure type">;

def AnyBoxLike : TypeConstraint<Or<[fir_BoxType.predicate,
    fir_BoxCharType.predicate, fir_BoxProcType.predicate]>, "any box">;

def AnyRefOrBox : TypeConstraint<Or<[fir_ReferenceType.predicate,
    fir_HeapType.predicate, fir_PointerType.predicate, fir_BoxType.predicate]>,
    "any reference or box">;

def fir_DimsType : Type<CPred<"$_self.isa<fir::DimsType>()">, "dim type">;
def AnyEmboxLike : TypeConstraint<Or<[AnyInteger.predicate,
    Index.predicate, fir_IntegerType.predicate, fir_DimsType.predicate]>,
    "any legal embox argument type">;
def AnyEmboxArg : Type<AnyEmboxLike.predicate, "embox argument type">;

def fir_TypeDescType : Type<CPred<"$_self.isa<fir::TypeDescType>()">,
    "type desc type">;
def fir_FieldType : Type<CPred<"$_self.isa<fir::FieldType>()">, "field type">;
def fir_LenType : Type<CPred<"$_self.isa<fir::LenType>()">,
    "LEN parameter type">;

def AnyComponentLike : TypeConstraint<Or<[AnyInteger.predicate,
    Index.predicate, fir_IntegerType.predicate, fir_FieldType.predicate]>,
    "any coordinate index">;
def AnyComponentType : Type<AnyComponentLike.predicate, "coordinate type">;

def AnyCoordinateLike : TypeConstraint<Or<[AnyInteger.predicate,
    Index.predicate, fir_IntegerType.predicate, fir_FieldType.predicate,
    fir_LenType.predicate]>, "any coordinate index">;
def AnyCoordinateType : Type<AnyCoordinateLike.predicate, "coordinate type">;

// Base class for FIR operations.
// All operations automatically get a prefix of "fir.".
class fir_Op<string mnemonic, list<OpTrait> traits>
  : Op<fir_Dialect, mnemonic, traits>;

// Base class for FIR operations that take a single argument
class fir_SimpleOp<string mnemonic, list<OpTrait> traits>
  : fir_Op<mnemonic, traits> {

  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
}

// Base builder for allocate operations
def fir_AllocateOpBuilder : OpBuilder<
  "Builder *builder, OperationState &result, Type inType,"
  "ArrayRef<Value> lenParams = {}, ArrayRef<Value> sizes = {},"
  "ArrayRef<NamedAttribute> attributes = {}",
  [{
    result.addTypes(getRefTy(inType));
    result.addAttribute("in_type", mlir::TypeAttr::get(inType));
    result.addOperands(sizes);
    for (auto namedAttr : attributes)
      result.addAttribute(namedAttr.first, namedAttr.second);
  }]>;

def fir_NamedAllocateOpBuilder : OpBuilder<
  "Builder *builder, OperationState &result, Type inType, StringRef name,"
  "ArrayRef<Value> lenParams = {}, ArrayRef<Value> sizes = {},"
  "ArrayRef<NamedAttribute> attributes = {}",
  [{
    result.addTypes(getRefTy(inType));
    result.addAttribute("in_type", mlir::TypeAttr::get(inType));
    result.addAttribute("name", builder->getStringAttr(name));
    result.addOperands(sizes);
    for (auto namedAttr : attributes)
      result.addAttribute(namedAttr.first, namedAttr.second);
  }]>;

def fir_OneResultOpBuilder : OpBuilder<
  "Builder *, OperationState &result, Type resultType,"
  "ArrayRef<Value> operands, ArrayRef<NamedAttribute> attributes = {}",
  [{
    if (resultType)
      result.addTypes(resultType);
    result.addOperands(operands);
    for (auto namedAttr : attributes)
      result.addAttribute(namedAttr.first, namedAttr.second);
  }]>;

// Base class of FIR operations that return 1 result
class fir_OneResultOp<string mnemonic, list<OpTrait> traits = []> :
    fir_Op<mnemonic, traits>, Results<(outs fir_Type:$res)> {
  let builders = [fir_OneResultOpBuilder];
}

// Base class of FIR operations that have 1 argument and return 1 result
class fir_SimpleOneResultOp<string mnemonic, list<OpTrait> traits = []> :
    fir_SimpleOp<mnemonic, traits> {
  let builders = [fir_OneResultOpBuilder];
}

class fir_TwoBuilders<OpBuilder b1, OpBuilder b2> {
  list<OpBuilder> builders = [b1, b2];
}

class fir_AllocatableBaseOp<string mnemonic, list<OpTrait> traits = []> :
    fir_Op<mnemonic, traits>, Results<(outs fir_Type:$res)> {
  let arguments = (ins
    OptionalAttr<StrAttr>:$name,
    OptionalAttr<BoolAttr>:$target
  );
}

class fir_AllocatableOp<string mnemonic, list<OpTrait> traits =[]> :
    fir_AllocatableBaseOp<mnemonic, !listconcat([NoSideEffect], traits)>,
    fir_TwoBuilders<fir_AllocateOpBuilder, fir_NamedAllocateOpBuilder>,
    Arguments<(ins TypeAttr:$in_type, Variadic<AnyIntegerType>:$args)> {

  let parser = [{
    mlir::Type intype;
    if (parser.parseType(intype))
      return mlir::failure();
    auto &builder = parser.getBuilder();
    result.addAttribute(inType(), mlir::TypeAttr::get(intype));
    llvm::SmallVector<mlir::OpAsmParser::OperandType, 8> operands;
    llvm::SmallVector<mlir::Type, 8> typeVec;
    bool hasOperands = false;
    if (!parser.parseOptionalLParen()) {
      // parse the LEN params of the derived type. (<params> : <types)
      if (parser.parseOperandList(operands,
                                  mlir::OpAsmParser::Delimiter::None) ||
          parser.parseColonTypeList(typeVec) ||
          parser.parseRParen())
        return mlir::failure();
      auto lens = builder.getI32IntegerAttr(operands.size());
      result.addAttribute(lenpName(), lens);
      hasOperands = true;
    }
    if (!parser.parseOptionalComma()) {
      // parse size to scale by, vector of n dimensions of type index
      auto opSize = operands.size();
      if (parser.parseOperandList(operands, mlir::OpAsmParser::Delimiter::None))
        return mlir::failure();
      for (auto i = opSize, end = operands.size(); i != end; ++i)
        typeVec.push_back(builder.getIndexType());
      hasOperands = true;
    }
    if (hasOperands &&
        parser.resolveOperands(operands, typeVec, parser.getNameLoc(),
                               result.operands))
      return mlir::failure();
    mlir::Type restype = wrapResultType(intype);
    if (!restype) {
      parser.emitError(parser.getNameLoc(), "invalid allocate type: ")
          << intype;
      return mlir::failure();
    }
    if (parser.parseOptionalAttrDict(result.attributes) ||
        parser.addTypeToList(restype, result.types))
      return mlir::failure();
    return mlir::success();
  }];

  let printer = [{
    p << getOperationName() << ' ' << getAttr(inType());
    if (hasLenParams()) {
      // print the LEN parameters to a derived type in parens
      p << '(';
      p.printOperands(getLenParams());
      p << " : ";
      mlir::interleaveComma(getLenParams(), p.getStream(),
                            [&](const auto &opnd) {
                                p.printType(opnd.getType());
                            });
      p << ')';
    }
    // print the shape of the allocation (if any); all must be index type
    for (auto sh : getShapeOperands()) {
      p << ", ";
      p.printOperand(sh);
    }
    p.printOptionalAttrDict(getAttrs(), {inType(), lenpName()});
  }];

  string extraAllocClassDeclaration = [{
    static constexpr llvm::StringRef inType() { return "in_type"; }
    static constexpr llvm::StringRef lenpName() { return "len_param_count"; }
    mlir::Type getAllocatedType();
    bool hasLenParams() { return bool{getAttr(lenpName())}; }
    unsigned numLenParams() {
      if (auto val = getAttrOfType<mlir::IntegerAttr>(lenpName()))
        return val.getInt();
      return 0;
    }
    operand_range getLenParams() {
      return {operand_begin(), operand_begin() + numLenParams()};
    }
    operand_range getShapeOperands() {
      return {operand_begin() + numLenParams(), operand_end()};
    }
    static mlir::Type getRefTy(mlir::Type ty);

    /// Get the input type of the allocation
    mlir::Type getInType() {
      return getAttrOfType<mlir::TypeAttr>(inType()).getValue();
    }
  }];

  // Verify checks common to all allocation operations
  string allocVerify = [{
    llvm::SmallVector<llvm::StringRef, 8> visited;
    if (verifyInType(getInType(), visited))
      return emitOpError("invalid type for allocation");
    if (verifyRecordLenParams(getInType(), numLenParams()))
      return emitOpError("LEN params do not correspond to type");
  }];
}

// Memory SSA operations

def fir_AllocaOp : fir_AllocatableOp<"alloca"> {
  let summary = "allocate storage for a temporary on the stack given a type";

  let description = [{
    This primitive operation is used to allocate an object on the stack.  A
    reference to the object of type `!fir.ref<T>` is returned.  The returned
    object has an undefined state.  The allocation can be given an optional
    name.  The allocation may have a dynamic repetition count for allocating
    a sequence of locations for the specified type.

      %0 = fir.alloca f32
      %1 = fir.call @foo(%0) : (!fir.ref<f32>) -> i32
  }];

  let results = (outs fir_ReferenceType);

  let verifier = allocVerify#[{
    mlir::Type outType = getType();
    if (!outType.isa<fir::ReferenceType>())
      return emitOpError("must be a !fir.ref type");
    return mlir::success();
  }];

  let extraClassDeclaration = extraAllocClassDeclaration#[{
    static mlir::Type wrapResultType(mlir::Type intype);
  }];
}

def fir_LoadOp : fir_OneResultOp<"load", []>,
    Arguments<(ins AnyReferenceLike:$memref)> {
  let summary = "load a value from a memory reference";

  let description = [{
    Load a value from a memory reference into a virtual register.  Produces
    an immutable ssa-value of the referent type.
  }];

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value refVal",
    [{
      if (!refVal) {
        mlir::emitError(result.location, "LoadOp has null argument");
        return;
      }
      fir::ReferenceType refTy = refVal.getType().cast<fir::ReferenceType>();
      result.addOperands(refVal);
      result.addTypes(refTy.getEleTy());
    }]
  >];

  let parser = [{
    mlir::Type type;
    mlir::OpAsmParser::OperandType oper;
    if (parser.parseOperand(oper) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColonType(type) ||
        parser.resolveOperand(oper, type, result.operands))
       return mlir::failure();
    mlir::Type eleTy;
    if (getElementOf(eleTy, type) ||
        parser.addTypeToList(eleTy, result.types))
       return mlir::failure();
    return mlir::success();
  }];

  let printer = [{
    p << getOperationName() << ' ';
    p.printOperand(memref());
    p.printOptionalAttrDict(getAttrs(), {});
    p << " : " << memref().getType();
  }];

  let extraClassDeclaration = [{
    static mlir::ParseResult getElementOf(mlir::Type &ele, mlir::Type ref);
  }];
}

def fir_StoreOp : fir_Op<"store", []>,
    Arguments<(ins AnyType:$value, AnyReferenceLike:$memref)> {
  let summary = "store an SSA-value to a memory location";

  let description = [{
    Store an ssa-value (virtual register) to a memory reference.  The stored
    value must be of the same type as the referent type of the memory
    reference.
  }];

  let parser = [{
    mlir::Type type;
    mlir::OpAsmParser::OperandType oper;
    mlir::OpAsmParser::OperandType store;
    if (parser.parseOperand(oper) ||
        parser.parseKeyword("to") ||
        parser.parseOperand(store) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColonType(type) ||
        parser.resolveOperand(oper, elementType(type),
          result.operands) ||
        parser.resolveOperand(store, type, result.operands))
       return mlir::failure();
    return mlir::success();
  }];

  let printer = [{
    p << getOperationName() << ' ';
    p.printOperand(value());
    p << " to ";
    p.printOperand(memref());
    p.printOptionalAttrDict(getAttrs(), {});
    p << " : " << memref().getType();
  }];

  let verifier = [{
    if (value().getType() != fir::dyn_cast_ptrEleTy(memref().getType()))
      return emitOpError("store value type must match memory reference type");
    return mlir::success();
  }];

  let extraClassDeclaration = [{
    static mlir::Type elementType(mlir::Type refType);
  }];
}

def fir_UndefOp : fir_OneResultOp<"undefined", [NoSideEffect]> {
  let summary = "explicit undefined value of some type";

  let description = [{
    Constructs an ssa-value of the specified type with an undefined value.
    This operation is typically created internally by the mem2reg conversion
    pass.
  }];

  let parser = [{
    mlir::Type intype;
    if (parser.parseType(intype) ||
        parser.addTypeToList(intype, result.types))
      return mlir::failure();
    return mlir::success();
  }];

  let printer = [{ p << getOperationName() << ' ' << getType(); }];

  let verifier = [{
    if (auto ref = getType().dyn_cast<fir::ReferenceType>())
      return emitOpError("undefined values of type !fir.ref not allowed");
    return mlir::success();
  }];
}

def fir_AllocMemOp : fir_AllocatableOp<"allocmem"> {
  let summary = "allocate storage on the heap for an object of a given type";

  let description = [{
    Creates a heap memory reference suitable for storing a value of the
    given type, T.  The heap refernce returned has type `!fir.heap<T>`.
    The memory object is in an undefined state.  `allocmem` operations must
    be paired with `freemem` operations to avoid memory leaks.

      %0 = fir.allocmem !fir.array<10xf32>
      fir.freemem %0 : !fir.heap<!fir.array<10xf32>>
  }];

  let results = (outs fir_HeapType);

  let verifier = allocVerify#[{
    mlir::Type outType = getType();
    if (!outType.dyn_cast<fir::HeapType>())
      return emitOpError("must be a !fir.heap type");
    return mlir::success();
  }];

  let extraClassDeclaration = extraAllocClassDeclaration#[{
    static mlir::Type wrapResultType(mlir::Type intype);
  }];
}

def fir_FreeMemOp : fir_Op<"freemem", []> {
  let summary = "free a heap object";

  let description = [{
    Deallocates a heap memory reference that was allocated by an `allocmem`.
    The memory object that is deallocated is placed in an undefined state
    after `fir.freemem`.  Optimizations may treat the loading of an object
    in the undefined state as undefined behavior.  This includes aliasing
    references, such as the result of an `fir.embox`.

      %0 = fir.allocmem !fir.array<10xf32>
      fir.freemem %0 : !fir.heap<!fir.array<10xf32>>
  }];

  let arguments = (ins fir_HeapType:$heapref);

  let assemblyFormat = "$heapref attr-dict `:` type($heapref)";
}

// Terminator operations

class fir_SwitchTerminatorOp<string mnemonic, list<OpTrait> traits = []> :
    fir_Op<mnemonic, !listconcat(traits, [Terminator])>,
    Arguments<(ins Variadic<AnyType>:$args)>,
    Results<(outs)> {
  let builders = [OpBuilder<
    "Builder *, OperationState &result, Value selector,"
    "ArrayRef<Value> properOperands, ArrayRef<Block *> destinations,"
    "ArrayRef<ArrayRef<Value>> operands = {},"
    "ArrayRef<NamedAttribute> attributes = {}",
    [{
      result.addOperands(selector);
      result.addOperands(properOperands);
      for (auto kvp : llvm::zip(destinations, operands)) {
        result.addSuccessor(std::get<0>(kvp), std::get<1>(kvp));
      }
      for (auto namedAttr : attributes) {
        result.addAttribute(namedAttr.first, namedAttr.second);
      }
    }]
  >];

  string extraSwitchClassDeclaration = [{
    using Conditions = mlir::Value;
    static constexpr auto AttrName = "cases";

    // The number of destination conditions that may be tested
    unsigned getNumConditions() { return getNumDest(); }

    // The selector is the value being tested to determine the destination
    mlir::Value getSelector() { return getOperand(0); }

    // The number of blocks that may be branched to
    unsigned getNumDest() { return getOperation()->getNumSuccessors(); }
  }];
}

class fir_IntegralSwitchTerminatorOp<string mnemonic,
    list<OpTrait> traits = []> : fir_SwitchTerminatorOp<mnemonic, traits> {
  let parser = [{
    mlir::OpAsmParser::OperandType selector;
    mlir::Type type;
    if (parseSelector(parser, result, selector, type))
      return mlir::failure();

    llvm::SmallVector<mlir::Attribute, 4> ivalues;
    llvm::SmallVector<mlir::Block *, 4> dests;
    llvm::SmallVector<llvm::SmallVector<mlir::Value, 4>, 4> destArgs;
    while (true) {
      mlir::Attribute ivalue; // Integer or Unit
      mlir::Block *dest;
      llvm::SmallVector<mlir::Value, 4> destArg;
      llvm::SmallVector<mlir::NamedAttribute, 1> temp;
      if (parser.parseAttribute(ivalue, "i", temp) ||
          parser.parseComma() ||
          parser.parseSuccessorAndUseList(dest, destArg))
        return mlir::failure();
      ivalues.push_back(ivalue);
      dests.push_back(dest);
      destArgs.push_back(destArg);
      if (!parser.parseOptionalRSquare())
        break;
      if (parser.parseComma())
        return mlir::failure();
    }
    result.addAttribute(AttrName, parser.getBuilder().getArrayAttr(ivalues));
    for (unsigned i = 0, count = dests.size(); i != count; ++i)
      result.addSuccessor(dests[i], destArgs[i]);
    return mlir::success();
  }];

  let printer = [{
    p << getOperationName() << ' ';
    p.printOperand(getSelector());
    p << " : " << getSelector().getType() << " [";
    auto cases = getAttrOfType<mlir::ArrayAttr>(AttrName).getValue();
    for (unsigned i = 0, count = getNumConditions(); i != count; ++i) {
      if (i)
        p << ", ";
      auto &attr = cases[i];
      if (auto intAttr = attr.dyn_cast_or_null<mlir::IntegerAttr>())
        p << intAttr.getValue();
      else
        p.printAttribute(attr);
      p << ", ";
      p.printSuccessorAndUseList(getOperation(), i);
    }
    p << ']';
    p.printOptionalAttrDict(getAttrs(), {AttrName});
  }];

  let verifier = [{
    if (!(getSelector().getType().isa<mlir::IntegerType>() ||
          getSelector().getType().isa<mlir::IndexType>() ||
          getSelector().getType().isa<fir::IntType>()))
      return emitOpError("must be an integer");
    auto cases = getAttrOfType<mlir::ArrayAttr>(AttrName).getValue();
    for (unsigned i = 0, count = getNumConditions(); i != count; ++i) {
      auto &attr = cases[i];
      if (attr.dyn_cast_or_null<mlir::IntegerAttr>()) {
        // ok
      } else if (attr.dyn_cast_or_null<mlir::UnitAttr>()) {
        // ok
      } else {
        return emitOpError("invalid case alternative");
      }
    }
    return mlir::success();
  }];

  let extraClassDeclaration = extraSwitchClassDeclaration;
}

def fir_SelectOp : fir_IntegralSwitchTerminatorOp<"select"> {
  let summary = "a multiway branch";

  let description = [{
    A multiway branch terminator with similar semantics to C's `switch`
    statement.  A selector value is matched against a list of constants
    of the same type for a match.  When a match is found, control is
    transferred to the corresponding basic block.  A `select` must have
    at least one basic block with a corresponding `unit` match, and
    that block will be selected when all other conditions fail to match.
  }];
}

def fir_SelectRankOp : fir_IntegralSwitchTerminatorOp<"select_rank"> {
  let summary = "Fortran's SELECT RANK statement";

  let description = [{
    Similar to `select`, `select_rank` provides a way to express Fortran's
    SELECT RANK construct.  In this case, the rank of the selector value
    is matched against constants of integer type.  The structure is the
    same as `select`, but `select_rank` determines the rank of the selector
    variable at runtime to determine the best match.
  }];
}

def fir_SelectCaseOp : fir_SwitchTerminatorOp<"select_case"> {
  let summary = "Fortran's SELECT CASE statement";

  let description = [{
    Similar to `select`, `select_case` provides a way to express Fortran's
    SELECT CASE construct.  In this case, the selector value is matched
    against variables (not just constants) and ranges.  The structure is
    the same as `select`, but `select_case` allows for the expression of
    more complex match conditions.
  }];

  let parser = [{
    mlir::OpAsmParser::OperandType selector;
    mlir::Type type;
    if (parseSelector(parser, result, selector, type))
      return mlir::failure();

    llvm::SmallVector<mlir::Attribute, 4> attrs;
    llvm::SmallVector<mlir::OpAsmParser::OperandType, 4> opers;
    llvm::SmallVector<mlir::Block *, 4> dests;
    llvm::SmallVector<llvm::SmallVector<mlir::Value, 4>, 4> destArgs;
    while (true) {
      mlir::Attribute attr;
      mlir::Block *dest;
      llvm::SmallVector<mlir::Value, 4> destArg;
      llvm::SmallVector<mlir::NamedAttribute, 1> temp;
      if (parser.parseAttribute(attr, "a", temp) ||
          isValidCaseAttr(attr) ||
          parser.parseComma())
        return mlir::failure();
      attrs.push_back(attr);
      if (attr.dyn_cast_or_null<mlir::UnitAttr>()) {
        // do nothing
      } else if (attr.dyn_cast_or_null<fir::ClosedIntervalAttr>()) {
        mlir::OpAsmParser::OperandType oper1;
        mlir::OpAsmParser::OperandType oper2;
        if (parser.parseOperand(oper1) ||
            parser.parseComma() ||
            parser.parseOperand(oper2) ||
            parser.parseComma())
          return mlir::failure();
        opers.push_back(oper1);
        opers.push_back(oper2);
      } else {
        mlir::OpAsmParser::OperandType oper;
        if (parser.parseOperand(oper) ||
            parser.parseComma())
          return mlir::failure();
        opers.push_back(oper);
      }
      if (parser.parseSuccessorAndUseList(dest, destArg))
        return mlir::failure();
      dests.push_back(dest);
      destArgs.push_back(destArg);
      if (!parser.parseOptionalRSquare())
        break;
      if (parser.parseComma())
        return mlir::failure();
    }
    result.addAttribute(AttrName, parser.getBuilder().getArrayAttr(attrs));
    if (parser.resolveOperands(opers, type, result.operands))
      return mlir::failure();
    for (unsigned i = 0, count = dests.size(); i != count; ++i)
      result.addSuccessor(dests[i], destArgs[i]);
    return mlir::success();
  }];

  let printer = [{
    p << getOperationName() << ' ';
    p.printOperand(getSelector());
    p << " : " << getSelector().getType() << " [";
    auto cases = getAttrOfType<mlir::ArrayAttr>(AttrName).getValue();
    for (unsigned i = 0, count = getNumConditions(); i != count; ++i) {
      if (i)
        p << ", ";
      p << cases[i] << ", ";
      if (!cases[i].dyn_cast_or_null<mlir::UnitAttr>()) {
        p.printOperand(getCaseArg(i, 0));
        p << ", ";
        if (cases[i].dyn_cast_or_null<fir::ClosedIntervalAttr>()) {
          p.printOperand(getCaseArg(i, 1));
          p << ", ";
        }
      }
      p.printSuccessorAndUseList(getOperation(), i);
    }
    p << ']';
    p.printOptionalAttrDict(getAttrs(), {AttrName});
  }];

  let verifier = [{
    auto selTy = getSelector().getType();
    if (!(selTy.isa<mlir::IntegerType>() || selTy.isa<mlir::IndexType>() ||
          selTy.isa<fir::IntType>() || selTy.isa<fir::LogicalType>() ||
          selTy.isa<fir::CharacterType>()))
      return emitOpError("must be an integer, character, or logical");
    auto cases = getAttrOfType<mlir::ArrayAttr>(AttrName).getValue();
    unsigned numOpers = 0;
    for (unsigned i = 0, count = getNumConditions(); i != count; ++i) {
      if (isValidCaseAttr(cases[i]))
        return emitOpError("incorrect select case attribute type");
      if (cases[i].isa<ClosedIntervalAttr>())
        numOpers += 2;
      else if (!cases[i].isa<mlir::UnitAttr>())
        numOpers++;
    }
    if (numOpers + 1 != getOperation()->getSuccessorOperandIndex(0))
      return emitOpError("incorrect number of operands");
    return mlir::success();
  }];

  let extraClassDeclaration = extraSwitchClassDeclaration#[{
    mlir::Value getCaseArg(unsigned dest, unsigned ele) {
      assert(ele < 2);
      assert(dest < getNumConditions());
      auto cases = getAttrOfType<mlir::ArrayAttr>(AttrName).getValue();
      assert(cases.size() == getNumConditions());
      unsigned o = getCaseArgumentOffset(cases, dest);
      return getOperand(o + 1 + ele);
    }
  }];
}

def fir_SelectTypeOp : fir_SwitchTerminatorOp<"select_type"> {
  let summary = "Fortran's SELECT TYPE statement";

  let description = [{
    Similar to `select`, `select_type` provides a way to express Fortran's
    SELECT TYPE construct.  In this case, the type of the selector value
    is matched against a list of type descriptors.  The structure is the
    same as `select`, but `select_type` determines the type of the selector
    variable at runtime to determine the best match.
  }];

  let parser = [{
    mlir::OpAsmParser::OperandType selector;
    mlir::Type type;
    if (parseSelector(parser, result, selector, type))
      return mlir::failure();

    llvm::SmallVector<mlir::Attribute, 4> attrs;
    llvm::SmallVector<mlir::Block *, 4> dests;
    llvm::SmallVector<llvm::SmallVector<mlir::Value, 4>, 4> destArgs;
    while (true) {
      mlir::Attribute attr;
      mlir::Block *dest;
      llvm::SmallVector<mlir::Value, 4> destArg;
      llvm::SmallVector<mlir::NamedAttribute, 1> temp;
      if (parser.parseAttribute(attr, "a", temp) ||
          parser.parseComma() ||
          parser.parseSuccessorAndUseList(dest, destArg))
        return mlir::failure();
      attrs.push_back(attr);
      dests.push_back(dest);
      destArgs.push_back(destArg);
      if (!parser.parseOptionalRSquare())
        break;
      if (parser.parseComma())
        return mlir::failure();
    }
    result.addAttribute(AttrName, parser.getBuilder().getArrayAttr(attrs));
    for (unsigned i = 0, count = dests.size(); i != count; ++i)
      result.addSuccessor(dests[i], destArgs[i]);
    return mlir::success();
  }];

  let printer = [{
    p << getOperationName() << ' ';
    p.printOperand(getSelector());
    p << " : " << getSelector().getType() << " [";
    auto cases = getAttrOfType<mlir::ArrayAttr>(AttrName).getValue();
    for (unsigned i = 0, count = getNumConditions(); i != count; ++i) {
      if (i)
        p << ", ";
      p << cases[i] << ", ";
      p.printSuccessorAndUseList(getOperation(), i);
    }
    p << ']';
    p.printOptionalAttrDict(getAttrs(), {AttrName});
  }];

  let verifier = [{
    if (!(getSelector().getType().isa<fir::BoxType>()))
      return emitOpError("must be a boxed type");
    auto cases = getAttrOfType<mlir::ArrayAttr>(AttrName).getValue();
    for (unsigned i = 0, count = getNumConditions(); i != count; ++i) {
      auto &attr = cases[i];
      if (attr.dyn_cast_or_null<fir::ExactTypeAttr>() ||
          attr.dyn_cast_or_null<fir::SubclassAttr>() ||
          attr.dyn_cast_or_null<mlir::UnitAttr>()) {
        // ok
      } else {
        return emitOpError("invalid type-case alternative");
      }
    }
    return mlir::success();
  }];
  let extraClassDeclaration = extraSwitchClassDeclaration;
}

def fir_UnreachableOp : fir_Op<"unreachable", [Terminator]> {
  let summary = "the unreachable instruction";

  let description = [{
    Terminates a basic block with the assertion that the end of the block
    will never be reached at runtime.  This instruction can be used
    immediately after a call to the Fortran runtime to terminate the
    program, for example.
  }];

  let parser = "return mlir::success();";

  let printer = "p << getOperationName();";
}

def fir_FirEndOp : fir_Op<"end", [Terminator]> {
  let summary = "the end instruction";

  let description = [{
    The end terminator is a special terminator used inside various FIR
    operations that have regions.  End is thus the custom (and only) terminator
    for these operations.  It is implicit and need not appear in the textual
    representation.
  }];
}

// Operations on !fir.box<T> type objects

def fir_EmboxOp : fir_Op<"embox", [NoSideEffect]> {
  let summary = "boxes a given reference and (optional) dimension information";

  let description = [{
    Create a boxed reference value. In Fortran, the implementation can require
    extra information about an entity, such as its type, rank, etc.  This
    auxilliary information is packaged and abstracted as a value with box type
    by the calling routine. (In Fortran, these are called descriptors.)

    Syntax:
      `fir.embox` mem-ref [`,` layout] `:` `(` arg-types `)` `->` res-type

      where layout is dims | affine-map

    Example:

      %4 = fir.dims(%c1, %c10, %c1) : (index, index, index) -> !fir.dims<1>
      %5 = ... : !fir.ref<!fir.array<10:i32>>
      %6 = fir.embox %5, %4 : (!fir.ref<!fir.array<10:i32>>, !fir.dims<1>)
                            -> !fir.box<!fir.array<10:i32>>
  }];

  let arguments = (ins AnyReferenceLike:$memref, Variadic<AnyEmboxArg>:$args);

  let results = (outs fir_BoxType);

  let parser = [{
    mlir::FunctionType type;
    llvm::SmallVector<mlir::OpAsmParser::OperandType, 8> operands;
    mlir::OpAsmParser::OperandType memref;
    if (parser.parseOperand(memref))
      return mlir::failure();
    operands.push_back(memref);
    auto &builder = parser.getBuilder();
    if (!parser.parseOptionalLParen()) {
      if (parser.parseOperandList(operands,
                                  mlir::OpAsmParser::Delimiter::None) ||
          parser.parseRParen())
        return mlir::failure();
      auto lens = builder.getI32IntegerAttr(operands.size());
      result.addAttribute(lenpName(), lens);
    }
    if (!parser.parseOptionalComma()) {
      mlir::OpAsmParser::OperandType dims;
      if (parser.parseOperand(dims))
        return mlir::failure();
      operands.push_back(dims);
    } else if (!parser.parseOptionalLSquare()) {
      mlir::AffineMapAttr map;
      if (parser.parseAttribute(map, layoutName(), result.attributes) ||
          parser.parseRSquare())
        return mlir::failure();
    }
    if (parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColonType(type) ||
        parser.resolveOperands(operands, type.getInputs(),
            parser.getNameLoc(), result.operands) ||
        parser.addTypesToList(type.getResults(), result.types))
      return mlir::failure();
    return mlir::success();
  }];

  let printer = [{
    p << getOperationName() << ' ';
    p.printOperand(memref());
    if (hasLenParams()) {
      p << '(';
      p.printOperands(getLenParams());
      p << ')';
    }
    if (getNumOperands() == 2) {
      p << ", ";
      p.printOperands(dims());
    } else if (auto map = getAttr(layoutName())) {
      p << " [" << map << ']';
    }
    p.printOptionalAttrDict(getAttrs(), {layoutName(), lenpName()});
    p << " : ";
    p.printFunctionalType(getOperation());
  }];

  let verifier = [{
    if (hasLenParams()) {
      auto lenParams = numLenParams();
      auto eleTy = fir::dyn_cast_ptrEleTy(memref().getType());
      if (!eleTy)
        return emitOpError("must embox a memory reference type");
      if (auto rt = eleTy.dyn_cast<fir::RecordType>()) {
        if (lenParams != rt.getNumLenParams())
          return emitOpError("number of LEN params does not correspond"
                             " to the !fir.type type");
      } else {
        return emitOpError("LEN parameters require !fir.type type");
      }
      for (auto lp : getLenParams())
        if (lp.getType().isa<fir::DimsType>())
          return emitOpError("LEN parameters must be integral type");
    }
    if (dims().size() == 0) {
      // Ok. If there is no dims and no layout map, then emboxing a scalar.
      // TODO: Should the type be enforced? It already must agree.
    } else if (dims().size() == 1) {
      auto d = *dims().begin();
      if (!d.getType().isa<fir::DimsType>())
        return emitOpError("dimension argument must have !fir.dims type");
    } else {
      return emitOpError("embox can only have one !fir.dim argument");
    }
    return mlir::success();
  }];

  let extraClassDeclaration = [{
    static constexpr llvm::StringRef layoutName() { return "layout_map"; }
    static constexpr llvm::StringRef lenpName() { return "len_param_count"; }
    bool hasLenParams() { return bool{getAttr(lenpName())}; }
    unsigned numLenParams() {
      if (auto x = getAttrOfType<mlir::IntegerAttr>(lenpName()))
        return x.getInt();
      return 0;
    }
    operand_range getLenParams() {
      return {operand_begin(), operand_begin() + numLenParams()};
    }
    operand_range dims() {
      return {operand_begin() + numLenParams() + 1, operand_end()};
    }
  }];
}

def fir_EmboxCharOp : fir_Op<"emboxchar", [NoSideEffect]> {
  let arguments = (ins AnyReferenceLike:$memref, AnyIntegerLike:$len);
  let results = (outs fir_BoxCharType);

  let summary = "boxes a given CHARACTER reference and its LEN parameter";

  let description = [{
    Create a boxed CHARACTER value. The CHARACTER type has the LEN type
    parameter, the value of which may only be known at runtime.  Therefore,
    a variable of type CHARACTER has both its data reference as well as a
    LEN type parameter.

      CHARACTER(LEN=10) :: var

      %4 = ...         : !fir.ref<!fir.array<10:!fir.char<1>>>
      %5 = constant 10 : i32
      %6 = fir.emboxchar %4, %5 : (!fir.ref<!fir.array<10:!fir.char<1>>>,
                                   i32) -> !fir.boxchar<1>
  }];

  let assemblyFormat = [{
    $memref `,` $len attr-dict `:` functional-type(operands, results)
  }];

  let verifier = [{
    auto eleTy = elementTypeOf(memref().getType());
    if (!eleTy.dyn_cast<CharacterType>())
      return mlir::failure();
    return mlir::success();
  }];
}

def fir_EmboxProcOp : fir_Op<"emboxproc", [NoSideEffect]> {

  let summary = "boxes a given procedure and optional host context";

  let description = [{
    Creates an abstract encapsulation of a PROCEDURE POINTER along with an
    optional pointer to a host instance context. If the pointer is not to an
    internal procedure or the internal procedure does not need a host context
    then the form takes only the procedure's symbol.

      %0 = fir.emboxproc @f : ((i32) -> i32) -> !fir.boxproc<(i32) -> i32>

    An internal procedure requiring a host instance for correct execution uses
    the second form. The closure of the host procedure's state is passed as a
    reference to a tuple. It is the responsibility of the host to manage the
    context's values accordingly, up to and including inhibiting register
    promotion of local values.

      %5 = fir.emboxproc @g, %4 : ((i32) -> i32, !fir.ref<tuple<i32, i32>>) ->
                                  !fir.boxproc<(i32) -> i32>
  }];

  let arguments = (ins SymbolRefAttr:$funcname, AnyReferenceLike:$host);

  let results = (outs fir_BoxProcType);

  let parser = [{
    mlir::SymbolRefAttr procRef;
    if (parser.parseAttribute(procRef, "funcname", result.attributes))
      return mlir::failure();
    bool hasTuple = false;
    mlir::OpAsmParser::OperandType tupleRef;
    if (!parser.parseOptionalComma()) {
      if (parser.parseOperand(tupleRef))
        return mlir::failure();
      hasTuple = true;
    }
    mlir::FunctionType type;
    if (parser.parseColon() ||
        parser.parseLParen() ||
        parser.parseType(type))
      return mlir::failure();
    result.addAttribute("functype", mlir::TypeAttr::get(type));
    if (hasTuple) {
      mlir::Type tupleType;
      if (parser.parseComma() ||
          parser.parseType(tupleType) ||
          parser.resolveOperand(tupleRef, tupleType, result.operands))
        return mlir::failure();
    }
    mlir::Type boxType;
    if (parser.parseRParen() ||
        parser.parseArrow() ||
        parser.parseType(boxType) ||
        parser.addTypesToList(boxType, result.types))
      return mlir::failure();
    return mlir::success();
  }];

  let printer = [{
    p << getOperationName() << ' ' << getAttr("funcname");
    auto h = host();
    if (h) {
      p << ", ";
      p.printOperand(h);
    }
    p << " : (" << getAttr("functype");
    if (h)
      p << ", " << h.getType();
    p << ") -> " << getType();
  }];

  let verifier = [{
    // host bindings (optional) must be a reference to a tuple
    if (auto h = host()) {
      if (auto r = h.getType().dyn_cast<ReferenceType>()) {
        if (!r.getEleTy().dyn_cast<mlir::TupleType>())
          return mlir::failure();
      } else {
        return mlir::failure();
      }
    }
    return mlir::success();
  }];
}

def fir_UnboxOp : fir_SimpleOp<"unbox", [NoSideEffect]> {
  let summary = "unbox the boxed value into a tuple value";

  let description = [{
    Unboxes a value of `box` type into a tuple of information abstracted in
    that boxed value.
  }];

  let arguments = (ins fir_BoxType:$box);

  let results = (outs
    fir_ReferenceType,  // pointer to data
    AnyIntegerLike,     // size of a data element
    AnyIntegerLike,     // rank of data
    fir_TypeDescType,   // abstract type descriptor
    AnyIntegerLike,     // attribute flags (bitfields)
    fir_DimsType        // dimension information (if any)
  );
}

def fir_UnboxCharOp : fir_SimpleOp<"unboxchar", [NoSideEffect]>,
    Arguments<(ins fir_BoxCharType:$boxchar)>,
    Results<(outs fir_ReferenceType, AnyIntegerLike)> {
  let summary = "unbox a boxchar value into a pair value";

  let description = [{
    Unboxes a value of `boxchar` type into a pair consisting of a memory
    reference to the CHARACTER data and the LEN type parameter.
  }];
}

def fir_UnboxProcOp : fir_SimpleOp<"unboxproc", [NoSideEffect]>,
    Arguments<(ins fir_BoxProcType:$boxproc)>,
    Results<(outs FunctionType, fir_ReferenceType:$refTuple)> {
  let summary = "unbox a boxproc value into a pair value";

  let description = [{
    Unboxes a value of `boxproc` type into a pair consisting of a procedure
    pointer and a pointer to a host context.
  }];

  let verifier = [{
    if (auto eleTy = fir::dyn_cast_ptrEleTy(refTuple().getType()))
      if (eleTy.isa<mlir::TupleType>())
        return mlir::success();
    return emitOpError("second output argument has bad type");
  }];
}

def fir_BoxAddrOp : fir_SimpleOneResultOp<"box_addr", [NoSideEffect]> {

  let summary = "return a memory reference to the boxed value";

  let description = [{
    This operator is overloaded to work with values of type `box`,
    `boxchar`, and `boxproc`.  The result for each of these
    cases, respectively, is the address of the data, the address of the
    CHARACTER data, and the address of the procedure.
  }];

  let arguments = (ins fir_BoxType:$val);

  let results = (outs AnyReferenceLike);
}

def fir_BoxCharLenOp : fir_SimpleOp<"boxchar_len", [NoSideEffect]> {
  let summary = "return the LEN type parameter from a boxchar value";

  let description = [{
    Extracts the LEN type parameter from a `boxchar` value.
  }];

  let arguments = (ins fir_BoxCharType:$val);

  let results = (outs AnyIntegerLike);
}

def fir_BoxDimsOp : fir_Op<"box_dims", [NoSideEffect]> {
  let summary = "return the dynamic dimension information for the boxed value";

  let description = [{
    Returns the triple of lower bound, extent, and stride for `dim` dimension
    of `val`, which must have a `box` type.  The dimensions are enumerated from
    left to right from 0 to rank-1. This operation has undefined behavior if
    `dim` is out of bounds.
  }];

  let arguments = (ins fir_BoxType:$val, AnyIntegerLike:$dim);

  let results = (outs AnyIntegerLike, AnyIntegerLike, AnyIntegerLike);

  let assemblyFormat = [{
    $val `,` $dim attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    mlir::Type getTupleType();
  }];
}

def fir_BoxEleSizeOp : fir_SimpleOneResultOp<"box_elesize", [NoSideEffect]> {
  let summary = "return the size of an element of the boxed value";

  let description = [{
    Returns the size of an element in an entity of `box` type.  This size may
    not be known until runtime.
  }];

  let arguments = (ins fir_BoxType:$val);

  let results = (outs AnyIntegerLike);
}

def fir_BoxIsAllocOp : fir_SimpleOp<"box_isalloc", [NoSideEffect]>,
    Arguments<(ins fir_BoxType:$val)>,
    Results<(outs BoolLike)> {
  let summary = "is the boxed value an ALLOCATABLE?";

  let description = [{
    Determine if the boxed value was from an ALLOCATABLE entity.

      %ref = ... : !fir.heap<i64>
      %box = fir.embox %ref : !fir.box<i64>
      %isheap = fir.box_isalloc %box : i1
  }];
}

def fir_BoxIsArrayOp : fir_SimpleOp<"box_isarray", [NoSideEffect]>,
    Arguments<(ins fir_BoxType:$val)>,
    Results<(outs BoolLike)> {
  let summary = "is the boxed value an array?";

  let description = [{
    Determine if the boxed value has a positive (> 0) rank.

      %ref = ... : !fir.ref<i64>
      %dims = fir.gendims(1, 100, 1) : !fir.dims<1>
      %box = fir.embox %ref, %dims : !fir.box<i64>
      %isarr = fir.box_isarray %box : i1
  }];
}

def fir_BoxIsPtrOp : fir_SimpleOp<"box_isptr", [NoSideEffect]>,
    Arguments<(ins fir_BoxType:$val)>,
    Results<(outs BoolLike)> {
  let summary = "is the boxed value a POINTER?";

  let description = [{
    Determine if the boxed value was from a POINTER entity.

      %ptr = ... : !fir.ptr<i64>
      %box = fir.embox %ptr : !fir.box<i64>
      %isptr = fir.box_isptr %box : i1
  }];
}

def fir_BoxProcHostOp : fir_SimpleOp<"boxproc_host", [NoSideEffect]>,
    Arguments<(ins fir_BoxProcType:$val)>,
    Results<(outs fir_ReferenceType)> {
  let summary = "returns the host instance pointer (or null)";

  let description = [{
    Extract the host context pointer from a `boxproc` value.
  }];
}

def fir_BoxRankOp : fir_SimpleOneResultOp<"box_rank", [NoSideEffect]> {
  let summary = "return the number of dimensions for the boxed value";

  let description = [{
    Return the rank of a value of `box` type.  If the value is scalar, the
    rank is 0.
  }];

  let arguments = (ins fir_BoxType:$val);

  let results = (outs AnyIntegerType);
}

def fir_BoxTypeDescOp : fir_SimpleOneResultOp<"box_tdesc", [NoSideEffect]> {
  let summary = "return the type descriptor for the boxed value";

  let description = [{
    Return the opaque type descriptor of a value of `box` type.
  }];

  let arguments = (ins fir_BoxType:$val);

  let results = (outs fir_TypeDescType);
}

// Record and array type operations

def fir_CoordinateOp : fir_Op<"coordinate_of", [NoSideEffect]>,
    Arguments<(ins AnyRefOrBox:$ref, Variadic<AnyCoordinateType>:$coor)>,
    Results<(outs fir_ReferenceType)> {
  let summary = "Finds the coordinate (location) of a value in memory";

  let description = [{
    Determine a memory reference given a memory reference of composite type
    and a list of index values. (This returns the address of a value.)
  }];

  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];

  let verifier = [{
    // Recovering a LEN type parameter only makes sense from a boxed value
    for (auto co : coor())
      if (auto *s = co.getDefiningOp())
        if (dyn_cast_or_null<LenParamIndexOp>(s)) {
          if (getNumOperands() != 2)
            return emitOpError("len_param_index must be last argument");
          if (!ref().getType().dyn_cast<BoxType>())
            return emitOpError("len_param_index must be used on box type");
        }
    return mlir::success();
  }];
}

def fir_ExtractValueOp : fir_OneResultOp<"extract_value", [NoSideEffect]>,
    Arguments<(ins AnyCompositeLike:$adt, Variadic<AnyComponentType>:$coor)> {
  let summary = "Extract a value from an aggregate SSA-value";

  let description = [{
    Extract a subobject value given a value of composite type and a list of
    index values.
  }];

  let assemblyFormat = [{
    $adt `,` $coor attr-dict `:` functional-type(operands, results)
  }];
}

def fir_FieldIndexOp : fir_OneResultOp<"field_index", [NoSideEffect]>,
    Arguments<(ins StrAttr:$field_id, TypeAttr:$on_type,
               Variadic<AnyIntegerType>:$lenparams)> {
  let summary = "create a field index value from a field identifier";

  let description = [{
    Generate a field (offset) value from an identifier.  Field values may be
    lowered into exact offsets when the layout of a Fortran derived type is
    known at compile-time. The type of a field value is `!fir.field` and
    these values can be used with the `fir.coordinate_of`, `fir.extract_value`,
    or `fir.insert_value` instructions to compute (abstract) addresses of
    subobjects.
  }];

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, StringRef fieldName,"
    "Type recTy, ArrayRef<Value> operands = {}",
    [{
      result.addAttribute(fieldAttrName(), builder->getStringAttr(fieldName));
      result.addAttribute(typeAttrName(), TypeAttr::get(recTy));
      result.addOperands(operands);
    }]
  >];

  let parser = [{
    llvm::StringRef fieldName;
    auto &builder = parser.getBuilder();
    mlir::Type recty;
    if (parser.parseOptionalKeyword(&fieldName) ||
        parser.parseComma() ||
        parser.parseType(recty))
      return mlir::failure();
    result.addAttribute(fieldAttrName(), builder.getStringAttr(fieldName));
    if (!recty.dyn_cast<RecordType>())
      return mlir::failure();
    result.addAttribute(typeAttrName(), mlir::TypeAttr::get(recty));
    if (!parser.parseOptionalLParen()) {
      llvm::SmallVector<mlir::OpAsmParser::OperandType, 8> operands;
      llvm::SmallVector<mlir::Type, 8> types;
      auto loc = parser.getNameLoc();
      if (parser.parseOperandList(operands,
                                  mlir::OpAsmParser::Delimiter::None) ||
          parser.parseRParen() ||
          parser.parseColonTypeList(types) ||
          parser.resolveOperands(operands, types, loc, result.operands))
        return mlir::failure();
    }
    mlir::Type fieldType = fir::FieldType::get(builder.getContext());
    if (parser.addTypeToList(fieldType, result.types))
      return mlir::failure();
    return mlir::success();
  }];

  let printer = [{
    p << getOperationName() << ' '
      << getAttrOfType<mlir::StringAttr>(fieldAttrName()).getValue() << ", "
      << getAttr(typeAttrName());
    if (getNumOperands()) {
      p << '(';
      p.printOperands(lenparams());
      auto sep = ") : ";
      for (auto op : lenparams()) {
        p << sep;
        if (op)
          p.printType(op.getType());
        else
          p << "()";
        sep = ", ";
      }
    }
  }];

  let extraClassDeclaration = [{
    static constexpr llvm::StringRef fieldAttrName() { return "field_id"; }
    static constexpr llvm::StringRef typeAttrName() { return "on_type"; }
  }];
}

def fir_GenDimsOp : fir_OneResultOp<"gendims", [NoSideEffect]> {

  let summary = "generate a value of type `!fir.dims`";

  let description = [{
    The arguments are an ordered list of integral type values that is a
    multiple of 3 in length.  Each such triple is defined as: the lower
    index, the extent, and the stride for that dimension. The dimension
    information is given in the same row-to-column order as Fortran. This
    abstract dimension value must describe a reified object, so all dimension
    information must be specified.  The extent must be non-negative and the
    stride must not be zero.
  }];

  let arguments = (ins Variadic<AnyIntegerType>:$triples);

  let results = (outs fir_DimsType);

  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];

  let verifier = [{
    auto size = triples().size();
    if (size < 1 && size <= 16 * 3)
      return emitOpError("incorrect number of args");
    if (size % 3 != 0)
      return emitOpError("requires a multiple of 3 args");
    return mlir::success();
  }];
}

def fir_InsertValueOp : fir_OneResultOp<"insert_value", [NoSideEffect]> {
  let summary = "insert a new sub-value into a copy of an existing aggregate";

  let description = [{
    Insert a value into a composite value.
  }];

  let arguments = (ins AnyCompositeLike:$adt, AnyType:$val,
                       Variadic<AnyComponentType>:$coor);
  let results = (outs AnyCompositeLike);

  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
}

def fir_LenParamIndexOp : fir_OneResultOp<"len_param_index", [NoSideEffect]>,
    Arguments<(ins StrAttr:$field_id, TypeAttr:$on_type)> {
  let summary =
    "create a field index value from a LEN type parameter identifier";

  let description = [{
    Generate a LEN parameter (offset) value from an LEN parameter identifier.
    The type of a LEN parameter value is `!fir.len` and these values can be
    used with the `fir.coordinate_of` instructions to compute (abstract)
    addresses of LEN parameters.
  }];

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, StringRef fieldName, Type recTy",
    [{
      result.addAttribute(fieldAttrName(), builder->getStringAttr(fieldName));
      result.addAttribute(typeAttrName(), TypeAttr::get(recTy));
    }]
  >];

  let parser = [{
    llvm::StringRef fieldName;
    auto &builder = parser.getBuilder();
    mlir::Type recty;
    if (parser.parseOptionalKeyword(&fieldName) ||
        parser.parseComma() ||
        parser.parseType(recty))
      return mlir::failure();
    result.addAttribute(fieldAttrName(), builder.getStringAttr(fieldName));
    if (!recty.dyn_cast<RecordType>())
      return mlir::failure();
    result.addAttribute(typeAttrName(), mlir::TypeAttr::get(recty));
    mlir::Type lenType = fir::LenType::get(builder.getContext());
    if (parser.addTypeToList(lenType, result.types))
      return mlir::failure();
    return mlir::success();
  }];

  let printer = [{
    p << getOperationName() << ' '
      << getAttrOfType<mlir::StringAttr>(fieldAttrName()).getValue() << ", "
      << getAttr(typeAttrName());
  }];

  let extraClassDeclaration = [{
    static constexpr llvm::StringRef fieldAttrName() { return "field_id"; }
    static constexpr llvm::StringRef typeAttrName() { return "on_type"; }
    mlir::Type getOnType() {
      return getAttrOfType<TypeAttr>(typeAttrName()).getValue();
    }
  }];
}

// Fortran loops

def ImplicitFirTerminator : SingleBlockImplicitTerminator<"FirEndOp">;

def fir_LoopOp : fir_Op<"loop", [ImplicitFirTerminator]> {
  let summary = "generalized loop operation";
  let description = [{
    A generalized Fortran loop construct.
  }];

  let arguments = (ins
    Index:$lowerBound,
    Index:$upperBound,
    Variadic<Index>:$optStep,
    OptionalAttr<I32Attr>:$constantStep,
    OptionalAttr<UnitAttr>:$unordered
  );

  let results = (outs Variadic<Index>:$lastVal);

  let regions = (region SizedRegion<1>:$region);
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"mlir::Builder *builder, mlir::OperationState &result,"
              "int64_t lowerBound, int64_t upperBound, int64_t step = 1">,
    OpBuilder<"mlir::Builder *builder, mlir::OperationState &result,"
              "mlir::Value lowerBound, mlir::Value upperBound,"
              "llvm::ArrayRef<mlir::Value> step">
  ];

  let parser = "return parseLoopOp(parser, result);";

  let printer = [{
    p << getOperationName() << ' ' << getInductionVar() << " = "
      << lowerBound() << " to " << upperBound();
    auto s = optStep();
    if (s.begin() != s.end()) {
      p << " step ";
      p.printOperand(*s.begin());
    }
    if (unordered())
      p << " unordered";
    p.printRegion(region(), /*printEntryBlockArgs=*/false,
        /*printBlockTerminators=*/false);
    p.printOptionalAttrDict(getAttrs(), {unorderedName(), stepName()});
  }];

  let verifier = [{
    auto step = optStep();
    if (step.begin() != step.end()) {
      // FIXME: size of step must be 1
      auto *s = (*step.begin()).getDefiningOp();
      if (auto cst = dyn_cast_or_null<mlir::ConstantIndexOp>(s))
        if (cst.getValue() == 0)
          return emitOpError("constant step operand must be nonzero");
    }

    // Check that the body defines as single block argument for the induction
    // variable.
    auto *body = getBody();
    if (body->getNumArguments() != 1 ||
        !body->getArgument(0).getType().isIndex())
      return emitOpError("expected body to have a single index argument for "
                         "the induction variable");
    if (lastVal().size() > 1)
      return emitOpError("can only return one final value of iterator");
    return mlir::success();
  }];

  let extraClassDeclaration = [{
    static constexpr llvm::StringRef unorderedName() { return "unordered"; }
    static constexpr llvm::StringRef stepName() { return "step"; }

    bool isUnordered() { return getAttr(unorderedName()).isa<UnitAttr>(); }

    bool hasLastValue() { return lastVal().size(); }

    /// Get the body of the loop
    mlir::Block *getBody() { return &region().front(); }

    /// Get the block argument corresponding to the loop control value (PHI)
    mlir::Value getInductionVar() { return getBody()->getArgument(0); }

    mlir::OpBuilder getBodyBuilder() {
      return mlir::OpBuilder(getBody(), std::prev(getBody()->end()));
    }

    void setLowerBound(mlir::Value bound) {
      getOperation()->setOperand(0, bound);
    }

    void setUpperBound(mlir::Value bound) {
      getOperation()->setOperand(1, bound);
    }

    void setStep(mlir::Value step) {
      getOperation()->setOperand(2, step);
    }
  }];
}

def fir_WhereOp : fir_Op<"where", [ImplicitFirTerminator]> {
  let summary = "generalized conditional operation";
  let description = [{
    This is a generalized conditional construct.
  }];
  let arguments = (ins I1:$condition);
  let regions = (region SizedRegion<1>:$whereRegion, AnyRegion:$otherRegion);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, "
              "Value cond, bool withOtherRegion">
  ];
  let parser = [{ return parseWhereOp(parser, result); }];

  let printer = [{
    p << getOperationName() << ' ' << condition();
    p.printRegion(whereRegion(), /*printEntryBlockArgs=*/false,
        /*printBlockTerminators=*/false);

    // Print the 'else' regions if it exists and has a block.
    auto &otherReg = otherRegion();
    if (!otherReg.empty()) {
      p << " otherwise";
      p.printRegion(otherReg, /*printEntryBlockArgs=*/false,
          /*printBlockTerminators=*/false);
    }
    p.printOptionalAttrDict(getAttrs());
  }];

  let verifier = [{
    for (auto &region : getOperation()->getRegions()) {
      if (region.empty())
        continue;
      for (auto &b : region)
        if (b.getNumArguments() != 0)
          return emitOpError("requires that child entry blocks have no args");
    }
    return mlir::success();
  }];

  let extraClassDeclaration = [{
    mlir::OpBuilder getWhereBodyBuilder() {
      assert(!whereRegion().empty() && "Unexpected empty 'where' region.");
      mlir::Block &body = whereRegion().front();
      return mlir::OpBuilder(&body, std::prev(body.end()));
    }
    mlir::OpBuilder getOtherBodyBuilder() {
      assert(!otherRegion().empty() && "Unexpected empty 'other' region.");
      mlir::Block &body = otherRegion().front();
      return mlir::OpBuilder(&body, std::prev(body.end()));
    }
  }];
}

// Procedure call operations

def fir_CallOp : fir_Op<"call", []>,
    Arguments<(ins OptionalAttr<SymbolRefAttr>:$callee,
               Variadic<AnyType>:$args)>,
    Results<(outs Variadic<AnyType>)> {
  let summary = "call a procedure";

  let description = [{
    Provides a custom parser and pretty printer to allow a more readable syntax
    in the FIR dialect, e.g. `fir.call @sub(%12)` or `fir.call %20(%22,%23)`.
  }];

  let parser = "return parseCallOp(parser, result);";
  let printer = "printCallOp(p, *this);";
}

def fir_DispatchOp : fir_Op<"dispatch", []>,
    Arguments<(ins StrAttr:$method, fir_BoxType:$object,
               Variadic<AnyType>:$args)>,
    Results<(outs Variadic<AnyType>)> {
  let summary = "call a type-bound procedure";

  let description = [{
    Dynamic dispatch to the specified method.
  }];

  let parser = [{
    mlir::FunctionType calleeType;
    llvm::SmallVector<mlir::OpAsmParser::OperandType, 4> operands;
    auto calleeLoc = parser.getNameLoc();
    llvm::StringRef calleeName;
    if (parser.parseOptionalKeyword(&calleeName)) {
      mlir::StringAttr calleeAttr;
      if (parser.parseAttribute(calleeAttr, "method", result.attributes))
        return mlir::failure();
    } else {
      result.addAttribute("method",
          parser.getBuilder().getStringAttr(calleeName));
    }
    if (parser.parseOperandList(operands,
                                mlir::OpAsmParser::Delimiter::Paren) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColonType(calleeType) ||
        parser.addTypesToList(calleeType.getResults(), result.types) ||
        parser.resolveOperands(
            operands, calleeType.getInputs(), calleeLoc, result.operands))
      return mlir::failure();
    result.addAttribute("fn_type", mlir::TypeAttr::get(calleeType));
    return mlir::success();
  }];

  let printer = [{
    p << getOperationName() << ' ' << getAttr("method") << '(';
    p.printOperand(object());
    if (arg_operand_begin() != arg_operand_end()) {
      p << ", ";
      p.printOperands(args());
    }
    p << ')';
    p.printOptionalAttrDict(getAttrs(), {"fn_type", "method"});
    auto resTy{getResultTypes()};
    llvm::SmallVector<mlir::Type, 8> argTy(getOperandTypes());
    p << " : " << mlir::FunctionType::get(argTy, resTy, getContext());
  }];

  let extraClassDeclaration = [{
    mlir::FunctionType getFunctionType();
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }
    operand_iterator arg_operand_begin() { return operand_begin() + 1; }
    operand_iterator arg_operand_end() { return operand_end(); }
    llvm::StringRef passArgAttrName() { return "pass_arg_pos"; }
    unsigned passArgPos();
  }];
}

// Constant operations that support Fortran

def fir_StringLitOp : fir_Op<"string_lit", [NoSideEffect]>,
    Results<(outs fir_SequenceType)> {
  let summary = "create a string literal constant";

  let description = [{
    An FIR constant that represents a sequence of characters that correspond
    to Fortran's CHARACTER type, including a LEN.  We support CHARACTER values
    of different KINDs (different constant sizes).

    Example:

      %1 = fir.string_lit "Hello, World!"(13) : !fir.char<1> // ASCII
      %2 = fir.string_lit [158, 2345](2) : !fir.char<2>      // Wide chars
  }];

  let parser = [{
    auto &builder = parser.getBuilder();
    mlir::Attribute val;
    llvm::SmallVector<mlir::NamedAttribute, 2> attrs;
    if (parser.parseAttribute(val, "fake", attrs))
      return mlir::failure();
    if (auto v = val.dyn_cast<mlir::StringAttr>())
      result.attributes.push_back(builder.getNamedAttr(value(), v));
    else if (auto v = val.dyn_cast<mlir::ArrayAttr>())
      result.attributes.push_back(builder.getNamedAttr(xlist(), v));
    else
      return mlir::failure();
    mlir::IntegerAttr sz;
    mlir::Type type;
    if (parser.parseLParen() ||
        parser.parseAttribute(sz, size(), result.attributes) ||
        parser.parseRParen() || parser.parseColonType(type))
      return mlir::failure();
    type = fir::SequenceType::get({sz.getInt()}, type);
    if (parser.addTypesToList(type, result.types))
      return mlir::failure();
    return mlir::success();
  }];

  let printer = [{
    p << getOperationName() << ' ' << getValue() << '(';
    p << getSize().cast<mlir::IntegerAttr>().getValue() << ") : ";
    p.printType(getType().cast<fir::SequenceType>().getEleTy());
  }];

  let verifier = [{
    if (getSize().cast<mlir::IntegerAttr>().getValue().isNegative())
      return emitOpError("size must be non-negative");
    auto eleTy = getType().cast<fir::SequenceType>().getEleTy();
    if (!eleTy.dyn_cast<fir::CharacterType>())
      return emitOpError("must have !fir.char type");
    if (auto xl = getAttr(xlist())) {
      auto xList = xl.cast<mlir::ArrayAttr>();
      for (auto a : xList)
        if (!a.isa<mlir::IntegerAttr>())
	  return emitOpError("values in list must be integers");
    }
    return mlir::success();
  }];

  let extraClassDeclaration = [{
    static constexpr llvm::StringRef size() { return "size"; }
    static constexpr llvm::StringRef value() { return "value"; }
    static constexpr llvm::StringRef xlist() { return "xlist"; }

    // Get the LEN attribute of this character constant
    mlir::Attribute getSize() { return getAttr(size()); }
    
    // Get the string value of this character constant
    mlir::Attribute getValue() {
      if (auto attr = getAttr(value()))
        return attr;
      return getAttr(xlist());
    }

    /// Is this a wide character literal (1 character > 8 bits)
    bool isWideValue();
  }];
}

// Complex operations

class fir_ArithmeticOp<string mnemonic, list<OpTrait> traits = []> :
    fir_Op<mnemonic,
       !listconcat(traits, [NoSideEffect, SameOperandsAndResultType])>,
    Results<(outs AnyType)> {
  let parser = [{
    return impl::parseOneResultSameOperandTypeOp(parser, result);
  }];

  let printer = [{ return fir::printBinaryOp(this->getOperation(), p); }];
}

class fir_UnaryArithmeticOp<string mnemonic, list<OpTrait> traits = []> :
      fir_Op<mnemonic,
         !listconcat(traits, [NoSideEffect, SameOperandsAndResultType])>,
      Results<(outs AnyType)> {
  let parser = [{
    return impl::parseOneResultSameOperandTypeOp(parser, result);
  }];

  let printer = [{ return fir::printUnaryOp(this->getOperation(), p); }];
}

def fir_ConstfOp : fir_Op<"constf", [NoSideEffect]>,
    Results<(outs fir_RealType)> {
  let summary = "create a floating point constant";

  let description = [{
    FIXME
  }];

  let parser = [{
    fir::RealAttr flt;
    mlir::Type type;
    if (parser.parseLParen() ||
        parser.parseAttribute(flt, constAttrName(), result.attributes) ||
        parser.parseRParen() || parser.parseColonType(type) ||
        parser.addTypesToList(type, result.types))
      return mlir::failure();
    return mlir::success();
  }];

  let printer = [{
    p << getOperationName() << ' ' << getAttr(constAttrName()) << " : ";
    p.printType(getType());
  }];

  let verifier = [{
    if (!getType().isa<fir::RealType>())
      return emitOpError("must be a !fir.real type");
    return mlir::success();
  }];

  let extraClassDeclaration = [{
    static constexpr llvm::StringRef constAttrName() { return "constant"; }

    mlir::Attribute getValue() { return getAttr(constAttrName()); }
  }];
}

class RealUnaryArithmeticOp<string mnemonic, list<OpTrait> traits = []> :
      fir_UnaryArithmeticOp<mnemonic, traits>,
      Arguments<(ins AnyRealLike:$operand)>;

def fir_NegfOp : RealUnaryArithmeticOp<"negf">;

class RealArithmeticOp<string mnemonic, list<OpTrait> traits = []> :
      fir_ArithmeticOp<mnemonic, traits>,
      Arguments<(ins AnyRealLike:$lhs, AnyRealLike:$rhs)>;

def fir_AddfOp : RealArithmeticOp<"addf", [Commutative]>;
def fir_SubfOp : RealArithmeticOp<"subf">;
def fir_MulfOp : RealArithmeticOp<"mulf", [Commutative]>;
def fir_DivfOp : RealArithmeticOp<"divf">;
def fir_ModfOp : RealArithmeticOp<"modf">;
// Pow is a builtin call and not a primitive

def fir_CmpfOp : fir_Op<"cmpf",
    [NoSideEffect, SameTypeOperands, SameOperandsAndResultShape]> {
  let summary = "floating-point comparison operator";

  let description = [{
    Extends the standard floating-point comparison to handle the extended
    floating-point types found in FIR.
  }];

  let arguments = (ins AnyRealLike:$lhs, AnyRealLike:$rhs);

  let results = (outs AnyLogicalLike);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, CmpFPredicate predicate,"
    "Value lhs, Value rhs", [{
      fir::buildCmpFOp(builder, result, predicate, lhs, rhs);
  }]>];

  let parser = [{ return fir::parseCmpfOp(parser, result); }];

  let printer = [{ fir::printCmpfOp(p, *this); }];

  let extraClassDeclaration = [{
    static constexpr llvm::StringRef getPredicateAttrName() {
      return "predicate";
    }
    static CmpFPredicate getPredicateByName(llvm::StringRef name);

    CmpFPredicate getPredicate() {
      return (CmpFPredicate)getAttrOfType<mlir::IntegerAttr>(
          getPredicateAttrName()).getInt();
    }
  }];
}

def fir_ConstcOp : fir_Op<"constc", [NoSideEffect]>,
    Results<(outs fir_ComplexType)> {
  let summary = "create a complex constant";

  let description = [{
    FIXME
  }];

  let parser = [{
    fir::RealAttr realp;
    fir::RealAttr imagp;
    mlir::Type type;
    if (parser.parseLParen() ||
        parser.parseAttribute(realp, realAttrName(), result.attributes) ||
        parser.parseComma() ||
        parser.parseAttribute(imagp, imagAttrName(), result.attributes) ||
        parser.parseRParen() ||
        parser.parseColonType(type) ||
        parser.addTypesToList(type, result.types))
      return mlir::failure();
    return mlir::success();
  }];

  let printer = [{
    p << getOperationName() << " (0x";
    auto f1 = getAttr(realAttrName()).cast<mlir::FloatAttr>();
    auto i1 = f1.getValue().bitcastToAPInt();
    p.getStream().write_hex(i1.getZExtValue());
    p << ", 0x";
    auto f2 = getAttr(imagAttrName()).cast<mlir::FloatAttr>();
    auto i2 = f2.getValue().bitcastToAPInt();
    p.getStream().write_hex(i2.getZExtValue());
    p << ") : ";
    p.printType(getType());
  }];

  let verifier = [{
    if (!getType().isa<fir::CplxType>())
      return emitOpError("must be a !fir.complex type");
    return mlir::success();
  }];

  let extraClassDeclaration = [{
    static constexpr llvm::StringRef realAttrName() { return "real"; }
    static constexpr llvm::StringRef imagAttrName() { return "imaginary"; }

    mlir::Attribute getReal() { return getAttr(realAttrName()); }
    mlir::Attribute getImaginary() { return getAttr(imagAttrName()); }
  }];
}

class ComplexUnaryArithmeticOp<string mnemonic, list<OpTrait> traits = []> :
      fir_UnaryArithmeticOp<mnemonic, traits>,
      Arguments<(ins fir_ComplexType:$operand)>;

def fir_NegcOp : ComplexUnaryArithmeticOp<"negc">;

class ComplexArithmeticOp<string mnemonic, list<OpTrait> traits = []> :
      fir_ArithmeticOp<mnemonic, traits>,
      Arguments<(ins fir_ComplexType:$lhs, fir_ComplexType:$rhs)>;

def fir_AddcOp : ComplexArithmeticOp<"addc", [Commutative]>;
def fir_SubcOp : ComplexArithmeticOp<"subc">;
def fir_MulcOp : ComplexArithmeticOp<"mulc", [Commutative]>;
def fir_DivcOp : ComplexArithmeticOp<"divc">;
// Pow is a builtin call and not a primitive

def fir_CmpcOp : fir_Op<"cmpc",
    [NoSideEffect, SameTypeOperands, SameOperandsAndResultShape]>,
    Arguments<(ins fir_ComplexType:$lhs, fir_ComplexType:$rhs)>,
    Results<(outs AnyLogicalLike)> {
  let summary = "complex floating-point comparison operator";

  let description = [{
    A complex comparison to handle complex types found in FIR.
  }];

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, CmpFPredicate predicate,"
    "Value lhs, Value rhs", [{
      fir::buildCmpCOp(builder, result, predicate, lhs, rhs);
  }]>];

  let parser = [{ return fir::parseCmpcOp(parser, result); }];
  let printer = [{ fir::printCmpcOp(p, *this); }];

  let extraClassDeclaration = [{
    static constexpr llvm::StringRef getPredicateAttrName() {
      return "predicate";
    }

    CmpFPredicate getPredicate() {
      return (CmpFPredicate)getAttrOfType<mlir::IntegerAttr>(
          getPredicateAttrName()).getInt();
    }
  }];
}

// Other misc. operations

def fir_AddrOfOp : fir_OneResultOp<"address_of", [NoSideEffect]> {
  let summary = "convert a symbol to an SSA value";

  let description = [{
    Convert a symbol (a function or global reference) to an SSA-value to be
    used in other Operations.
  }];

  let arguments = (ins SymbolRefAttr:$symbol);

  let parser = [{
    mlir::SymbolRefAttr attr;
    mlir::Type type;
    if (parser.parseLParen() ||
        parser.parseAttribute(attr, "symbol", result.attributes) ||
        parser.parseRParen() ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColonType(type) ||
        parser.addTypeToList(type, result.types))
      return mlir::failure();
    return mlir::success();
  }];

  let printer = [{
    p << getOperationName() << " (" << getAttr("symbol") << ')';
    p.printOptionalAttrDict(getAttrs(), {"symbol"});
    p << " : " << getType();
  }];
}

def fir_ConvertOp : fir_OneResultOp<"convert", [NoSideEffect]> {
  let arguments = (ins AnyType:$value);
  let summary = "encapsulates all Fortran scalar type conversions";

  let description = [{
    Generalized type conversion. Not all pairs of types have conversions.
  }];

  let assemblyFormat = [{
    $value attr-dict `:` functional-type($value, results)
  }];
}

def FortranTypeAttr : Attr<And<[CPred<"$_self.isa<TypeAttr>()">,
    Or<[CPred<"$_self.cast<TypeAttr>().getValue().isa<fir::CharacterType>()">,
        CPred<"$_self.cast<TypeAttr>().getValue().isa<fir::CplxType>()">,
        CPred<"$_self.cast<TypeAttr>().getValue().isa<fir::IntType>()">,
        CPred<"$_self.cast<TypeAttr>().getValue().isa<fir::LogicalType>()">,
        CPred<"$_self.cast<TypeAttr>().getValue().isa<fir::RealType>()">,
        CPred<"$_self.cast<TypeAttr>().getValue().isa<fir::RecordType>()">]>]>,
    "Fortran surface type"> {
  let storageType = [{ TypeAttr }];
  let returnType = "Type";
  let convertFromStorage = "$_self.getValue().cast<Type>()";
}

def fir_GenTypeDescOp : fir_OneResultOp<"gentypedesc", [NoSideEffect]>,
  Arguments<(ins FortranTypeAttr:$in_type)> {
  let summary = "generate a type descriptor for a given type";

  let description = [{
    Generates a constant object that is an abstract type descriptor of the
    specified type.  The meta-type of a type descriptor for the type `T`
    is `!fir.tdesc<T>`.
  }];

  let parser = [{
    mlir::Type intype;
    if (parser.parseType(intype))
      return mlir::failure();
    result.addAttribute("in_type", mlir::TypeAttr::get(intype));
    mlir::Type restype = TypeDescType::get(intype);
    if (parser.addTypeToList(restype, result.types))
      return mlir::failure();
    return mlir::success();
  }];

  let builders = [
    OpBuilder<"Builder *, OperationState &result, mlir::TypeAttr inty">
  ];

  let printer = [{
    p << getOperationName() << ' ' << getAttr("in_type");
    p.printOptionalAttrDict(getAttrs(), {"in_type"});
  }];

  let verifier = [{
    mlir::Type resultTy = getType();
    if (auto tdesc = resultTy.dyn_cast<TypeDescType>()) {
      if (tdesc.getOfTy() != getInType())
        return emitOpError("wrapped type mismatched");
    } else {
      return emitOpError("must be !fir.tdesc type");
    }
    return mlir::success();
  }];

  let extraClassDeclaration = [{
    mlir::Type getInType() {
      // get the type that the type descriptor describes
      return getAttrOfType<mlir::TypeAttr>("in_type").getValue();
    }
  }];
}

def fir_NoReassocOp : fir_OneResultOp<"no_reassoc",
    [SameOperandsAndResultType]> {
  let arguments = (ins AnyType:$val);
  let summary = "synthetic op to prevent reassociation";

  let description = [{
    The operation is to make sure that the Fortran optimizer does not
    reassociate operations when they are syntactically surrounded by
    parenthesis.
  }];

  let assemblyFormat = "$val attr-dict `:` type($val)";
}

def fir_GlobalEntryOp : fir_Op<"global_entry", []> {

  let summary = "map a field entry in a global";

  let description = [{
    An entry in a global.  Allows a constant to be bound to a specific
    field in a record even though the layout of the record has not yet
    been reified.  This may be necessary if the object layout decisions
    are deferred to, for example, runtime by the implementation.
  }];

  let arguments = (ins StrAttr:$field, AnyType:$constant);

  let parser = [{
    mlir::Type type;
    mlir::OpAsmParser::OperandType constant;
    llvm::StringRef fieldName;
    if (parser.parseOptionalKeyword(&fieldName)) {
      mlir::StringAttr fieldAttr;
      if (parser.parseAttribute(fieldAttr, "field", result.attributes))
        return mlir::failure();
    } else {
      result.addAttribute("field",
          parser.getBuilder().getStringAttr(fieldName));
    }
    if (parser.parseComma() ||
        parser.parseOperand(constant) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColonType(type) ||
        parser.resolveOperand(constant, type, result.operands))
      return mlir::failure();
    result.addAttribute("in_type", mlir::TypeAttr::get(type));
    return mlir::success();
  }];
  let printer = [{
    p << getOperationName() << ' ' << getAttr("field") << ", ";
    p.printOperand(constant());
    p.printOptionalAttrDict(getAttrs(), {"in_type", "field"});
    p << " : " << getAttr("in_type");
  }];
}

def fir_DTEntryOp : fir_Op<"dt_entry", []>,
    Arguments<(ins StrAttr:$method, SymbolRefAttr:$proc)> {
  let summary = "map entry in a dispatch table";

  let description = [{
    An entry in a dispatch table.  Allows a function symbol to be bound
    to a specifier method identifier.  A dispatch operation uses the dynamic
    type of a distinguished argument to determine an exact dispatch table
    and uses the method identifier to select the type-bound procedure to
    be called.
  }];

  let parser = [{
    mlir::SymbolRefAttr calleeAttr;
    llvm::StringRef methodName;
    if (parser.parseOptionalKeyword(&methodName)) {
      mlir::StringAttr methodAttr;
      if (parser.parseAttribute(methodAttr, "method", result.attributes))
        return mlir::failure();
    } else {
      result.addAttribute("method",
          parser.getBuilder().getStringAttr(methodName));
    }
    if (parser.parseComma() ||
        parser.parseAttribute(calleeAttr, "proc", result.attributes))
      return mlir::failure();
    return mlir::success();
  }];
  let printer = [{
    p << getOperationName() << ' ' << getAttr("method") << ", "
      << getAttr("proc");
  }];
}

#endif
