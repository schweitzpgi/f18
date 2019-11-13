# Elemental Expression Lowering To FIR
FIR operations are not elemental. Instead elemental expression from the parse-tree must be lowered to `fir.loop` operations
that are looping over the array elements.
FIR lowering of array expressions allow the implementation of the design in 
https://github.com/flang-compiler/f18/blob/master/documentation/ArrayComposition.md.
Lowering could implement the transformations proposed in the above mentioned document while lowering.
However, it also may not have to assuming it will be possible to perform these transformations later on FIR.

## Observations
### Storage must be associated to array expression to lower it
The first observation is that a `fir.loop` is more or less a regular loop and
it is not returning an ssa value that would be the result of the array expression associated to the loop.
Instead, a `fir.store` operation in the `fir.loop` must store array result elements into some memory reference.
This implies that every `fir.loop` computing an array expression must be given a memory reference that can store
the array result.


### Scalar sub-expression must be evaluated at most once.
In `array + cos(scalar)`, `cos(scalar)` must not be evaluated for every element of `array`.

### Shape may be a compiled time constant shape or not
In case it is not, FIR must be inserted to compute the `fir.loop` bounds.
In case the resulting shape is empty (one dimension extent is zero),
nothing should be evaluated in the expression unless it was evaluated to deduce the shape.

### Short-circuiting requirements
On top of the null-shape case, they are other cases that should be short-circuited. This is stated here because this
should combine well with array expression lowering.
Case identified so far are:
- `E` is null shaped -> do not evaluate `E`
- `b.OR.SE`, `b.AND.SE` -> do not evaluate `SE` when `b` is false and `SE` may have side effects.
- Same as above for the right side if the left side is array shaped?
- `c1 // c2` if the consumer of the expression only needs a length `<= length(c1)`.
- Should we skip evaluation on IEEE NaN, Inf... (we know `qNan + foo()` is `qNaN`) ?

This short circuiting logic has to be implemented before the `fir.loop`,
it may require to produce different `fir.loop` based on the path taken. For instance, in `b.AND.SE`,
after evaluating `b`, one path should set the temp result to `SE` while the other should set it to `.False.`.

Short-circuiting requires introducing a CFG of basic blocks which makes
expression lowering changing the basic block context while emitting FIR.
The creation of basic block for short-circuiting can comply to the following rules:

While lowering expr `E` in basic block `B1`, expression lowering will
return with the builder pointing at the end of basic block `Bi` such as:
- `B1` strictly dominates `Bi` (all paths from `B1` leads to `Bi` and all paths arriving to `Bi` went through `B1`).
- `Bi` may or may not be `B1`.
- The SSA value returned by expression lowering is defined in `Bi` and is the value of the expression `E`
- The CFG graph starting at `B1` and ending in `Bi` is a DAG (considering `fir.loop` is an
 operation and not a CFG loop).

These rules mainly mean that expression lowering does not have to (and should not) create crazy CFG in order
to achieve short-circuiting.

These rules should guarantee that we do not have to track the CFG state while lowering an expression other than the current insertion point
(and maybe some places to load symbols. allocate temps...).

### An array shape does not imply an expression is elemental
All expressions with an array shape are not elemental, for instance `foo(array)` where `foo` is a user
function that is not elemental cannot be replaced by call to `foo` on each `array` element.

### Some non-elemental expression can be seen as an elemental expression and an affine map
As stated in https://github.com/flang-compiler/f18/blob/master/documentation/ArrayComposition.md,
transformational functions are not elemental but they many of them can simply be viewed as be viewed as
changing the way their array arguments are to be accessed (i.e/ they are functions operating on the indexes and can be seen in MLIR as affine maps).

### Do not early optimize but do not prevent later optimizations
Expression lowering from the parse-tree to FIR is not meant to be an optimization pass,
it should only produce FIR that can be optimized later. For instance, if as stated in ArrayComposition.md,
it is an end goal to avoid unnecessary temporary arrays, lowering is not the place to perform deep temporary removal optimization.
It is not entirely clear yet what "can be optimized later" implies.
The strategy should be to start lowering in a simple way and if it turns out that some optimizations are later not possible, it should be
preferred to find a way to convey the information that is missing in FIR to enable the optimization rather than to perform the optimization
while lowering.

### Not all elemental operation should be decomposed
`merge(x, y, b)` when b is scalar does not require a loop to be implemented.
It can be implemented as single branch to load the right value or even better, a merge
on the memory reference.

Question: If lowering were to emit a loop, would later optimization be able to transform this to a simple branch ?

Note: This anyway falls in the short-circuiting cases.

Also, the simple expression `array` should not be broken into a loop if this is not needed,
it has the value `fir.load %array`.

## Expression Lowering Design

### Proposed Generated FIR structure for one array expression
1. Load all required symbols (+)
2. Compute shape (if not compiled time constant)
3. Short-circuit if null shape
4. Allocate tmp result (stack or heap?) (+)
5. Compute scalar operands (+)
6. Short-circuit if scalar operands are enough for result
7. Compute array sub-expressions (+)
8. Loop on element (+)
9. short-circuit targets 

Steps 2. to 9. are repeated for each "non-inlined" array sub expression.
By "non-inlined", it is meant that it is a sub-expression that is computed in its own loop and
is associated its own memory storage.

For each FIR part followed by (+), we may need to either do a pass or keep track of the MLIR insertion point
while going through the array expression so that the FIR is emitted at the right location to get
the structure described above.

TODO: discuss and justify this structure
### Proposed level of sub-expression inlining
Several alternatives regarding "inlining" of sub-expression computation in their parent expression `fir.loop`
are discussed below. The plan is to start by the easiest and to see if FIR optimizers cannot re-group
the sub-expression into one loop when it is possible. 

#### One Array Expression Node - One Loop
Every array node of an expression tree is lowered to its own loop
and a temp storage created for it.
That is `x + y + z` where `x`, `y` and `z` are arrays ends up as two loops, the first computing
`x+z` and storing this in a temp array `tmp1`, the second loop computes `x + tmp1` into a second temp `tmp2`.

Pros: Very easy lowering. We do not need to keep of the current `fir.loop` we should emit code into(it must be the one we just created for the node we are visiting)
, and assuming we are going through the expression tree in a postfix way, we do not need to keep track of where to emit sub-expressions, they already were.
Cons: Will generate many loops and temporaries that we know without deep analysis are not needed.

#### One "Perfectly" Elemental Expression Sub-Tree - One Loop
`x + y + z` is lowered into a single loop that directly computes `x(i) + y(i) + z(i)` for each element and store it into a single temp.
When one of the array sub-expression is not elemental, it is computed on its own outside of the loop and stored into another temp that is then accessed before the loop.
For instance, `x + foo(y) + z` where `foo(y)` is an array ends up:
- compute `foo(y)` into `tmp1`
- compute `x + tmp1 + z`
With this alternative, transformational such as `transpose` are still treated as `foo`.
Pros: Generates less FIR, the original expression structure is better preserved inside the loop.
Cons: Slightly more complex lowering, we need to keep track of the loop where we want to emit the "inlined" operations and where to insert
the computation of sub-expressions that are not "inlined".

#### One "Affine" Elemental Expression Sub-Tree - One Loop
Same as before, but also apply transformationals as index re-mapping instead of computing it outside of the loop.
For instance `x + transpose(y) + z` becomes a loop computing each `x(i, j) + y(j, i) + z(i, j)`.
Pros: Generates minimal FIR. Not much more complex that the previous one (mostly the same thing).
Cons: Requires more knowledge about transformationals, an index mapping state must be propagated while lowering.


## Examples
Below are illustrations of Fortran to FIR lowering based on the above design (with one node, one loop). 

### Simple examples
Fortran:
```
  real :: x(4), y(4)
  y = x
```
Here, `x` expression should not be split into a loop.

FIR:
```
 %1 = fir.load %x // that is the whole expression evaluation
 // This is actually the assignment lowering
 %fir.store %1 to %y
```
Question: This is more related to assignment lowering, but what will decide whether `%1`
is has to be copied inside a temporary or not (e.g. in case of pointers that may overlap or not).
Is it implicit in here because `%1` requires a storage later and
FIR will determine whether `%y` can directly be used ? 
### Outline scalar sub-expression outside of the loop
Fortran:
```
  real :: x(4), y(4)
  y = x + cos(1.)
```
FIR:
```
  %tmp = fir.alloca !fir.array<4:f32>
  %cst = constant 1.000000e+00 : f32
  %1 = fir.call "cos" %cst
  fir.loop %i = 1 to 4 {
    %2 = fir.extract_value %x, %i
    %3 = fir.addf %2, %1
    %4 = fir.coordinate_of %tmp, %i
    fir.store %3 to %4 
  }
  %5 = fir.load %tmp // This is the expression result
  // The assignment part is not part of the expression lowering
  fir.store %5 to %y
```
### Dynamic shape example
A more complex case where the shape is not a compile time constant.
Fortran:
```
   real, allocatable :: x(:), y(:)
   y = x + cos(1.)
```
FIR:
```
  %_,_, %1 = fir.box_dims %x
  %cst_0 = constant 0 : i64
  %3 = cmpi eq %1 %cst_0
  cond_br %3 ^bnullshape ^bb1
  ^bb1:
    %tmp = fir.allocmem !fir.array<?:f32> %1
    %cst = constant 1.000000e+00 : f32
    %cosres = fir.call "cos" %cst
    %lbx1, _, _ = fir.box_dims %x
    %cst_1 = constant 1 : i64
    %lbx0 = subi lbx1 %cst_1
    fir.loop %i = 1 to %1 {

      %ix = subi %i %lbx0
      %2 = fir.coordinate_of %x, %i
      %3 = fir.load %2
      %4 = fir.addf %3, %cosres
      %5 = fir.coordinate_of %tmp, %i
      fir.store %4 to %5
    }
    %6 = fir.load %tmp
    fir.freemem %tmp // are we allowed to do this so soon?
    jmp bresult(%6)
  ^bnullshape:
    // Create a null shaped result.
    %nullres = undef !fir.array<0:f32>
    %r = fir.convert %nullres !fir.array<?:f32> // not sure we need this
    jmp bresult(%r)
  ^bresult(%result : !fir.array<?:f32>)
    //%result is the expression result

  // The assignment is not part of the expression lowering
  // It is also more complex because y is an allocatable so not shown
  // here
```

Observation: The loop is on the shape, not the actual bounds of `x`. So we need to compute the related coordinate for `x` inside the loop.
Here, it is a bit stupid, but in `x + y`, it does not make sense to loop on `x` bounds because the bounds of y might be different, so we loop on
the extent starting from 1.

Questions: Does it matter? Could we hide this inside `fir.coordinate_of` ?

### Short-circuiting example
Scalar driven short-circuiting
In case `b1` is scalar and `foo()` has side effects in `b1.AND.foo()`,
f18 guarantees `foo()` will only be called if required (when `b1` is true).
This short circuiting logic has to be evaluated before the `fir.loop`.

Fortran:
```
 real :: x(4), y(4), z(4)
 z = merge(x, y, b1.AND.foo())
```
`foo` returns an array with `x` shape. 

FIR:
```
  ^bb0
    %tmp = fir.alloca !fir.array<4:f32>
    %1 = fir.load b1
    %2 = fir.convert b1 i1
    %3 = cond_br %i1 ^bb1 ^bb2(%1)
  ^bb1:
    %4 = fir.call "foo"
    jmp ^bb3(%4)
  ^bb2
    %tmp2 = fir.alloca !fir.array<4:fir.logical<4>>
    for.loop j 1 to 4 {
      %5 = fir.coordinate_of %tmp, %i
      fir.store %1 to %5
    }
    %6 = fir.load %tmp2
    jmp ^bb3(%6)
  ^bb3(%bool : fir.array<4:fir.logical<4>>)
    fir.loop %i = 1 to 4 {
    %7 = fir.extract_value %x, %i
    %8 = fir.extract_value %y, %i
    %9 = fir.call "fir.merge.i32" %7 %8 %bool
    %10 = fir.coordinate_of %tmp, %i
    fir.store %9 to %10
    }
    %11 = fir.load %tmp // This is the expression result

    // The assignment part is not part of the expression lowering
    fir.store %11 to %z
```


Observation: In the example above, in case `b1` is false, we are going through two loops for nothing.
(the result is simply `fir.load %y`). We could instead have rewritten the expression as
`merge(merge(x, y, foo()), y, b1)` which would lead to FIR:
```
  ^bb0
    %1 = fir.load b1
    %2 = fir.convert b1 i1
    %3 = cond_br %i1 ^bb1 ^bb2(%y)
  ^bb1:
    %tmp = fir.alloca !fir.array<4:f32> 
    %4 = fir.call "foo"
    fir.loop %i = 1 to 4 {
      %5 = fir.extract_value %x, %i
      %6 = fir.extract_value %y, %i
      %7 = fir.extract_value %4, %i
      %8 = fir.call "fir.merge.i32" %5 %6 %7
      %9 = fir.coordinate_of %tmp, %i
      fir.store %8 to %9
    }
    jmp ^bb3(%4)
  ^bb2(%resRef : fir.ref<fir.array<4:f32>>)
    %res = fir.load %resRef // This is the expression result
  ```

Question: Can optimizers still do this kind of rewriting later?

### Questions from examples
Arrays lower bounds may be different from 1, for fir.array, this knowledge disappears in FIR.
*Stupid question*: Is `fir.extract_value` on a `fir.array` 0 or 1 based ?
For things coming out of a descriptor (fir.box), the knowledge is still there
(inside the fir.box `access-map`).

If we allocate temporaries on the heap, when can we deallocate them (is it safe to access after the `fir.freemem` an
ssa value that was loaded from it before the `fir.freemem`)?
