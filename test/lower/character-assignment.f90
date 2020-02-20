! Simple character assignment tests
subroutine assign1(s1, s2)
  character(*, 1) :: s1, s2
  s1 = s2
end subroutine
subroutine assign2(s1, s2)
  character(*, 2) :: s1, s2
  s1 = s2
end subroutine
subroutine assign4(s1, s2)
  character(*, 4) :: s1, s2
  s1 = s2
end subroutine

! Test substring assignment
subroutine assign_substring1(s1, s2, lb, ub)
  character(*, 1) :: s1, s2
  integer :: lb, ub
  s1(lb:ub) = s2
end subroutine
subroutine assign_substring2(s1, s2, lb, ub)
  character(*, 2) :: s1, s2
  integer :: lb, ub
  s1(lb:ub) = s2
end subroutine
subroutine assign_substring4(s1, s2, lb, ub)
  character(*, 4) :: s1, s2
  integer :: lb, ub
  s1(lb:ub) = s2
end subroutine

! Test overlapping assignments
! RHS depends on LHS in a way that if no temp is made to evaluate
! RHS, these tests should fail.
subroutine assign_overlap1(s1, s2, lb)
  character(*, 1) :: s1, s2
  integer :: lb
  s1(lb:) = s1
end subroutine
subroutine assign_overlap2(s1, s2, lb)
  character(*, 2) :: s1, s2
  integer :: lb
  s1(lb:) = s1
end subroutine
subroutine assign_overlap4(s1, s2, lb)
  character(*, 4) :: s1, s2
  integer :: lb
  s1(lb:) = s1
end subroutine

! Test when length is given in specification expressions
subroutine assign_spec_expr_len1(s1, s2, l1, l2)
  integer :: l1, l2
  character(l1, 1) :: s1
  character(l2, 1) :: s2
  s1 = s2
end subroutine
subroutine assign_spec_expr_len2(s1, s2, l1, l2)
  integer :: l1, l2
  character(l1, 2) :: s1
  character(l2, 2) :: s2
  s1 = s2
end subroutine
subroutine assign_spec_expr_len4(s1, s2, l1, l2)
  integer :: l1, l2
  character(l1, 4) :: s1
  character(l2, 4) :: s2
  s1 = s2
end subroutine

subroutine assign_hello_world1(s1)
  character(*, 1) :: s1
  s1 = "Hello World!"
end subroutine

! FIXME: Tests with constant with string_lit + kind 2 or 4 fails when lowering
! to LLVM IR.
!
!subroutine assign_hello_world2(s1)
!  character(*, 2) :: s1
!  s1 = 2_"Hello World!"
!end subroutine
!
! bbc -emit-llvm error:
! f18-llvm-project/llvm/lib/IR/Instructions.cpp:1364:
! void llvm::StoreInst::AssertOK(): Assertion `getOperand(0)->getType() ==
! cast<PointerType>(getOperand(1)->getType())->getElementType() && "Ptr must be
! a pointer to Val type!"' failed.
!
! Yet, the types look ok at the FIR level:
!
!    %0 = fir.string_lit "H"(12) : !fir.char<2>
!    %c12_i64 = constant 12 : i64
!    %1 = fir.alloca !fir.array<12x!fir.char<2>>
!    fir.store %0 to %1 : !fir.ref<!fir.array<12x!fir.char<2>>>

