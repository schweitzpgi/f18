! RUN: bbc %s -emit-fir | FileCheck %s

!CHECK: func @_QPsub() {
subroutine sub()
! CHECK: }
end subroutine

!CHECK: func @_QPasubroutine() {
subroutine AsUbRoUtInE()
! CHECK: }
end subroutine

! CHECK-LABEL: func @_QPfoo() -> f32 {
function foo()
  real(4) :: foo
! CHECK: }
end function

module testMod
contains
  !CHECK: func @_QMtestmodPsub() {
  subroutine sub()
  ! CHECK: }
  end subroutine

  ! CHECK-LABEL: func @_QMtestmodPfoo() -> f32 {
  function foo()
    real(4) :: foo
  ! CHECK: }
  end function
end module


function foo2()
  real(4) :: foo2
contains
  !CHECK: func @_QFfoo2Psub() {
  subroutine sub()
  ! CHECK: }
  end subroutine

  !CHECK: func @_QFfoo2Pfoo() {
  subroutine foo()
  ! CHECK: }
  end subroutine
end function

subroutine sUb2()
contains
  !CHECK: func @_QFsub2Psub() {
  subroutine sub()
  ! CHECK: }
  end subroutine

  !CHECK: func @_QFsub2Pfoo() {
  subroutine Foo()
  ! CHECK: }
  end subroutine
end subroutine

module testMod2
contains
  subroutine sub()
  contains
    !CHECK: func @_QMtestmod2FsubPsubsub() {
    subroutine subSub()
    ! CHECK: }
    end subroutine
  end subroutine
end module


module color_points
  interface
    module subroutine draw()
    end subroutine
    module function erase()
      integer(4) :: erase
    end function
  end interface
end module color_points

submodule (color_points) color_points_a
contains
  !CHECK: func @_QMcolor_pointsScolor_points_aPsub() {
  subroutine sub
  end subroutine
  ! CHECK: }
end submodule

submodule (color_points:color_points_a) impl
contains
  subroutine foo
    contains
    !CHECK: func @_QMcolor_pointsScolor_points_aSimplFfooPbar() {
    subroutine bar
    ! CHECK: }
    end subroutine
  end subroutine
  !CHECK: func @_QMcolor_pointsPdraw() {
  module subroutine draw()
  end subroutine
  !FIXME func @_QMcolor_pointsPerase() -> i32 {
  module procedure erase
  ! CHECK: }
  end procedure
end submodule


! CHECK: func @MAIN_() {
program test
! CHECK: }
end program
