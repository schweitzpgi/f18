! Note: The Fortran runtime libraries have dependences on the C++ runtime and
! LLVM libraries.  To work around the former, this test explicitly links in
! libstdc++.a.  To work around the latter, the source of Common/enum-set.h was
! hacked to exclude references to llvm ADTs.
! Note: On linux, the Fortran runtime wants to include libm as well.

! RUN: bbc %s -o - | tco | llc --filetype=obj -o %t.o
! RUN: cc -I%S/../.. %S/main.c -c -o %t.main.o
! RUN: cc %t.o %t.main.o -L%L -lFortranRuntime -lFortranDecimal -lstdc++ -lm
! RUN: ./a.out | FileCheck %s

! CHECK: Hello, World!
  print *, "Hello, World!"
  end
