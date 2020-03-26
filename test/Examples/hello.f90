! RUN: bbc %s -o - | tco | llc --filetype=obj -o %t.o
! RUN: cc -I%S/../.. %S/main.c -c -o %t.main.o
! RUN: cc %t.o %t.main.o $(llvm-config --ldflags --libs) -L$(dirname $(which bbc))/../lib -lFortranRuntime -lFortranDecimal -lstdc++
! RUN: ./a.out | FileCheck %s

! CHECK: Hello, World!
  print *, "Hello, World!"
  end
