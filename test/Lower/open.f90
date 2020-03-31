! RUN: bbc %s -o - | FileCheck %s

! CHECK-LABEL: _QQmain
! CHECK: call {{.*}}BeginOpenUnit
! CHECK-DAG: call {{.*}}SetFile
! CHECK-DAG: call {{.*}}SetAccess
! CHECK: call {{.*}}EndIoStatement

  open(8, file="foo", access="sequential")
end
