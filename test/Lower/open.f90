! RUN: bbc %s -o - | FileCheck %s

! CHECK-LABEL: _QQmain
! CHECK: call {{.*}}BeginOpenUnit
! CHECK-DAG: call {{.*}}SetFile
! CHECK-DAG: call {{.*}}SetAccess
! CHECK: call {{.*}}EndIoStatement

  open(8, file="foo", access="sequential")

! CHECK: call {{.*}}BeginBackspace
! CHECK: call {{.*}}EndIoStatement
  backspace(8)
  
! CHECK: call {{.*}}BeginFlush
! CHECK: call {{.*}}EndIoStatement
  flush(8)
  
! CHECK: call {{.*}}BeginRewind
! CHECK: call {{.*}}EndIoStatement
  rewind(8)

! CHECK: call {{.*}}BeginClose
! CHECK: call {{.*}}EndIoStatement
  close(8)
end
