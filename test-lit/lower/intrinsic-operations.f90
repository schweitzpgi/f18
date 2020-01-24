! Generated test file
! RUN: bbc %s | FileCheck %s
! CHECK-LABEL:eqv0_test
LOGICAL(1) FUNCTION eqv0_test(x0, x1)
LOGICAL(1) :: x0
LOGICAL(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = cmpi "eq", [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<1>
eqv0_test = x0 .EQV. x1
END FUNCTION

! CHECK-LABEL:eqv1_test
LOGICAL(2) FUNCTION eqv1_test(x0, x1)
LOGICAL(1) :: x0
LOGICAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = cmpi "eq", [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<2>
eqv1_test = x0 .EQV. x1
END FUNCTION

! CHECK-LABEL:eqv2_test
LOGICAL(4) FUNCTION eqv2_test(x0, x1)
LOGICAL(1) :: x0
LOGICAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = cmpi "eq", [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<4>
eqv2_test = x0 .EQV. x1
END FUNCTION

! CHECK-LABEL:eqv3_test
LOGICAL(8) FUNCTION eqv3_test(x0, x1)
LOGICAL(1) :: x0
LOGICAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = cmpi "eq", [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<8>
eqv3_test = x0 .EQV. x1
END FUNCTION

! CHECK-LABEL:eqv4_test
LOGICAL(2) FUNCTION eqv4_test(x0, x1)
LOGICAL(2) :: x0
LOGICAL(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = cmpi "eq", [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<2>
eqv4_test = x0 .EQV. x1
END FUNCTION

! CHECK-LABEL:eqv5_test
LOGICAL(2) FUNCTION eqv5_test(x0, x1)
LOGICAL(2) :: x0
LOGICAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = cmpi "eq", [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<2>
eqv5_test = x0 .EQV. x1
END FUNCTION

! CHECK-LABEL:eqv6_test
LOGICAL(4) FUNCTION eqv6_test(x0, x1)
LOGICAL(2) :: x0
LOGICAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = cmpi "eq", [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<4>
eqv6_test = x0 .EQV. x1
END FUNCTION

! CHECK-LABEL:eqv7_test
LOGICAL(8) FUNCTION eqv7_test(x0, x1)
LOGICAL(2) :: x0
LOGICAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = cmpi "eq", [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<8>
eqv7_test = x0 .EQV. x1
END FUNCTION

! CHECK-LABEL:eqv8_test
LOGICAL(4) FUNCTION eqv8_test(x0, x1)
LOGICAL(4) :: x0
LOGICAL(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = cmpi "eq", [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<4>
eqv8_test = x0 .EQV. x1
END FUNCTION

! CHECK-LABEL:eqv9_test
LOGICAL(4) FUNCTION eqv9_test(x0, x1)
LOGICAL(4) :: x0
LOGICAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = cmpi "eq", [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<4>
eqv9_test = x0 .EQV. x1
END FUNCTION

! CHECK-LABEL:eqv10_test
LOGICAL(4) FUNCTION eqv10_test(x0, x1)
LOGICAL(4) :: x0
LOGICAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = cmpi "eq", [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<4>
eqv10_test = x0 .EQV. x1
END FUNCTION

! CHECK-LABEL:eqv11_test
LOGICAL(8) FUNCTION eqv11_test(x0, x1)
LOGICAL(4) :: x0
LOGICAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = cmpi "eq", [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<8>
eqv11_test = x0 .EQV. x1
END FUNCTION

! CHECK-LABEL:eqv12_test
LOGICAL(8) FUNCTION eqv12_test(x0, x1)
LOGICAL(8) :: x0
LOGICAL(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = cmpi "eq", [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<8>
eqv12_test = x0 .EQV. x1
END FUNCTION

! CHECK-LABEL:eqv13_test
LOGICAL(8) FUNCTION eqv13_test(x0, x1)
LOGICAL(8) :: x0
LOGICAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = cmpi "eq", [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<8>
eqv13_test = x0 .EQV. x1
END FUNCTION

! CHECK-LABEL:eqv14_test
LOGICAL(8) FUNCTION eqv14_test(x0, x1)
LOGICAL(8) :: x0
LOGICAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = cmpi "eq", [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<8>
eqv14_test = x0 .EQV. x1
END FUNCTION

! CHECK-LABEL:eqv15_test
LOGICAL(8) FUNCTION eqv15_test(x0, x1)
LOGICAL(8) :: x0
LOGICAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = cmpi "eq", [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<8>
eqv15_test = x0 .EQV. x1
END FUNCTION

! CHECK-LABEL:neqv16_test
LOGICAL(1) FUNCTION neqv16_test(x0, x1)
LOGICAL(1) :: x0
LOGICAL(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = cmpi "ne", [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<1>
neqv16_test = x0 .NEQV. x1
END FUNCTION

! CHECK-LABEL:neqv17_test
LOGICAL(2) FUNCTION neqv17_test(x0, x1)
LOGICAL(1) :: x0
LOGICAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = cmpi "ne", [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<2>
neqv17_test = x0 .NEQV. x1
END FUNCTION

! CHECK-LABEL:neqv18_test
LOGICAL(4) FUNCTION neqv18_test(x0, x1)
LOGICAL(1) :: x0
LOGICAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = cmpi "ne", [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<4>
neqv18_test = x0 .NEQV. x1
END FUNCTION

! CHECK-LABEL:neqv19_test
LOGICAL(8) FUNCTION neqv19_test(x0, x1)
LOGICAL(1) :: x0
LOGICAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = cmpi "ne", [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<8>
neqv19_test = x0 .NEQV. x1
END FUNCTION

! CHECK-LABEL:neqv20_test
LOGICAL(2) FUNCTION neqv20_test(x0, x1)
LOGICAL(2) :: x0
LOGICAL(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = cmpi "ne", [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<2>
neqv20_test = x0 .NEQV. x1
END FUNCTION

! CHECK-LABEL:neqv21_test
LOGICAL(2) FUNCTION neqv21_test(x0, x1)
LOGICAL(2) :: x0
LOGICAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = cmpi "ne", [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<2>
neqv21_test = x0 .NEQV. x1
END FUNCTION

! CHECK-LABEL:neqv22_test
LOGICAL(4) FUNCTION neqv22_test(x0, x1)
LOGICAL(2) :: x0
LOGICAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = cmpi "ne", [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<4>
neqv22_test = x0 .NEQV. x1
END FUNCTION

! CHECK-LABEL:neqv23_test
LOGICAL(8) FUNCTION neqv23_test(x0, x1)
LOGICAL(2) :: x0
LOGICAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = cmpi "ne", [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<8>
neqv23_test = x0 .NEQV. x1
END FUNCTION

! CHECK-LABEL:neqv24_test
LOGICAL(4) FUNCTION neqv24_test(x0, x1)
LOGICAL(4) :: x0
LOGICAL(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = cmpi "ne", [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<4>
neqv24_test = x0 .NEQV. x1
END FUNCTION

! CHECK-LABEL:neqv25_test
LOGICAL(4) FUNCTION neqv25_test(x0, x1)
LOGICAL(4) :: x0
LOGICAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = cmpi "ne", [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<4>
neqv25_test = x0 .NEQV. x1
END FUNCTION

! CHECK-LABEL:neqv26_test
LOGICAL(4) FUNCTION neqv26_test(x0, x1)
LOGICAL(4) :: x0
LOGICAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = cmpi "ne", [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<4>
neqv26_test = x0 .NEQV. x1
END FUNCTION

! CHECK-LABEL:neqv27_test
LOGICAL(8) FUNCTION neqv27_test(x0, x1)
LOGICAL(4) :: x0
LOGICAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = cmpi "ne", [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<8>
neqv27_test = x0 .NEQV. x1
END FUNCTION

! CHECK-LABEL:neqv28_test
LOGICAL(8) FUNCTION neqv28_test(x0, x1)
LOGICAL(8) :: x0
LOGICAL(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = cmpi "ne", [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<8>
neqv28_test = x0 .NEQV. x1
END FUNCTION

! CHECK-LABEL:neqv29_test
LOGICAL(8) FUNCTION neqv29_test(x0, x1)
LOGICAL(8) :: x0
LOGICAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = cmpi "ne", [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<8>
neqv29_test = x0 .NEQV. x1
END FUNCTION

! CHECK-LABEL:neqv30_test
LOGICAL(8) FUNCTION neqv30_test(x0, x1)
LOGICAL(8) :: x0
LOGICAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = cmpi "ne", [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<8>
neqv30_test = x0 .NEQV. x1
END FUNCTION

! CHECK-LABEL:neqv31_test
LOGICAL(8) FUNCTION neqv31_test(x0, x1)
LOGICAL(8) :: x0
LOGICAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = cmpi "ne", [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<8>
neqv31_test = x0 .NEQV. x1
END FUNCTION

! CHECK-LABEL:or32_test
LOGICAL(1) FUNCTION or32_test(x0, x1)
LOGICAL(1) :: x0
LOGICAL(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = or [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<1>
or32_test = x0 .OR. x1
END FUNCTION

! CHECK-LABEL:or33_test
LOGICAL(2) FUNCTION or33_test(x0, x1)
LOGICAL(1) :: x0
LOGICAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = or [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<2>
or33_test = x0 .OR. x1
END FUNCTION

! CHECK-LABEL:or34_test
LOGICAL(4) FUNCTION or34_test(x0, x1)
LOGICAL(1) :: x0
LOGICAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = or [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<4>
or34_test = x0 .OR. x1
END FUNCTION

! CHECK-LABEL:or35_test
LOGICAL(8) FUNCTION or35_test(x0, x1)
LOGICAL(1) :: x0
LOGICAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = or [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<8>
or35_test = x0 .OR. x1
END FUNCTION

! CHECK-LABEL:or36_test
LOGICAL(2) FUNCTION or36_test(x0, x1)
LOGICAL(2) :: x0
LOGICAL(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = or [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<2>
or36_test = x0 .OR. x1
END FUNCTION

! CHECK-LABEL:or37_test
LOGICAL(2) FUNCTION or37_test(x0, x1)
LOGICAL(2) :: x0
LOGICAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = or [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<2>
or37_test = x0 .OR. x1
END FUNCTION

! CHECK-LABEL:or38_test
LOGICAL(4) FUNCTION or38_test(x0, x1)
LOGICAL(2) :: x0
LOGICAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = or [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<4>
or38_test = x0 .OR. x1
END FUNCTION

! CHECK-LABEL:or39_test
LOGICAL(8) FUNCTION or39_test(x0, x1)
LOGICAL(2) :: x0
LOGICAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = or [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<8>
or39_test = x0 .OR. x1
END FUNCTION

! CHECK-LABEL:or40_test
LOGICAL(4) FUNCTION or40_test(x0, x1)
LOGICAL(4) :: x0
LOGICAL(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = or [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<4>
or40_test = x0 .OR. x1
END FUNCTION

! CHECK-LABEL:or41_test
LOGICAL(4) FUNCTION or41_test(x0, x1)
LOGICAL(4) :: x0
LOGICAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = or [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<4>
or41_test = x0 .OR. x1
END FUNCTION

! CHECK-LABEL:or42_test
LOGICAL(4) FUNCTION or42_test(x0, x1)
LOGICAL(4) :: x0
LOGICAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = or [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<4>
or42_test = x0 .OR. x1
END FUNCTION

! CHECK-LABEL:or43_test
LOGICAL(8) FUNCTION or43_test(x0, x1)
LOGICAL(4) :: x0
LOGICAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = or [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<8>
or43_test = x0 .OR. x1
END FUNCTION

! CHECK-LABEL:or44_test
LOGICAL(8) FUNCTION or44_test(x0, x1)
LOGICAL(8) :: x0
LOGICAL(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = or [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<8>
or44_test = x0 .OR. x1
END FUNCTION

! CHECK-LABEL:or45_test
LOGICAL(8) FUNCTION or45_test(x0, x1)
LOGICAL(8) :: x0
LOGICAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = or [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<8>
or45_test = x0 .OR. x1
END FUNCTION

! CHECK-LABEL:or46_test
LOGICAL(8) FUNCTION or46_test(x0, x1)
LOGICAL(8) :: x0
LOGICAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = or [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<8>
or46_test = x0 .OR. x1
END FUNCTION

! CHECK-LABEL:or47_test
LOGICAL(8) FUNCTION or47_test(x0, x1)
LOGICAL(8) :: x0
LOGICAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = or [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<8>
or47_test = x0 .OR. x1
END FUNCTION

! CHECK-LABEL:and48_test
LOGICAL(1) FUNCTION and48_test(x0, x1)
LOGICAL(1) :: x0
LOGICAL(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = and [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<1>
and48_test = x0 .AND. x1
END FUNCTION

! CHECK-LABEL:and49_test
LOGICAL(2) FUNCTION and49_test(x0, x1)
LOGICAL(1) :: x0
LOGICAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = and [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<2>
and49_test = x0 .AND. x1
END FUNCTION

! CHECK-LABEL:and50_test
LOGICAL(4) FUNCTION and50_test(x0, x1)
LOGICAL(1) :: x0
LOGICAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = and [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<4>
and50_test = x0 .AND. x1
END FUNCTION

! CHECK-LABEL:and51_test
LOGICAL(8) FUNCTION and51_test(x0, x1)
LOGICAL(1) :: x0
LOGICAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = and [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<8>
and51_test = x0 .AND. x1
END FUNCTION

! CHECK-LABEL:and52_test
LOGICAL(2) FUNCTION and52_test(x0, x1)
LOGICAL(2) :: x0
LOGICAL(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = and [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<2>
and52_test = x0 .AND. x1
END FUNCTION

! CHECK-LABEL:and53_test
LOGICAL(2) FUNCTION and53_test(x0, x1)
LOGICAL(2) :: x0
LOGICAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = and [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<2>
and53_test = x0 .AND. x1
END FUNCTION

! CHECK-LABEL:and54_test
LOGICAL(4) FUNCTION and54_test(x0, x1)
LOGICAL(2) :: x0
LOGICAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = and [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<4>
and54_test = x0 .AND. x1
END FUNCTION

! CHECK-LABEL:and55_test
LOGICAL(8) FUNCTION and55_test(x0, x1)
LOGICAL(2) :: x0
LOGICAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = and [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<8>
and55_test = x0 .AND. x1
END FUNCTION

! CHECK-LABEL:and56_test
LOGICAL(4) FUNCTION and56_test(x0, x1)
LOGICAL(4) :: x0
LOGICAL(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = and [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<4>
and56_test = x0 .AND. x1
END FUNCTION

! CHECK-LABEL:and57_test
LOGICAL(4) FUNCTION and57_test(x0, x1)
LOGICAL(4) :: x0
LOGICAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = and [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<4>
and57_test = x0 .AND. x1
END FUNCTION

! CHECK-LABEL:and58_test
LOGICAL(4) FUNCTION and58_test(x0, x1)
LOGICAL(4) :: x0
LOGICAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = and [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<4>
and58_test = x0 .AND. x1
END FUNCTION

! CHECK-LABEL:and59_test
LOGICAL(8) FUNCTION and59_test(x0, x1)
LOGICAL(4) :: x0
LOGICAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = and [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<8>
and59_test = x0 .AND. x1
END FUNCTION

! CHECK-LABEL:and60_test
LOGICAL(8) FUNCTION and60_test(x0, x1)
LOGICAL(8) :: x0
LOGICAL(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = and [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<8>
and60_test = x0 .AND. x1
END FUNCTION

! CHECK-LABEL:and61_test
LOGICAL(8) FUNCTION and61_test(x0, x1)
LOGICAL(8) :: x0
LOGICAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = and [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<8>
and61_test = x0 .AND. x1
END FUNCTION

! CHECK-LABEL:and62_test
LOGICAL(8) FUNCTION and62_test(x0, x1)
LOGICAL(8) :: x0
LOGICAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = and [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<8>
and62_test = x0 .AND. x1
END FUNCTION

! CHECK-LABEL:and63_test
LOGICAL(8) FUNCTION and63_test(x0, x1)
LOGICAL(8) :: x0
LOGICAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK-DAG:[[reg4:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i1
! CHECK:[[reg5:%[0-9]+]] = and [[reg3]], [[reg4]]
! CHECK:fir.convert [[reg5]] {{.*}} -> !fir.logical<8>
and63_test = x0 .AND. x1
END FUNCTION

! CHECK-LABEL:not64_test
LOGICAL(1) FUNCTION not64_test(x0)
LOGICAL(1) :: x0
! CHECK:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK:[[reg3:%[0-9]+]] = xor [[reg2]], %true
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<1>
not64_test = .NOT. x0
END FUNCTION

! CHECK-LABEL:not65_test
LOGICAL(2) FUNCTION not65_test(x0)
LOGICAL(2) :: x0
! CHECK:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK:[[reg3:%[0-9]+]] = xor [[reg2]], %true
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<2>
not65_test = .NOT. x0
END FUNCTION

! CHECK-LABEL:not66_test
LOGICAL(4) FUNCTION not66_test(x0)
LOGICAL(4) :: x0
! CHECK:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK:[[reg3:%[0-9]+]] = xor [[reg2]], %true
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
not66_test = .NOT. x0
END FUNCTION

! CHECK-LABEL:not67_test
LOGICAL(8) FUNCTION not67_test(x0)
LOGICAL(8) :: x0
! CHECK:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i1
! CHECK:[[reg3:%[0-9]+]] = xor [[reg2]], %true
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<8>
not67_test = .NOT. x0
END FUNCTION

! CHECK-LABEL:eq68_test
LOGICAL FUNCTION eq68_test(x0, x1)
INTEGER(1) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = cmpi "eq", [[reg1]], [[reg2]] : i8
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
eq68_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq69_test
LOGICAL FUNCTION eq69_test(x0, x1)
INTEGER(1) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "eq", [[reg2]], [[reg3]] : i16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq69_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq70_test
LOGICAL FUNCTION eq70_test(x0, x1)
INTEGER(1) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "eq", [[reg2]], [[reg3]] : i32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq70_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq71_test
LOGICAL FUNCTION eq71_test(x0, x1)
INTEGER(1) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "eq", [[reg2]], [[reg3]] : i64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq71_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq72_test
LOGICAL FUNCTION eq72_test(x0, x1)
INTEGER(1) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i128
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "eq", [[reg2]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq72_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq73_test
LOGICAL FUNCTION eq73_test(x0, x1)
INTEGER(1) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg2]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq73_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq74_test
LOGICAL FUNCTION eq74_test(x0, x1)
INTEGER(1) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg2]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq74_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq75_test
LOGICAL FUNCTION eq75_test(x0, x1)
INTEGER(1) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg2]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq75_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq76_test
LOGICAL FUNCTION eq76_test(x0, x1)
INTEGER(1) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq76_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq77_test
LOGICAL FUNCTION eq77_test(x0, x1)
INTEGER(1) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq77_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq78_test
LOGICAL FUNCTION eq78_test(x0, x1)
INTEGER(2) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i16
! CHECK:[[reg4:%[0-9]+]] = cmpi "eq", [[reg1]], [[reg3]] : i16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq78_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq79_test
LOGICAL FUNCTION eq79_test(x0, x1)
INTEGER(2) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = cmpi "eq", [[reg1]], [[reg2]] : i16
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
eq79_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq80_test
LOGICAL FUNCTION eq80_test(x0, x1)
INTEGER(2) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "eq", [[reg2]], [[reg3]] : i32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq80_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq81_test
LOGICAL FUNCTION eq81_test(x0, x1)
INTEGER(2) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "eq", [[reg2]], [[reg3]] : i64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq81_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq82_test
LOGICAL FUNCTION eq82_test(x0, x1)
INTEGER(2) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i128
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "eq", [[reg2]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq82_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq83_test
LOGICAL FUNCTION eq83_test(x0, x1)
INTEGER(2) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg2]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq83_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq84_test
LOGICAL FUNCTION eq84_test(x0, x1)
INTEGER(2) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg2]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq84_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq85_test
LOGICAL FUNCTION eq85_test(x0, x1)
INTEGER(2) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg2]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq85_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq86_test
LOGICAL FUNCTION eq86_test(x0, x1)
INTEGER(2) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq86_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq87_test
LOGICAL FUNCTION eq87_test(x0, x1)
INTEGER(2) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq87_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq88_test
LOGICAL FUNCTION eq88_test(x0, x1)
INTEGER(4) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i32
! CHECK:[[reg4:%[0-9]+]] = cmpi "eq", [[reg1]], [[reg3]] : i32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq88_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq89_test
LOGICAL FUNCTION eq89_test(x0, x1)
INTEGER(4) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i32
! CHECK:[[reg4:%[0-9]+]] = cmpi "eq", [[reg1]], [[reg3]] : i32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq89_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq90_test
LOGICAL FUNCTION eq90_test(x0, x1)
INTEGER(4) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = cmpi "eq", [[reg1]], [[reg2]] : i32
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
eq90_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq91_test
LOGICAL FUNCTION eq91_test(x0, x1)
INTEGER(4) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "eq", [[reg2]], [[reg3]] : i64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq91_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq92_test
LOGICAL FUNCTION eq92_test(x0, x1)
INTEGER(4) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i128
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "eq", [[reg2]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq92_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq93_test
LOGICAL FUNCTION eq93_test(x0, x1)
INTEGER(4) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg2]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq93_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq94_test
LOGICAL FUNCTION eq94_test(x0, x1)
INTEGER(4) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg2]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq94_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq95_test
LOGICAL FUNCTION eq95_test(x0, x1)
INTEGER(4) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg2]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq95_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq96_test
LOGICAL FUNCTION eq96_test(x0, x1)
INTEGER(4) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq96_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq97_test
LOGICAL FUNCTION eq97_test(x0, x1)
INTEGER(4) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq97_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq98_test
LOGICAL FUNCTION eq98_test(x0, x1)
INTEGER(8) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i64
! CHECK:[[reg4:%[0-9]+]] = cmpi "eq", [[reg1]], [[reg3]] : i64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq98_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq99_test
LOGICAL FUNCTION eq99_test(x0, x1)
INTEGER(8) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i64
! CHECK:[[reg4:%[0-9]+]] = cmpi "eq", [[reg1]], [[reg3]] : i64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq99_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq100_test
LOGICAL FUNCTION eq100_test(x0, x1)
INTEGER(8) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i64
! CHECK:[[reg4:%[0-9]+]] = cmpi "eq", [[reg1]], [[reg3]] : i64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq100_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq101_test
LOGICAL FUNCTION eq101_test(x0, x1)
INTEGER(8) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = cmpi "eq", [[reg1]], [[reg2]] : i64
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
eq101_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq102_test
LOGICAL FUNCTION eq102_test(x0, x1)
INTEGER(8) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i128
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "eq", [[reg2]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq102_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq103_test
LOGICAL FUNCTION eq103_test(x0, x1)
INTEGER(8) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg2]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq103_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq104_test
LOGICAL FUNCTION eq104_test(x0, x1)
INTEGER(8) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg2]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq104_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq105_test
LOGICAL FUNCTION eq105_test(x0, x1)
INTEGER(8) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg2]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq105_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq106_test
LOGICAL FUNCTION eq106_test(x0, x1)
INTEGER(8) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq106_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq107_test
LOGICAL FUNCTION eq107_test(x0, x1)
INTEGER(8) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq107_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq108_test
LOGICAL FUNCTION eq108_test(x0, x1)
INTEGER(16) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i128
! CHECK:[[reg4:%[0-9]+]] = cmpi "eq", [[reg1]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq108_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq109_test
LOGICAL FUNCTION eq109_test(x0, x1)
INTEGER(16) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i128
! CHECK:[[reg4:%[0-9]+]] = cmpi "eq", [[reg1]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq109_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq110_test
LOGICAL FUNCTION eq110_test(x0, x1)
INTEGER(16) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i128
! CHECK:[[reg4:%[0-9]+]] = cmpi "eq", [[reg1]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq110_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq111_test
LOGICAL FUNCTION eq111_test(x0, x1)
INTEGER(16) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i128
! CHECK:[[reg4:%[0-9]+]] = cmpi "eq", [[reg1]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq111_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq112_test
LOGICAL FUNCTION eq112_test(x0, x1)
INTEGER(16) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = cmpi "eq", [[reg1]], [[reg2]] : i128
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
eq112_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq113_test
LOGICAL FUNCTION eq113_test(x0, x1)
INTEGER(16) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg2]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq113_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq114_test
LOGICAL FUNCTION eq114_test(x0, x1)
INTEGER(16) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg2]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq114_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq115_test
LOGICAL FUNCTION eq115_test(x0, x1)
INTEGER(16) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg2]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq115_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq116_test
LOGICAL FUNCTION eq116_test(x0, x1)
INTEGER(16) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq116_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq117_test
LOGICAL FUNCTION eq117_test(x0, x1)
INTEGER(16) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq117_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq118_test
LOGICAL FUNCTION eq118_test(x0, x1)
REAL(2) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg1]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq118_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq119_test
LOGICAL FUNCTION eq119_test(x0, x1)
REAL(2) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg1]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq119_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq120_test
LOGICAL FUNCTION eq120_test(x0, x1)
REAL(2) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg1]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq120_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq121_test
LOGICAL FUNCTION eq121_test(x0, x1)
REAL(2) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg1]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq121_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq122_test
LOGICAL FUNCTION eq122_test(x0, x1)
REAL(2) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg1]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq122_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq123_test
LOGICAL FUNCTION eq123_test(x0, x1)
REAL(2) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = fir.cmpf "oeq", [[reg1]], [[reg2]] : f16
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
eq123_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq124_test
LOGICAL FUNCTION eq124_test(x0, x1)
REAL(2) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg2]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq124_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq125_test
LOGICAL FUNCTION eq125_test(x0, x1)
REAL(2) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg2]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq125_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq126_test
LOGICAL FUNCTION eq126_test(x0, x1)
REAL(2) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq126_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq127_test
LOGICAL FUNCTION eq127_test(x0, x1)
REAL(2) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq127_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq128_test
LOGICAL FUNCTION eq128_test(x0, x1)
REAL(4) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg1]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq128_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq129_test
LOGICAL FUNCTION eq129_test(x0, x1)
REAL(4) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg1]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq129_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq130_test
LOGICAL FUNCTION eq130_test(x0, x1)
REAL(4) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg1]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq130_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq131_test
LOGICAL FUNCTION eq131_test(x0, x1)
REAL(4) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg1]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq131_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq132_test
LOGICAL FUNCTION eq132_test(x0, x1)
REAL(4) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg1]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq132_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq133_test
LOGICAL FUNCTION eq133_test(x0, x1)
REAL(4) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg1]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq133_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq134_test
LOGICAL FUNCTION eq134_test(x0, x1)
REAL(4) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = fir.cmpf "oeq", [[reg1]], [[reg2]] : f32
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
eq134_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq135_test
LOGICAL FUNCTION eq135_test(x0, x1)
REAL(4) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg2]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq135_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq136_test
LOGICAL FUNCTION eq136_test(x0, x1)
REAL(4) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq136_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq137_test
LOGICAL FUNCTION eq137_test(x0, x1)
REAL(4) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq137_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq138_test
LOGICAL FUNCTION eq138_test(x0, x1)
REAL(8) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg1]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq138_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq139_test
LOGICAL FUNCTION eq139_test(x0, x1)
REAL(8) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg1]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq139_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq140_test
LOGICAL FUNCTION eq140_test(x0, x1)
REAL(8) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg1]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq140_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq141_test
LOGICAL FUNCTION eq141_test(x0, x1)
REAL(8) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg1]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq141_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq142_test
LOGICAL FUNCTION eq142_test(x0, x1)
REAL(8) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg1]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq142_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq143_test
LOGICAL FUNCTION eq143_test(x0, x1)
REAL(8) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg1]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq143_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq144_test
LOGICAL FUNCTION eq144_test(x0, x1)
REAL(8) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg1]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq144_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq145_test
LOGICAL FUNCTION eq145_test(x0, x1)
REAL(8) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = fir.cmpf "oeq", [[reg1]], [[reg2]] : f64
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
eq145_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq146_test
LOGICAL FUNCTION eq146_test(x0, x1)
REAL(8) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq146_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq147_test
LOGICAL FUNCTION eq147_test(x0, x1)
REAL(8) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq147_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq148_test
LOGICAL FUNCTION eq148_test(x0, x1)
REAL(10) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq148_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq149_test
LOGICAL FUNCTION eq149_test(x0, x1)
REAL(10) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq149_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq150_test
LOGICAL FUNCTION eq150_test(x0, x1)
REAL(10) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq150_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq151_test
LOGICAL FUNCTION eq151_test(x0, x1)
REAL(10) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq151_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq152_test
LOGICAL FUNCTION eq152_test(x0, x1)
REAL(10) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq152_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq153_test
LOGICAL FUNCTION eq153_test(x0, x1)
REAL(10) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq153_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq154_test
LOGICAL FUNCTION eq154_test(x0, x1)
REAL(10) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq154_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq155_test
LOGICAL FUNCTION eq155_test(x0, x1)
REAL(10) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq155_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq156_test
LOGICAL FUNCTION eq156_test(x0, x1)
REAL(10) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = fir.cmpf "oeq", [[reg1]], [[reg2]] : !fir.real<10>
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
eq156_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq157_test
LOGICAL FUNCTION eq157_test(x0, x1)
REAL(10) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq157_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq158_test
LOGICAL FUNCTION eq158_test(x0, x1)
REAL(16) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq158_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq159_test
LOGICAL FUNCTION eq159_test(x0, x1)
REAL(16) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq159_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq160_test
LOGICAL FUNCTION eq160_test(x0, x1)
REAL(16) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq160_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq161_test
LOGICAL FUNCTION eq161_test(x0, x1)
REAL(16) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq161_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq162_test
LOGICAL FUNCTION eq162_test(x0, x1)
REAL(16) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq162_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq163_test
LOGICAL FUNCTION eq163_test(x0, x1)
REAL(16) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq163_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq164_test
LOGICAL FUNCTION eq164_test(x0, x1)
REAL(16) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq164_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq165_test
LOGICAL FUNCTION eq165_test(x0, x1)
REAL(16) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq165_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq166_test
LOGICAL FUNCTION eq166_test(x0, x1)
REAL(16) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oeq", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
eq166_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:eq167_test
LOGICAL FUNCTION eq167_test(x0, x1)
REAL(16) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = fir.cmpf "oeq", [[reg1]], [[reg2]] : !fir.real<16>
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
eq167_test = x0 .EQ. x1
END FUNCTION

! CHECK-LABEL:ne168_test
LOGICAL FUNCTION ne168_test(x0, x1)
INTEGER(1) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = cmpi "ne", [[reg1]], [[reg2]] : i8
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
ne168_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne169_test
LOGICAL FUNCTION ne169_test(x0, x1)
INTEGER(1) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "ne", [[reg2]], [[reg3]] : i16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne169_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne170_test
LOGICAL FUNCTION ne170_test(x0, x1)
INTEGER(1) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "ne", [[reg2]], [[reg3]] : i32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne170_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne171_test
LOGICAL FUNCTION ne171_test(x0, x1)
INTEGER(1) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "ne", [[reg2]], [[reg3]] : i64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne171_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne172_test
LOGICAL FUNCTION ne172_test(x0, x1)
INTEGER(1) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i128
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "ne", [[reg2]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne172_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne173_test
LOGICAL FUNCTION ne173_test(x0, x1)
INTEGER(1) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg2]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne173_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne174_test
LOGICAL FUNCTION ne174_test(x0, x1)
INTEGER(1) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg2]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne174_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne175_test
LOGICAL FUNCTION ne175_test(x0, x1)
INTEGER(1) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg2]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne175_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne176_test
LOGICAL FUNCTION ne176_test(x0, x1)
INTEGER(1) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne176_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne177_test
LOGICAL FUNCTION ne177_test(x0, x1)
INTEGER(1) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne177_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne178_test
LOGICAL FUNCTION ne178_test(x0, x1)
INTEGER(2) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i16
! CHECK:[[reg4:%[0-9]+]] = cmpi "ne", [[reg1]], [[reg3]] : i16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne178_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne179_test
LOGICAL FUNCTION ne179_test(x0, x1)
INTEGER(2) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = cmpi "ne", [[reg1]], [[reg2]] : i16
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
ne179_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne180_test
LOGICAL FUNCTION ne180_test(x0, x1)
INTEGER(2) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "ne", [[reg2]], [[reg3]] : i32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne180_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne181_test
LOGICAL FUNCTION ne181_test(x0, x1)
INTEGER(2) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "ne", [[reg2]], [[reg3]] : i64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne181_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne182_test
LOGICAL FUNCTION ne182_test(x0, x1)
INTEGER(2) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i128
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "ne", [[reg2]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne182_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne183_test
LOGICAL FUNCTION ne183_test(x0, x1)
INTEGER(2) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg2]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne183_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne184_test
LOGICAL FUNCTION ne184_test(x0, x1)
INTEGER(2) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg2]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne184_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne185_test
LOGICAL FUNCTION ne185_test(x0, x1)
INTEGER(2) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg2]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne185_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne186_test
LOGICAL FUNCTION ne186_test(x0, x1)
INTEGER(2) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne186_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne187_test
LOGICAL FUNCTION ne187_test(x0, x1)
INTEGER(2) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne187_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne188_test
LOGICAL FUNCTION ne188_test(x0, x1)
INTEGER(4) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i32
! CHECK:[[reg4:%[0-9]+]] = cmpi "ne", [[reg1]], [[reg3]] : i32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne188_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne189_test
LOGICAL FUNCTION ne189_test(x0, x1)
INTEGER(4) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i32
! CHECK:[[reg4:%[0-9]+]] = cmpi "ne", [[reg1]], [[reg3]] : i32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne189_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne190_test
LOGICAL FUNCTION ne190_test(x0, x1)
INTEGER(4) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = cmpi "ne", [[reg1]], [[reg2]] : i32
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
ne190_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne191_test
LOGICAL FUNCTION ne191_test(x0, x1)
INTEGER(4) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "ne", [[reg2]], [[reg3]] : i64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne191_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne192_test
LOGICAL FUNCTION ne192_test(x0, x1)
INTEGER(4) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i128
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "ne", [[reg2]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne192_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne193_test
LOGICAL FUNCTION ne193_test(x0, x1)
INTEGER(4) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg2]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne193_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne194_test
LOGICAL FUNCTION ne194_test(x0, x1)
INTEGER(4) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg2]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne194_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne195_test
LOGICAL FUNCTION ne195_test(x0, x1)
INTEGER(4) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg2]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne195_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne196_test
LOGICAL FUNCTION ne196_test(x0, x1)
INTEGER(4) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne196_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne197_test
LOGICAL FUNCTION ne197_test(x0, x1)
INTEGER(4) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne197_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne198_test
LOGICAL FUNCTION ne198_test(x0, x1)
INTEGER(8) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i64
! CHECK:[[reg4:%[0-9]+]] = cmpi "ne", [[reg1]], [[reg3]] : i64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne198_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne199_test
LOGICAL FUNCTION ne199_test(x0, x1)
INTEGER(8) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i64
! CHECK:[[reg4:%[0-9]+]] = cmpi "ne", [[reg1]], [[reg3]] : i64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne199_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne200_test
LOGICAL FUNCTION ne200_test(x0, x1)
INTEGER(8) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i64
! CHECK:[[reg4:%[0-9]+]] = cmpi "ne", [[reg1]], [[reg3]] : i64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne200_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne201_test
LOGICAL FUNCTION ne201_test(x0, x1)
INTEGER(8) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = cmpi "ne", [[reg1]], [[reg2]] : i64
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
ne201_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne202_test
LOGICAL FUNCTION ne202_test(x0, x1)
INTEGER(8) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i128
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "ne", [[reg2]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne202_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne203_test
LOGICAL FUNCTION ne203_test(x0, x1)
INTEGER(8) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg2]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne203_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne204_test
LOGICAL FUNCTION ne204_test(x0, x1)
INTEGER(8) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg2]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne204_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne205_test
LOGICAL FUNCTION ne205_test(x0, x1)
INTEGER(8) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg2]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne205_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne206_test
LOGICAL FUNCTION ne206_test(x0, x1)
INTEGER(8) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne206_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne207_test
LOGICAL FUNCTION ne207_test(x0, x1)
INTEGER(8) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne207_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne208_test
LOGICAL FUNCTION ne208_test(x0, x1)
INTEGER(16) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i128
! CHECK:[[reg4:%[0-9]+]] = cmpi "ne", [[reg1]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne208_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne209_test
LOGICAL FUNCTION ne209_test(x0, x1)
INTEGER(16) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i128
! CHECK:[[reg4:%[0-9]+]] = cmpi "ne", [[reg1]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne209_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne210_test
LOGICAL FUNCTION ne210_test(x0, x1)
INTEGER(16) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i128
! CHECK:[[reg4:%[0-9]+]] = cmpi "ne", [[reg1]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne210_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne211_test
LOGICAL FUNCTION ne211_test(x0, x1)
INTEGER(16) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i128
! CHECK:[[reg4:%[0-9]+]] = cmpi "ne", [[reg1]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne211_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne212_test
LOGICAL FUNCTION ne212_test(x0, x1)
INTEGER(16) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = cmpi "ne", [[reg1]], [[reg2]] : i128
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
ne212_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne213_test
LOGICAL FUNCTION ne213_test(x0, x1)
INTEGER(16) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg2]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne213_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne214_test
LOGICAL FUNCTION ne214_test(x0, x1)
INTEGER(16) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg2]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne214_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne215_test
LOGICAL FUNCTION ne215_test(x0, x1)
INTEGER(16) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg2]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne215_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne216_test
LOGICAL FUNCTION ne216_test(x0, x1)
INTEGER(16) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne216_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne217_test
LOGICAL FUNCTION ne217_test(x0, x1)
INTEGER(16) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne217_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne218_test
LOGICAL FUNCTION ne218_test(x0, x1)
REAL(2) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg1]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne218_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne219_test
LOGICAL FUNCTION ne219_test(x0, x1)
REAL(2) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg1]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne219_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne220_test
LOGICAL FUNCTION ne220_test(x0, x1)
REAL(2) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg1]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne220_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne221_test
LOGICAL FUNCTION ne221_test(x0, x1)
REAL(2) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg1]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne221_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne222_test
LOGICAL FUNCTION ne222_test(x0, x1)
REAL(2) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg1]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne222_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne223_test
LOGICAL FUNCTION ne223_test(x0, x1)
REAL(2) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = fir.cmpf "une", [[reg1]], [[reg2]] : f16
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
ne223_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne224_test
LOGICAL FUNCTION ne224_test(x0, x1)
REAL(2) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg2]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne224_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne225_test
LOGICAL FUNCTION ne225_test(x0, x1)
REAL(2) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg2]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne225_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne226_test
LOGICAL FUNCTION ne226_test(x0, x1)
REAL(2) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne226_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne227_test
LOGICAL FUNCTION ne227_test(x0, x1)
REAL(2) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne227_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne228_test
LOGICAL FUNCTION ne228_test(x0, x1)
REAL(4) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg1]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne228_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne229_test
LOGICAL FUNCTION ne229_test(x0, x1)
REAL(4) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg1]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne229_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne230_test
LOGICAL FUNCTION ne230_test(x0, x1)
REAL(4) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg1]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne230_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne231_test
LOGICAL FUNCTION ne231_test(x0, x1)
REAL(4) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg1]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne231_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne232_test
LOGICAL FUNCTION ne232_test(x0, x1)
REAL(4) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg1]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne232_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne233_test
LOGICAL FUNCTION ne233_test(x0, x1)
REAL(4) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg1]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne233_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne234_test
LOGICAL FUNCTION ne234_test(x0, x1)
REAL(4) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = fir.cmpf "une", [[reg1]], [[reg2]] : f32
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
ne234_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne235_test
LOGICAL FUNCTION ne235_test(x0, x1)
REAL(4) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg2]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne235_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne236_test
LOGICAL FUNCTION ne236_test(x0, x1)
REAL(4) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne236_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne237_test
LOGICAL FUNCTION ne237_test(x0, x1)
REAL(4) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne237_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne238_test
LOGICAL FUNCTION ne238_test(x0, x1)
REAL(8) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg1]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne238_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne239_test
LOGICAL FUNCTION ne239_test(x0, x1)
REAL(8) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg1]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne239_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne240_test
LOGICAL FUNCTION ne240_test(x0, x1)
REAL(8) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg1]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne240_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne241_test
LOGICAL FUNCTION ne241_test(x0, x1)
REAL(8) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg1]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne241_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne242_test
LOGICAL FUNCTION ne242_test(x0, x1)
REAL(8) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg1]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne242_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne243_test
LOGICAL FUNCTION ne243_test(x0, x1)
REAL(8) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg1]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne243_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne244_test
LOGICAL FUNCTION ne244_test(x0, x1)
REAL(8) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg1]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne244_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne245_test
LOGICAL FUNCTION ne245_test(x0, x1)
REAL(8) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = fir.cmpf "une", [[reg1]], [[reg2]] : f64
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
ne245_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne246_test
LOGICAL FUNCTION ne246_test(x0, x1)
REAL(8) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne246_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne247_test
LOGICAL FUNCTION ne247_test(x0, x1)
REAL(8) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne247_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne248_test
LOGICAL FUNCTION ne248_test(x0, x1)
REAL(10) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne248_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne249_test
LOGICAL FUNCTION ne249_test(x0, x1)
REAL(10) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne249_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne250_test
LOGICAL FUNCTION ne250_test(x0, x1)
REAL(10) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne250_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne251_test
LOGICAL FUNCTION ne251_test(x0, x1)
REAL(10) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne251_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne252_test
LOGICAL FUNCTION ne252_test(x0, x1)
REAL(10) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne252_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne253_test
LOGICAL FUNCTION ne253_test(x0, x1)
REAL(10) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne253_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne254_test
LOGICAL FUNCTION ne254_test(x0, x1)
REAL(10) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne254_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne255_test
LOGICAL FUNCTION ne255_test(x0, x1)
REAL(10) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne255_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne256_test
LOGICAL FUNCTION ne256_test(x0, x1)
REAL(10) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = fir.cmpf "une", [[reg1]], [[reg2]] : !fir.real<10>
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
ne256_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne257_test
LOGICAL FUNCTION ne257_test(x0, x1)
REAL(10) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne257_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne258_test
LOGICAL FUNCTION ne258_test(x0, x1)
REAL(16) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne258_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne259_test
LOGICAL FUNCTION ne259_test(x0, x1)
REAL(16) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne259_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne260_test
LOGICAL FUNCTION ne260_test(x0, x1)
REAL(16) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne260_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne261_test
LOGICAL FUNCTION ne261_test(x0, x1)
REAL(16) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne261_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne262_test
LOGICAL FUNCTION ne262_test(x0, x1)
REAL(16) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne262_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne263_test
LOGICAL FUNCTION ne263_test(x0, x1)
REAL(16) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne263_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne264_test
LOGICAL FUNCTION ne264_test(x0, x1)
REAL(16) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne264_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne265_test
LOGICAL FUNCTION ne265_test(x0, x1)
REAL(16) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne265_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne266_test
LOGICAL FUNCTION ne266_test(x0, x1)
REAL(16) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "une", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ne266_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:ne267_test
LOGICAL FUNCTION ne267_test(x0, x1)
REAL(16) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = fir.cmpf "une", [[reg1]], [[reg2]] : !fir.real<16>
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
ne267_test = x0 .NE. x1
END FUNCTION

! CHECK-LABEL:lt268_test
LOGICAL FUNCTION lt268_test(x0, x1)
INTEGER(1) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = cmpi "slt", [[reg1]], [[reg2]] : i8
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
lt268_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt269_test
LOGICAL FUNCTION lt269_test(x0, x1)
INTEGER(1) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "slt", [[reg2]], [[reg3]] : i16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt269_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt270_test
LOGICAL FUNCTION lt270_test(x0, x1)
INTEGER(1) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "slt", [[reg2]], [[reg3]] : i32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt270_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt271_test
LOGICAL FUNCTION lt271_test(x0, x1)
INTEGER(1) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "slt", [[reg2]], [[reg3]] : i64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt271_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt272_test
LOGICAL FUNCTION lt272_test(x0, x1)
INTEGER(1) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i128
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "slt", [[reg2]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt272_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt273_test
LOGICAL FUNCTION lt273_test(x0, x1)
INTEGER(1) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg2]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt273_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt274_test
LOGICAL FUNCTION lt274_test(x0, x1)
INTEGER(1) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg2]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt274_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt275_test
LOGICAL FUNCTION lt275_test(x0, x1)
INTEGER(1) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg2]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt275_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt276_test
LOGICAL FUNCTION lt276_test(x0, x1)
INTEGER(1) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt276_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt277_test
LOGICAL FUNCTION lt277_test(x0, x1)
INTEGER(1) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt277_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt278_test
LOGICAL FUNCTION lt278_test(x0, x1)
INTEGER(2) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i16
! CHECK:[[reg4:%[0-9]+]] = cmpi "slt", [[reg1]], [[reg3]] : i16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt278_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt279_test
LOGICAL FUNCTION lt279_test(x0, x1)
INTEGER(2) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = cmpi "slt", [[reg1]], [[reg2]] : i16
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
lt279_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt280_test
LOGICAL FUNCTION lt280_test(x0, x1)
INTEGER(2) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "slt", [[reg2]], [[reg3]] : i32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt280_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt281_test
LOGICAL FUNCTION lt281_test(x0, x1)
INTEGER(2) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "slt", [[reg2]], [[reg3]] : i64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt281_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt282_test
LOGICAL FUNCTION lt282_test(x0, x1)
INTEGER(2) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i128
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "slt", [[reg2]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt282_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt283_test
LOGICAL FUNCTION lt283_test(x0, x1)
INTEGER(2) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg2]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt283_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt284_test
LOGICAL FUNCTION lt284_test(x0, x1)
INTEGER(2) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg2]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt284_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt285_test
LOGICAL FUNCTION lt285_test(x0, x1)
INTEGER(2) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg2]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt285_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt286_test
LOGICAL FUNCTION lt286_test(x0, x1)
INTEGER(2) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt286_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt287_test
LOGICAL FUNCTION lt287_test(x0, x1)
INTEGER(2) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt287_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt288_test
LOGICAL FUNCTION lt288_test(x0, x1)
INTEGER(4) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i32
! CHECK:[[reg4:%[0-9]+]] = cmpi "slt", [[reg1]], [[reg3]] : i32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt288_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt289_test
LOGICAL FUNCTION lt289_test(x0, x1)
INTEGER(4) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i32
! CHECK:[[reg4:%[0-9]+]] = cmpi "slt", [[reg1]], [[reg3]] : i32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt289_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt290_test
LOGICAL FUNCTION lt290_test(x0, x1)
INTEGER(4) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = cmpi "slt", [[reg1]], [[reg2]] : i32
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
lt290_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt291_test
LOGICAL FUNCTION lt291_test(x0, x1)
INTEGER(4) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "slt", [[reg2]], [[reg3]] : i64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt291_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt292_test
LOGICAL FUNCTION lt292_test(x0, x1)
INTEGER(4) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i128
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "slt", [[reg2]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt292_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt293_test
LOGICAL FUNCTION lt293_test(x0, x1)
INTEGER(4) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg2]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt293_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt294_test
LOGICAL FUNCTION lt294_test(x0, x1)
INTEGER(4) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg2]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt294_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt295_test
LOGICAL FUNCTION lt295_test(x0, x1)
INTEGER(4) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg2]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt295_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt296_test
LOGICAL FUNCTION lt296_test(x0, x1)
INTEGER(4) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt296_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt297_test
LOGICAL FUNCTION lt297_test(x0, x1)
INTEGER(4) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt297_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt298_test
LOGICAL FUNCTION lt298_test(x0, x1)
INTEGER(8) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i64
! CHECK:[[reg4:%[0-9]+]] = cmpi "slt", [[reg1]], [[reg3]] : i64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt298_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt299_test
LOGICAL FUNCTION lt299_test(x0, x1)
INTEGER(8) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i64
! CHECK:[[reg4:%[0-9]+]] = cmpi "slt", [[reg1]], [[reg3]] : i64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt299_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt300_test
LOGICAL FUNCTION lt300_test(x0, x1)
INTEGER(8) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i64
! CHECK:[[reg4:%[0-9]+]] = cmpi "slt", [[reg1]], [[reg3]] : i64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt300_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt301_test
LOGICAL FUNCTION lt301_test(x0, x1)
INTEGER(8) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = cmpi "slt", [[reg1]], [[reg2]] : i64
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
lt301_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt302_test
LOGICAL FUNCTION lt302_test(x0, x1)
INTEGER(8) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i128
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "slt", [[reg2]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt302_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt303_test
LOGICAL FUNCTION lt303_test(x0, x1)
INTEGER(8) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg2]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt303_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt304_test
LOGICAL FUNCTION lt304_test(x0, x1)
INTEGER(8) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg2]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt304_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt305_test
LOGICAL FUNCTION lt305_test(x0, x1)
INTEGER(8) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg2]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt305_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt306_test
LOGICAL FUNCTION lt306_test(x0, x1)
INTEGER(8) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt306_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt307_test
LOGICAL FUNCTION lt307_test(x0, x1)
INTEGER(8) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt307_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt308_test
LOGICAL FUNCTION lt308_test(x0, x1)
INTEGER(16) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i128
! CHECK:[[reg4:%[0-9]+]] = cmpi "slt", [[reg1]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt308_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt309_test
LOGICAL FUNCTION lt309_test(x0, x1)
INTEGER(16) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i128
! CHECK:[[reg4:%[0-9]+]] = cmpi "slt", [[reg1]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt309_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt310_test
LOGICAL FUNCTION lt310_test(x0, x1)
INTEGER(16) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i128
! CHECK:[[reg4:%[0-9]+]] = cmpi "slt", [[reg1]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt310_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt311_test
LOGICAL FUNCTION lt311_test(x0, x1)
INTEGER(16) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i128
! CHECK:[[reg4:%[0-9]+]] = cmpi "slt", [[reg1]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt311_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt312_test
LOGICAL FUNCTION lt312_test(x0, x1)
INTEGER(16) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = cmpi "slt", [[reg1]], [[reg2]] : i128
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
lt312_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt313_test
LOGICAL FUNCTION lt313_test(x0, x1)
INTEGER(16) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg2]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt313_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt314_test
LOGICAL FUNCTION lt314_test(x0, x1)
INTEGER(16) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg2]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt314_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt315_test
LOGICAL FUNCTION lt315_test(x0, x1)
INTEGER(16) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg2]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt315_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt316_test
LOGICAL FUNCTION lt316_test(x0, x1)
INTEGER(16) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt316_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt317_test
LOGICAL FUNCTION lt317_test(x0, x1)
INTEGER(16) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt317_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt318_test
LOGICAL FUNCTION lt318_test(x0, x1)
REAL(2) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg1]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt318_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt319_test
LOGICAL FUNCTION lt319_test(x0, x1)
REAL(2) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg1]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt319_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt320_test
LOGICAL FUNCTION lt320_test(x0, x1)
REAL(2) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg1]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt320_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt321_test
LOGICAL FUNCTION lt321_test(x0, x1)
REAL(2) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg1]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt321_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt322_test
LOGICAL FUNCTION lt322_test(x0, x1)
REAL(2) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg1]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt322_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt323_test
LOGICAL FUNCTION lt323_test(x0, x1)
REAL(2) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = fir.cmpf "olt", [[reg1]], [[reg2]] : f16
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
lt323_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt324_test
LOGICAL FUNCTION lt324_test(x0, x1)
REAL(2) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg2]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt324_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt325_test
LOGICAL FUNCTION lt325_test(x0, x1)
REAL(2) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg2]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt325_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt326_test
LOGICAL FUNCTION lt326_test(x0, x1)
REAL(2) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt326_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt327_test
LOGICAL FUNCTION lt327_test(x0, x1)
REAL(2) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt327_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt328_test
LOGICAL FUNCTION lt328_test(x0, x1)
REAL(4) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg1]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt328_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt329_test
LOGICAL FUNCTION lt329_test(x0, x1)
REAL(4) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg1]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt329_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt330_test
LOGICAL FUNCTION lt330_test(x0, x1)
REAL(4) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg1]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt330_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt331_test
LOGICAL FUNCTION lt331_test(x0, x1)
REAL(4) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg1]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt331_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt332_test
LOGICAL FUNCTION lt332_test(x0, x1)
REAL(4) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg1]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt332_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt333_test
LOGICAL FUNCTION lt333_test(x0, x1)
REAL(4) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg1]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt333_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt334_test
LOGICAL FUNCTION lt334_test(x0, x1)
REAL(4) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = fir.cmpf "olt", [[reg1]], [[reg2]] : f32
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
lt334_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt335_test
LOGICAL FUNCTION lt335_test(x0, x1)
REAL(4) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg2]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt335_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt336_test
LOGICAL FUNCTION lt336_test(x0, x1)
REAL(4) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt336_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt337_test
LOGICAL FUNCTION lt337_test(x0, x1)
REAL(4) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt337_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt338_test
LOGICAL FUNCTION lt338_test(x0, x1)
REAL(8) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg1]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt338_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt339_test
LOGICAL FUNCTION lt339_test(x0, x1)
REAL(8) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg1]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt339_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt340_test
LOGICAL FUNCTION lt340_test(x0, x1)
REAL(8) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg1]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt340_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt341_test
LOGICAL FUNCTION lt341_test(x0, x1)
REAL(8) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg1]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt341_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt342_test
LOGICAL FUNCTION lt342_test(x0, x1)
REAL(8) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg1]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt342_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt343_test
LOGICAL FUNCTION lt343_test(x0, x1)
REAL(8) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg1]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt343_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt344_test
LOGICAL FUNCTION lt344_test(x0, x1)
REAL(8) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg1]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt344_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt345_test
LOGICAL FUNCTION lt345_test(x0, x1)
REAL(8) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = fir.cmpf "olt", [[reg1]], [[reg2]] : f64
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
lt345_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt346_test
LOGICAL FUNCTION lt346_test(x0, x1)
REAL(8) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt346_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt347_test
LOGICAL FUNCTION lt347_test(x0, x1)
REAL(8) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt347_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt348_test
LOGICAL FUNCTION lt348_test(x0, x1)
REAL(10) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt348_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt349_test
LOGICAL FUNCTION lt349_test(x0, x1)
REAL(10) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt349_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt350_test
LOGICAL FUNCTION lt350_test(x0, x1)
REAL(10) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt350_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt351_test
LOGICAL FUNCTION lt351_test(x0, x1)
REAL(10) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt351_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt352_test
LOGICAL FUNCTION lt352_test(x0, x1)
REAL(10) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt352_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt353_test
LOGICAL FUNCTION lt353_test(x0, x1)
REAL(10) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt353_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt354_test
LOGICAL FUNCTION lt354_test(x0, x1)
REAL(10) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt354_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt355_test
LOGICAL FUNCTION lt355_test(x0, x1)
REAL(10) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt355_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt356_test
LOGICAL FUNCTION lt356_test(x0, x1)
REAL(10) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = fir.cmpf "olt", [[reg1]], [[reg2]] : !fir.real<10>
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
lt356_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt357_test
LOGICAL FUNCTION lt357_test(x0, x1)
REAL(10) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt357_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt358_test
LOGICAL FUNCTION lt358_test(x0, x1)
REAL(16) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt358_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt359_test
LOGICAL FUNCTION lt359_test(x0, x1)
REAL(16) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt359_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt360_test
LOGICAL FUNCTION lt360_test(x0, x1)
REAL(16) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt360_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt361_test
LOGICAL FUNCTION lt361_test(x0, x1)
REAL(16) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt361_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt362_test
LOGICAL FUNCTION lt362_test(x0, x1)
REAL(16) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt362_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt363_test
LOGICAL FUNCTION lt363_test(x0, x1)
REAL(16) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt363_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt364_test
LOGICAL FUNCTION lt364_test(x0, x1)
REAL(16) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt364_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt365_test
LOGICAL FUNCTION lt365_test(x0, x1)
REAL(16) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt365_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt366_test
LOGICAL FUNCTION lt366_test(x0, x1)
REAL(16) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "olt", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
lt366_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:lt367_test
LOGICAL FUNCTION lt367_test(x0, x1)
REAL(16) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = fir.cmpf "olt", [[reg1]], [[reg2]] : !fir.real<16>
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
lt367_test = x0 .LT. x1
END FUNCTION

! CHECK-LABEL:le368_test
LOGICAL FUNCTION le368_test(x0, x1)
INTEGER(1) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = cmpi "sle", [[reg1]], [[reg2]] : i8
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
le368_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le369_test
LOGICAL FUNCTION le369_test(x0, x1)
INTEGER(1) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "sle", [[reg2]], [[reg3]] : i16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le369_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le370_test
LOGICAL FUNCTION le370_test(x0, x1)
INTEGER(1) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "sle", [[reg2]], [[reg3]] : i32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le370_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le371_test
LOGICAL FUNCTION le371_test(x0, x1)
INTEGER(1) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "sle", [[reg2]], [[reg3]] : i64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le371_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le372_test
LOGICAL FUNCTION le372_test(x0, x1)
INTEGER(1) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i128
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "sle", [[reg2]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le372_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le373_test
LOGICAL FUNCTION le373_test(x0, x1)
INTEGER(1) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg2]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le373_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le374_test
LOGICAL FUNCTION le374_test(x0, x1)
INTEGER(1) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg2]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le374_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le375_test
LOGICAL FUNCTION le375_test(x0, x1)
INTEGER(1) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg2]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le375_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le376_test
LOGICAL FUNCTION le376_test(x0, x1)
INTEGER(1) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le376_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le377_test
LOGICAL FUNCTION le377_test(x0, x1)
INTEGER(1) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le377_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le378_test
LOGICAL FUNCTION le378_test(x0, x1)
INTEGER(2) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i16
! CHECK:[[reg4:%[0-9]+]] = cmpi "sle", [[reg1]], [[reg3]] : i16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le378_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le379_test
LOGICAL FUNCTION le379_test(x0, x1)
INTEGER(2) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = cmpi "sle", [[reg1]], [[reg2]] : i16
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
le379_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le380_test
LOGICAL FUNCTION le380_test(x0, x1)
INTEGER(2) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "sle", [[reg2]], [[reg3]] : i32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le380_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le381_test
LOGICAL FUNCTION le381_test(x0, x1)
INTEGER(2) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "sle", [[reg2]], [[reg3]] : i64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le381_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le382_test
LOGICAL FUNCTION le382_test(x0, x1)
INTEGER(2) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i128
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "sle", [[reg2]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le382_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le383_test
LOGICAL FUNCTION le383_test(x0, x1)
INTEGER(2) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg2]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le383_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le384_test
LOGICAL FUNCTION le384_test(x0, x1)
INTEGER(2) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg2]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le384_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le385_test
LOGICAL FUNCTION le385_test(x0, x1)
INTEGER(2) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg2]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le385_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le386_test
LOGICAL FUNCTION le386_test(x0, x1)
INTEGER(2) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le386_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le387_test
LOGICAL FUNCTION le387_test(x0, x1)
INTEGER(2) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le387_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le388_test
LOGICAL FUNCTION le388_test(x0, x1)
INTEGER(4) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i32
! CHECK:[[reg4:%[0-9]+]] = cmpi "sle", [[reg1]], [[reg3]] : i32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le388_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le389_test
LOGICAL FUNCTION le389_test(x0, x1)
INTEGER(4) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i32
! CHECK:[[reg4:%[0-9]+]] = cmpi "sle", [[reg1]], [[reg3]] : i32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le389_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le390_test
LOGICAL FUNCTION le390_test(x0, x1)
INTEGER(4) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = cmpi "sle", [[reg1]], [[reg2]] : i32
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
le390_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le391_test
LOGICAL FUNCTION le391_test(x0, x1)
INTEGER(4) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "sle", [[reg2]], [[reg3]] : i64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le391_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le392_test
LOGICAL FUNCTION le392_test(x0, x1)
INTEGER(4) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i128
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "sle", [[reg2]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le392_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le393_test
LOGICAL FUNCTION le393_test(x0, x1)
INTEGER(4) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg2]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le393_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le394_test
LOGICAL FUNCTION le394_test(x0, x1)
INTEGER(4) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg2]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le394_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le395_test
LOGICAL FUNCTION le395_test(x0, x1)
INTEGER(4) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg2]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le395_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le396_test
LOGICAL FUNCTION le396_test(x0, x1)
INTEGER(4) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le396_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le397_test
LOGICAL FUNCTION le397_test(x0, x1)
INTEGER(4) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le397_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le398_test
LOGICAL FUNCTION le398_test(x0, x1)
INTEGER(8) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i64
! CHECK:[[reg4:%[0-9]+]] = cmpi "sle", [[reg1]], [[reg3]] : i64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le398_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le399_test
LOGICAL FUNCTION le399_test(x0, x1)
INTEGER(8) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i64
! CHECK:[[reg4:%[0-9]+]] = cmpi "sle", [[reg1]], [[reg3]] : i64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le399_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le400_test
LOGICAL FUNCTION le400_test(x0, x1)
INTEGER(8) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i64
! CHECK:[[reg4:%[0-9]+]] = cmpi "sle", [[reg1]], [[reg3]] : i64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le400_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le401_test
LOGICAL FUNCTION le401_test(x0, x1)
INTEGER(8) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = cmpi "sle", [[reg1]], [[reg2]] : i64
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
le401_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le402_test
LOGICAL FUNCTION le402_test(x0, x1)
INTEGER(8) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i128
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "sle", [[reg2]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le402_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le403_test
LOGICAL FUNCTION le403_test(x0, x1)
INTEGER(8) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg2]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le403_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le404_test
LOGICAL FUNCTION le404_test(x0, x1)
INTEGER(8) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg2]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le404_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le405_test
LOGICAL FUNCTION le405_test(x0, x1)
INTEGER(8) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg2]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le405_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le406_test
LOGICAL FUNCTION le406_test(x0, x1)
INTEGER(8) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le406_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le407_test
LOGICAL FUNCTION le407_test(x0, x1)
INTEGER(8) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le407_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le408_test
LOGICAL FUNCTION le408_test(x0, x1)
INTEGER(16) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i128
! CHECK:[[reg4:%[0-9]+]] = cmpi "sle", [[reg1]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le408_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le409_test
LOGICAL FUNCTION le409_test(x0, x1)
INTEGER(16) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i128
! CHECK:[[reg4:%[0-9]+]] = cmpi "sle", [[reg1]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le409_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le410_test
LOGICAL FUNCTION le410_test(x0, x1)
INTEGER(16) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i128
! CHECK:[[reg4:%[0-9]+]] = cmpi "sle", [[reg1]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le410_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le411_test
LOGICAL FUNCTION le411_test(x0, x1)
INTEGER(16) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i128
! CHECK:[[reg4:%[0-9]+]] = cmpi "sle", [[reg1]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le411_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le412_test
LOGICAL FUNCTION le412_test(x0, x1)
INTEGER(16) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = cmpi "sle", [[reg1]], [[reg2]] : i128
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
le412_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le413_test
LOGICAL FUNCTION le413_test(x0, x1)
INTEGER(16) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg2]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le413_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le414_test
LOGICAL FUNCTION le414_test(x0, x1)
INTEGER(16) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg2]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le414_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le415_test
LOGICAL FUNCTION le415_test(x0, x1)
INTEGER(16) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg2]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le415_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le416_test
LOGICAL FUNCTION le416_test(x0, x1)
INTEGER(16) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le416_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le417_test
LOGICAL FUNCTION le417_test(x0, x1)
INTEGER(16) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le417_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le418_test
LOGICAL FUNCTION le418_test(x0, x1)
REAL(2) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg1]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le418_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le419_test
LOGICAL FUNCTION le419_test(x0, x1)
REAL(2) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg1]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le419_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le420_test
LOGICAL FUNCTION le420_test(x0, x1)
REAL(2) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg1]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le420_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le421_test
LOGICAL FUNCTION le421_test(x0, x1)
REAL(2) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg1]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le421_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le422_test
LOGICAL FUNCTION le422_test(x0, x1)
REAL(2) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg1]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le422_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le423_test
LOGICAL FUNCTION le423_test(x0, x1)
REAL(2) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = fir.cmpf "ole", [[reg1]], [[reg2]] : f16
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
le423_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le424_test
LOGICAL FUNCTION le424_test(x0, x1)
REAL(2) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg2]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le424_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le425_test
LOGICAL FUNCTION le425_test(x0, x1)
REAL(2) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg2]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le425_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le426_test
LOGICAL FUNCTION le426_test(x0, x1)
REAL(2) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le426_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le427_test
LOGICAL FUNCTION le427_test(x0, x1)
REAL(2) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le427_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le428_test
LOGICAL FUNCTION le428_test(x0, x1)
REAL(4) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg1]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le428_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le429_test
LOGICAL FUNCTION le429_test(x0, x1)
REAL(4) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg1]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le429_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le430_test
LOGICAL FUNCTION le430_test(x0, x1)
REAL(4) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg1]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le430_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le431_test
LOGICAL FUNCTION le431_test(x0, x1)
REAL(4) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg1]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le431_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le432_test
LOGICAL FUNCTION le432_test(x0, x1)
REAL(4) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg1]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le432_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le433_test
LOGICAL FUNCTION le433_test(x0, x1)
REAL(4) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg1]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le433_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le434_test
LOGICAL FUNCTION le434_test(x0, x1)
REAL(4) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = fir.cmpf "ole", [[reg1]], [[reg2]] : f32
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
le434_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le435_test
LOGICAL FUNCTION le435_test(x0, x1)
REAL(4) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg2]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le435_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le436_test
LOGICAL FUNCTION le436_test(x0, x1)
REAL(4) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le436_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le437_test
LOGICAL FUNCTION le437_test(x0, x1)
REAL(4) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le437_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le438_test
LOGICAL FUNCTION le438_test(x0, x1)
REAL(8) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg1]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le438_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le439_test
LOGICAL FUNCTION le439_test(x0, x1)
REAL(8) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg1]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le439_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le440_test
LOGICAL FUNCTION le440_test(x0, x1)
REAL(8) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg1]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le440_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le441_test
LOGICAL FUNCTION le441_test(x0, x1)
REAL(8) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg1]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le441_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le442_test
LOGICAL FUNCTION le442_test(x0, x1)
REAL(8) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg1]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le442_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le443_test
LOGICAL FUNCTION le443_test(x0, x1)
REAL(8) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg1]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le443_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le444_test
LOGICAL FUNCTION le444_test(x0, x1)
REAL(8) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg1]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le444_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le445_test
LOGICAL FUNCTION le445_test(x0, x1)
REAL(8) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = fir.cmpf "ole", [[reg1]], [[reg2]] : f64
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
le445_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le446_test
LOGICAL FUNCTION le446_test(x0, x1)
REAL(8) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le446_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le447_test
LOGICAL FUNCTION le447_test(x0, x1)
REAL(8) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le447_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le448_test
LOGICAL FUNCTION le448_test(x0, x1)
REAL(10) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le448_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le449_test
LOGICAL FUNCTION le449_test(x0, x1)
REAL(10) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le449_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le450_test
LOGICAL FUNCTION le450_test(x0, x1)
REAL(10) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le450_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le451_test
LOGICAL FUNCTION le451_test(x0, x1)
REAL(10) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le451_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le452_test
LOGICAL FUNCTION le452_test(x0, x1)
REAL(10) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le452_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le453_test
LOGICAL FUNCTION le453_test(x0, x1)
REAL(10) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le453_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le454_test
LOGICAL FUNCTION le454_test(x0, x1)
REAL(10) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le454_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le455_test
LOGICAL FUNCTION le455_test(x0, x1)
REAL(10) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le455_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le456_test
LOGICAL FUNCTION le456_test(x0, x1)
REAL(10) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = fir.cmpf "ole", [[reg1]], [[reg2]] : !fir.real<10>
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
le456_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le457_test
LOGICAL FUNCTION le457_test(x0, x1)
REAL(10) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le457_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le458_test
LOGICAL FUNCTION le458_test(x0, x1)
REAL(16) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le458_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le459_test
LOGICAL FUNCTION le459_test(x0, x1)
REAL(16) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le459_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le460_test
LOGICAL FUNCTION le460_test(x0, x1)
REAL(16) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le460_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le461_test
LOGICAL FUNCTION le461_test(x0, x1)
REAL(16) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le461_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le462_test
LOGICAL FUNCTION le462_test(x0, x1)
REAL(16) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le462_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le463_test
LOGICAL FUNCTION le463_test(x0, x1)
REAL(16) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le463_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le464_test
LOGICAL FUNCTION le464_test(x0, x1)
REAL(16) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le464_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le465_test
LOGICAL FUNCTION le465_test(x0, x1)
REAL(16) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le465_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le466_test
LOGICAL FUNCTION le466_test(x0, x1)
REAL(16) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ole", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
le466_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:le467_test
LOGICAL FUNCTION le467_test(x0, x1)
REAL(16) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = fir.cmpf "ole", [[reg1]], [[reg2]] : !fir.real<16>
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
le467_test = x0 .LE. x1
END FUNCTION

! CHECK-LABEL:gt468_test
LOGICAL FUNCTION gt468_test(x0, x1)
INTEGER(1) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = cmpi "sgt", [[reg1]], [[reg2]] : i8
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
gt468_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt469_test
LOGICAL FUNCTION gt469_test(x0, x1)
INTEGER(1) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "sgt", [[reg2]], [[reg3]] : i16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt469_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt470_test
LOGICAL FUNCTION gt470_test(x0, x1)
INTEGER(1) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "sgt", [[reg2]], [[reg3]] : i32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt470_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt471_test
LOGICAL FUNCTION gt471_test(x0, x1)
INTEGER(1) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "sgt", [[reg2]], [[reg3]] : i64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt471_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt472_test
LOGICAL FUNCTION gt472_test(x0, x1)
INTEGER(1) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i128
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "sgt", [[reg2]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt472_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt473_test
LOGICAL FUNCTION gt473_test(x0, x1)
INTEGER(1) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg2]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt473_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt474_test
LOGICAL FUNCTION gt474_test(x0, x1)
INTEGER(1) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg2]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt474_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt475_test
LOGICAL FUNCTION gt475_test(x0, x1)
INTEGER(1) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg2]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt475_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt476_test
LOGICAL FUNCTION gt476_test(x0, x1)
INTEGER(1) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt476_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt477_test
LOGICAL FUNCTION gt477_test(x0, x1)
INTEGER(1) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt477_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt478_test
LOGICAL FUNCTION gt478_test(x0, x1)
INTEGER(2) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i16
! CHECK:[[reg4:%[0-9]+]] = cmpi "sgt", [[reg1]], [[reg3]] : i16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt478_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt479_test
LOGICAL FUNCTION gt479_test(x0, x1)
INTEGER(2) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = cmpi "sgt", [[reg1]], [[reg2]] : i16
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
gt479_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt480_test
LOGICAL FUNCTION gt480_test(x0, x1)
INTEGER(2) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "sgt", [[reg2]], [[reg3]] : i32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt480_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt481_test
LOGICAL FUNCTION gt481_test(x0, x1)
INTEGER(2) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "sgt", [[reg2]], [[reg3]] : i64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt481_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt482_test
LOGICAL FUNCTION gt482_test(x0, x1)
INTEGER(2) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i128
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "sgt", [[reg2]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt482_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt483_test
LOGICAL FUNCTION gt483_test(x0, x1)
INTEGER(2) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg2]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt483_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt484_test
LOGICAL FUNCTION gt484_test(x0, x1)
INTEGER(2) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg2]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt484_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt485_test
LOGICAL FUNCTION gt485_test(x0, x1)
INTEGER(2) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg2]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt485_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt486_test
LOGICAL FUNCTION gt486_test(x0, x1)
INTEGER(2) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt486_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt487_test
LOGICAL FUNCTION gt487_test(x0, x1)
INTEGER(2) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt487_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt488_test
LOGICAL FUNCTION gt488_test(x0, x1)
INTEGER(4) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i32
! CHECK:[[reg4:%[0-9]+]] = cmpi "sgt", [[reg1]], [[reg3]] : i32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt488_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt489_test
LOGICAL FUNCTION gt489_test(x0, x1)
INTEGER(4) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i32
! CHECK:[[reg4:%[0-9]+]] = cmpi "sgt", [[reg1]], [[reg3]] : i32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt489_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt490_test
LOGICAL FUNCTION gt490_test(x0, x1)
INTEGER(4) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = cmpi "sgt", [[reg1]], [[reg2]] : i32
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
gt490_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt491_test
LOGICAL FUNCTION gt491_test(x0, x1)
INTEGER(4) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "sgt", [[reg2]], [[reg3]] : i64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt491_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt492_test
LOGICAL FUNCTION gt492_test(x0, x1)
INTEGER(4) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i128
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "sgt", [[reg2]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt492_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt493_test
LOGICAL FUNCTION gt493_test(x0, x1)
INTEGER(4) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg2]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt493_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt494_test
LOGICAL FUNCTION gt494_test(x0, x1)
INTEGER(4) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg2]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt494_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt495_test
LOGICAL FUNCTION gt495_test(x0, x1)
INTEGER(4) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg2]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt495_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt496_test
LOGICAL FUNCTION gt496_test(x0, x1)
INTEGER(4) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt496_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt497_test
LOGICAL FUNCTION gt497_test(x0, x1)
INTEGER(4) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt497_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt498_test
LOGICAL FUNCTION gt498_test(x0, x1)
INTEGER(8) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i64
! CHECK:[[reg4:%[0-9]+]] = cmpi "sgt", [[reg1]], [[reg3]] : i64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt498_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt499_test
LOGICAL FUNCTION gt499_test(x0, x1)
INTEGER(8) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i64
! CHECK:[[reg4:%[0-9]+]] = cmpi "sgt", [[reg1]], [[reg3]] : i64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt499_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt500_test
LOGICAL FUNCTION gt500_test(x0, x1)
INTEGER(8) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i64
! CHECK:[[reg4:%[0-9]+]] = cmpi "sgt", [[reg1]], [[reg3]] : i64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt500_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt501_test
LOGICAL FUNCTION gt501_test(x0, x1)
INTEGER(8) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = cmpi "sgt", [[reg1]], [[reg2]] : i64
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
gt501_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt502_test
LOGICAL FUNCTION gt502_test(x0, x1)
INTEGER(8) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i128
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "sgt", [[reg2]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt502_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt503_test
LOGICAL FUNCTION gt503_test(x0, x1)
INTEGER(8) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg2]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt503_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt504_test
LOGICAL FUNCTION gt504_test(x0, x1)
INTEGER(8) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg2]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt504_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt505_test
LOGICAL FUNCTION gt505_test(x0, x1)
INTEGER(8) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg2]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt505_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt506_test
LOGICAL FUNCTION gt506_test(x0, x1)
INTEGER(8) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt506_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt507_test
LOGICAL FUNCTION gt507_test(x0, x1)
INTEGER(8) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt507_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt508_test
LOGICAL FUNCTION gt508_test(x0, x1)
INTEGER(16) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i128
! CHECK:[[reg4:%[0-9]+]] = cmpi "sgt", [[reg1]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt508_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt509_test
LOGICAL FUNCTION gt509_test(x0, x1)
INTEGER(16) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i128
! CHECK:[[reg4:%[0-9]+]] = cmpi "sgt", [[reg1]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt509_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt510_test
LOGICAL FUNCTION gt510_test(x0, x1)
INTEGER(16) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i128
! CHECK:[[reg4:%[0-9]+]] = cmpi "sgt", [[reg1]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt510_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt511_test
LOGICAL FUNCTION gt511_test(x0, x1)
INTEGER(16) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i128
! CHECK:[[reg4:%[0-9]+]] = cmpi "sgt", [[reg1]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt511_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt512_test
LOGICAL FUNCTION gt512_test(x0, x1)
INTEGER(16) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = cmpi "sgt", [[reg1]], [[reg2]] : i128
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
gt512_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt513_test
LOGICAL FUNCTION gt513_test(x0, x1)
INTEGER(16) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg2]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt513_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt514_test
LOGICAL FUNCTION gt514_test(x0, x1)
INTEGER(16) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg2]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt514_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt515_test
LOGICAL FUNCTION gt515_test(x0, x1)
INTEGER(16) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg2]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt515_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt516_test
LOGICAL FUNCTION gt516_test(x0, x1)
INTEGER(16) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt516_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt517_test
LOGICAL FUNCTION gt517_test(x0, x1)
INTEGER(16) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt517_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt518_test
LOGICAL FUNCTION gt518_test(x0, x1)
REAL(2) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg1]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt518_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt519_test
LOGICAL FUNCTION gt519_test(x0, x1)
REAL(2) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg1]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt519_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt520_test
LOGICAL FUNCTION gt520_test(x0, x1)
REAL(2) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg1]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt520_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt521_test
LOGICAL FUNCTION gt521_test(x0, x1)
REAL(2) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg1]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt521_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt522_test
LOGICAL FUNCTION gt522_test(x0, x1)
REAL(2) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg1]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt522_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt523_test
LOGICAL FUNCTION gt523_test(x0, x1)
REAL(2) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = fir.cmpf "ogt", [[reg1]], [[reg2]] : f16
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
gt523_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt524_test
LOGICAL FUNCTION gt524_test(x0, x1)
REAL(2) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg2]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt524_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt525_test
LOGICAL FUNCTION gt525_test(x0, x1)
REAL(2) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg2]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt525_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt526_test
LOGICAL FUNCTION gt526_test(x0, x1)
REAL(2) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt526_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt527_test
LOGICAL FUNCTION gt527_test(x0, x1)
REAL(2) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt527_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt528_test
LOGICAL FUNCTION gt528_test(x0, x1)
REAL(4) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg1]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt528_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt529_test
LOGICAL FUNCTION gt529_test(x0, x1)
REAL(4) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg1]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt529_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt530_test
LOGICAL FUNCTION gt530_test(x0, x1)
REAL(4) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg1]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt530_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt531_test
LOGICAL FUNCTION gt531_test(x0, x1)
REAL(4) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg1]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt531_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt532_test
LOGICAL FUNCTION gt532_test(x0, x1)
REAL(4) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg1]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt532_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt533_test
LOGICAL FUNCTION gt533_test(x0, x1)
REAL(4) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg1]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt533_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt534_test
LOGICAL FUNCTION gt534_test(x0, x1)
REAL(4) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = fir.cmpf "ogt", [[reg1]], [[reg2]] : f32
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
gt534_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt535_test
LOGICAL FUNCTION gt535_test(x0, x1)
REAL(4) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg2]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt535_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt536_test
LOGICAL FUNCTION gt536_test(x0, x1)
REAL(4) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt536_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt537_test
LOGICAL FUNCTION gt537_test(x0, x1)
REAL(4) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt537_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt538_test
LOGICAL FUNCTION gt538_test(x0, x1)
REAL(8) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg1]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt538_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt539_test
LOGICAL FUNCTION gt539_test(x0, x1)
REAL(8) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg1]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt539_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt540_test
LOGICAL FUNCTION gt540_test(x0, x1)
REAL(8) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg1]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt540_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt541_test
LOGICAL FUNCTION gt541_test(x0, x1)
REAL(8) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg1]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt541_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt542_test
LOGICAL FUNCTION gt542_test(x0, x1)
REAL(8) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg1]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt542_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt543_test
LOGICAL FUNCTION gt543_test(x0, x1)
REAL(8) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg1]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt543_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt544_test
LOGICAL FUNCTION gt544_test(x0, x1)
REAL(8) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg1]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt544_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt545_test
LOGICAL FUNCTION gt545_test(x0, x1)
REAL(8) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = fir.cmpf "ogt", [[reg1]], [[reg2]] : f64
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
gt545_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt546_test
LOGICAL FUNCTION gt546_test(x0, x1)
REAL(8) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt546_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt547_test
LOGICAL FUNCTION gt547_test(x0, x1)
REAL(8) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt547_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt548_test
LOGICAL FUNCTION gt548_test(x0, x1)
REAL(10) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt548_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt549_test
LOGICAL FUNCTION gt549_test(x0, x1)
REAL(10) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt549_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt550_test
LOGICAL FUNCTION gt550_test(x0, x1)
REAL(10) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt550_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt551_test
LOGICAL FUNCTION gt551_test(x0, x1)
REAL(10) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt551_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt552_test
LOGICAL FUNCTION gt552_test(x0, x1)
REAL(10) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt552_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt553_test
LOGICAL FUNCTION gt553_test(x0, x1)
REAL(10) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt553_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt554_test
LOGICAL FUNCTION gt554_test(x0, x1)
REAL(10) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt554_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt555_test
LOGICAL FUNCTION gt555_test(x0, x1)
REAL(10) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt555_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt556_test
LOGICAL FUNCTION gt556_test(x0, x1)
REAL(10) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = fir.cmpf "ogt", [[reg1]], [[reg2]] : !fir.real<10>
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
gt556_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt557_test
LOGICAL FUNCTION gt557_test(x0, x1)
REAL(10) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt557_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt558_test
LOGICAL FUNCTION gt558_test(x0, x1)
REAL(16) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt558_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt559_test
LOGICAL FUNCTION gt559_test(x0, x1)
REAL(16) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt559_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt560_test
LOGICAL FUNCTION gt560_test(x0, x1)
REAL(16) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt560_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt561_test
LOGICAL FUNCTION gt561_test(x0, x1)
REAL(16) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt561_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt562_test
LOGICAL FUNCTION gt562_test(x0, x1)
REAL(16) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt562_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt563_test
LOGICAL FUNCTION gt563_test(x0, x1)
REAL(16) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt563_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt564_test
LOGICAL FUNCTION gt564_test(x0, x1)
REAL(16) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt564_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt565_test
LOGICAL FUNCTION gt565_test(x0, x1)
REAL(16) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt565_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt566_test
LOGICAL FUNCTION gt566_test(x0, x1)
REAL(16) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "ogt", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
gt566_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:gt567_test
LOGICAL FUNCTION gt567_test(x0, x1)
REAL(16) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = fir.cmpf "ogt", [[reg1]], [[reg2]] : !fir.real<16>
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
gt567_test = x0 .GT. x1
END FUNCTION

! CHECK-LABEL:ge568_test
LOGICAL FUNCTION ge568_test(x0, x1)
INTEGER(1) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = cmpi "sge", [[reg1]], [[reg2]] : i8
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
ge568_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge569_test
LOGICAL FUNCTION ge569_test(x0, x1)
INTEGER(1) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "sge", [[reg2]], [[reg3]] : i16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge569_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge570_test
LOGICAL FUNCTION ge570_test(x0, x1)
INTEGER(1) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "sge", [[reg2]], [[reg3]] : i32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge570_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge571_test
LOGICAL FUNCTION ge571_test(x0, x1)
INTEGER(1) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "sge", [[reg2]], [[reg3]] : i64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge571_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge572_test
LOGICAL FUNCTION ge572_test(x0, x1)
INTEGER(1) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i128
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "sge", [[reg2]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge572_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge573_test
LOGICAL FUNCTION ge573_test(x0, x1)
INTEGER(1) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg2]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge573_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge574_test
LOGICAL FUNCTION ge574_test(x0, x1)
INTEGER(1) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg2]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge574_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge575_test
LOGICAL FUNCTION ge575_test(x0, x1)
INTEGER(1) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg2]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge575_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge576_test
LOGICAL FUNCTION ge576_test(x0, x1)
INTEGER(1) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge576_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge577_test
LOGICAL FUNCTION ge577_test(x0, x1)
INTEGER(1) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge577_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge578_test
LOGICAL FUNCTION ge578_test(x0, x1)
INTEGER(2) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i16
! CHECK:[[reg4:%[0-9]+]] = cmpi "sge", [[reg1]], [[reg3]] : i16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge578_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge579_test
LOGICAL FUNCTION ge579_test(x0, x1)
INTEGER(2) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = cmpi "sge", [[reg1]], [[reg2]] : i16
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
ge579_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge580_test
LOGICAL FUNCTION ge580_test(x0, x1)
INTEGER(2) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "sge", [[reg2]], [[reg3]] : i32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge580_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge581_test
LOGICAL FUNCTION ge581_test(x0, x1)
INTEGER(2) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "sge", [[reg2]], [[reg3]] : i64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge581_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge582_test
LOGICAL FUNCTION ge582_test(x0, x1)
INTEGER(2) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i128
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "sge", [[reg2]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge582_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge583_test
LOGICAL FUNCTION ge583_test(x0, x1)
INTEGER(2) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg2]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge583_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge584_test
LOGICAL FUNCTION ge584_test(x0, x1)
INTEGER(2) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg2]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge584_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge585_test
LOGICAL FUNCTION ge585_test(x0, x1)
INTEGER(2) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg2]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge585_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge586_test
LOGICAL FUNCTION ge586_test(x0, x1)
INTEGER(2) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge586_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge587_test
LOGICAL FUNCTION ge587_test(x0, x1)
INTEGER(2) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge587_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge588_test
LOGICAL FUNCTION ge588_test(x0, x1)
INTEGER(4) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i32
! CHECK:[[reg4:%[0-9]+]] = cmpi "sge", [[reg1]], [[reg3]] : i32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge588_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge589_test
LOGICAL FUNCTION ge589_test(x0, x1)
INTEGER(4) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i32
! CHECK:[[reg4:%[0-9]+]] = cmpi "sge", [[reg1]], [[reg3]] : i32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge589_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge590_test
LOGICAL FUNCTION ge590_test(x0, x1)
INTEGER(4) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = cmpi "sge", [[reg1]], [[reg2]] : i32
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
ge590_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge591_test
LOGICAL FUNCTION ge591_test(x0, x1)
INTEGER(4) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "sge", [[reg2]], [[reg3]] : i64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge591_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge592_test
LOGICAL FUNCTION ge592_test(x0, x1)
INTEGER(4) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i128
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "sge", [[reg2]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge592_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge593_test
LOGICAL FUNCTION ge593_test(x0, x1)
INTEGER(4) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg2]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge593_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge594_test
LOGICAL FUNCTION ge594_test(x0, x1)
INTEGER(4) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg2]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge594_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge595_test
LOGICAL FUNCTION ge595_test(x0, x1)
INTEGER(4) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg2]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge595_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge596_test
LOGICAL FUNCTION ge596_test(x0, x1)
INTEGER(4) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge596_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge597_test
LOGICAL FUNCTION ge597_test(x0, x1)
INTEGER(4) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge597_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge598_test
LOGICAL FUNCTION ge598_test(x0, x1)
INTEGER(8) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i64
! CHECK:[[reg4:%[0-9]+]] = cmpi "sge", [[reg1]], [[reg3]] : i64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge598_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge599_test
LOGICAL FUNCTION ge599_test(x0, x1)
INTEGER(8) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i64
! CHECK:[[reg4:%[0-9]+]] = cmpi "sge", [[reg1]], [[reg3]] : i64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge599_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge600_test
LOGICAL FUNCTION ge600_test(x0, x1)
INTEGER(8) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i64
! CHECK:[[reg4:%[0-9]+]] = cmpi "sge", [[reg1]], [[reg3]] : i64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge600_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge601_test
LOGICAL FUNCTION ge601_test(x0, x1)
INTEGER(8) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = cmpi "sge", [[reg1]], [[reg2]] : i64
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
ge601_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge602_test
LOGICAL FUNCTION ge602_test(x0, x1)
INTEGER(8) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i128
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = cmpi "sge", [[reg2]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge602_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge603_test
LOGICAL FUNCTION ge603_test(x0, x1)
INTEGER(8) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg2]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge603_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge604_test
LOGICAL FUNCTION ge604_test(x0, x1)
INTEGER(8) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg2]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge604_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge605_test
LOGICAL FUNCTION ge605_test(x0, x1)
INTEGER(8) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg2]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge605_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge606_test
LOGICAL FUNCTION ge606_test(x0, x1)
INTEGER(8) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge606_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge607_test
LOGICAL FUNCTION ge607_test(x0, x1)
INTEGER(8) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge607_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge608_test
LOGICAL FUNCTION ge608_test(x0, x1)
INTEGER(16) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i128
! CHECK:[[reg4:%[0-9]+]] = cmpi "sge", [[reg1]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge608_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge609_test
LOGICAL FUNCTION ge609_test(x0, x1)
INTEGER(16) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i128
! CHECK:[[reg4:%[0-9]+]] = cmpi "sge", [[reg1]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge609_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge610_test
LOGICAL FUNCTION ge610_test(x0, x1)
INTEGER(16) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i128
! CHECK:[[reg4:%[0-9]+]] = cmpi "sge", [[reg1]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge610_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge611_test
LOGICAL FUNCTION ge611_test(x0, x1)
INTEGER(16) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i128
! CHECK:[[reg4:%[0-9]+]] = cmpi "sge", [[reg1]], [[reg3]] : i128
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge611_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge612_test
LOGICAL FUNCTION ge612_test(x0, x1)
INTEGER(16) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = cmpi "sge", [[reg1]], [[reg2]] : i128
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
ge612_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge613_test
LOGICAL FUNCTION ge613_test(x0, x1)
INTEGER(16) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg2]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge613_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge614_test
LOGICAL FUNCTION ge614_test(x0, x1)
INTEGER(16) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg2]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge614_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge615_test
LOGICAL FUNCTION ge615_test(x0, x1)
INTEGER(16) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg2]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge615_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge616_test
LOGICAL FUNCTION ge616_test(x0, x1)
INTEGER(16) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge616_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge617_test
LOGICAL FUNCTION ge617_test(x0, x1)
INTEGER(16) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge617_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge618_test
LOGICAL FUNCTION ge618_test(x0, x1)
REAL(2) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg1]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge618_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge619_test
LOGICAL FUNCTION ge619_test(x0, x1)
REAL(2) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg1]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge619_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge620_test
LOGICAL FUNCTION ge620_test(x0, x1)
REAL(2) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg1]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge620_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge621_test
LOGICAL FUNCTION ge621_test(x0, x1)
REAL(2) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg1]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge621_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge622_test
LOGICAL FUNCTION ge622_test(x0, x1)
REAL(2) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg1]], [[reg3]] : f16
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge622_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge623_test
LOGICAL FUNCTION ge623_test(x0, x1)
REAL(2) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = fir.cmpf "oge", [[reg1]], [[reg2]] : f16
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
ge623_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge624_test
LOGICAL FUNCTION ge624_test(x0, x1)
REAL(2) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg2]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge624_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge625_test
LOGICAL FUNCTION ge625_test(x0, x1)
REAL(2) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg2]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge625_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge626_test
LOGICAL FUNCTION ge626_test(x0, x1)
REAL(2) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge626_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge627_test
LOGICAL FUNCTION ge627_test(x0, x1)
REAL(2) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge627_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge628_test
LOGICAL FUNCTION ge628_test(x0, x1)
REAL(4) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg1]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge628_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge629_test
LOGICAL FUNCTION ge629_test(x0, x1)
REAL(4) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg1]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge629_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge630_test
LOGICAL FUNCTION ge630_test(x0, x1)
REAL(4) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg1]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge630_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge631_test
LOGICAL FUNCTION ge631_test(x0, x1)
REAL(4) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg1]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge631_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge632_test
LOGICAL FUNCTION ge632_test(x0, x1)
REAL(4) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg1]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge632_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge633_test
LOGICAL FUNCTION ge633_test(x0, x1)
REAL(4) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg1]], [[reg3]] : f32
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge633_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge634_test
LOGICAL FUNCTION ge634_test(x0, x1)
REAL(4) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = fir.cmpf "oge", [[reg1]], [[reg2]] : f32
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
ge634_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge635_test
LOGICAL FUNCTION ge635_test(x0, x1)
REAL(4) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg2]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge635_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge636_test
LOGICAL FUNCTION ge636_test(x0, x1)
REAL(4) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge636_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge637_test
LOGICAL FUNCTION ge637_test(x0, x1)
REAL(4) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge637_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge638_test
LOGICAL FUNCTION ge638_test(x0, x1)
REAL(8) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg1]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge638_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge639_test
LOGICAL FUNCTION ge639_test(x0, x1)
REAL(8) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg1]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge639_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge640_test
LOGICAL FUNCTION ge640_test(x0, x1)
REAL(8) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg1]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge640_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge641_test
LOGICAL FUNCTION ge641_test(x0, x1)
REAL(8) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg1]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge641_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge642_test
LOGICAL FUNCTION ge642_test(x0, x1)
REAL(8) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg1]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge642_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge643_test
LOGICAL FUNCTION ge643_test(x0, x1)
REAL(8) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg1]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge643_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge644_test
LOGICAL FUNCTION ge644_test(x0, x1)
REAL(8) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg1]], [[reg3]] : f64
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge644_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge645_test
LOGICAL FUNCTION ge645_test(x0, x1)
REAL(8) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = fir.cmpf "oge", [[reg1]], [[reg2]] : f64
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
ge645_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge646_test
LOGICAL FUNCTION ge646_test(x0, x1)
REAL(8) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg2]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge646_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge647_test
LOGICAL FUNCTION ge647_test(x0, x1)
REAL(8) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge647_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge648_test
LOGICAL FUNCTION ge648_test(x0, x1)
REAL(10) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge648_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge649_test
LOGICAL FUNCTION ge649_test(x0, x1)
REAL(10) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge649_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge650_test
LOGICAL FUNCTION ge650_test(x0, x1)
REAL(10) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge650_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge651_test
LOGICAL FUNCTION ge651_test(x0, x1)
REAL(10) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge651_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge652_test
LOGICAL FUNCTION ge652_test(x0, x1)
REAL(10) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge652_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge653_test
LOGICAL FUNCTION ge653_test(x0, x1)
REAL(10) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge653_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge654_test
LOGICAL FUNCTION ge654_test(x0, x1)
REAL(10) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge654_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge655_test
LOGICAL FUNCTION ge655_test(x0, x1)
REAL(10) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg1]], [[reg3]] : !fir.real<10>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge655_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge656_test
LOGICAL FUNCTION ge656_test(x0, x1)
REAL(10) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = fir.cmpf "oge", [[reg1]], [[reg2]] : !fir.real<10>
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
ge656_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge657_test
LOGICAL FUNCTION ge657_test(x0, x1)
REAL(10) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg2]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge657_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge658_test
LOGICAL FUNCTION ge658_test(x0, x1)
REAL(16) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge658_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge659_test
LOGICAL FUNCTION ge659_test(x0, x1)
REAL(16) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge659_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge660_test
LOGICAL FUNCTION ge660_test(x0, x1)
REAL(16) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge660_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge661_test
LOGICAL FUNCTION ge661_test(x0, x1)
REAL(16) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge661_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge662_test
LOGICAL FUNCTION ge662_test(x0, x1)
REAL(16) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge662_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge663_test
LOGICAL FUNCTION ge663_test(x0, x1)
REAL(16) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge663_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge664_test
LOGICAL FUNCTION ge664_test(x0, x1)
REAL(16) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge664_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge665_test
LOGICAL FUNCTION ge665_test(x0, x1)
REAL(16) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge665_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge666_test
LOGICAL FUNCTION ge666_test(x0, x1)
REAL(16) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:[[reg4:%[0-9]+]] = fir.cmpf "oge", [[reg1]], [[reg3]] : !fir.real<16>
! CHECK:fir.convert [[reg4]] {{.*}} -> !fir.logical<4>
ge666_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:ge667_test
LOGICAL FUNCTION ge667_test(x0, x1)
REAL(16) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:[[reg3:%[0-9]+]] = fir.cmpf "oge", [[reg1]], [[reg2]] : !fir.real<16>
! CHECK:fir.convert [[reg3]] {{.*}} -> !fir.logical<4>
ge667_test = x0 .GE. x1
END FUNCTION

! CHECK-LABEL:add668_test
INTEGER(1) FUNCTION add668_test(x0, x1)
INTEGER(1) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:addi [[reg1]], [[reg2]] : i8
add668_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add669_test
INTEGER(2) FUNCTION add669_test(x0, x1)
INTEGER(1) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:addi [[reg2]], [[reg3]] : i16
add669_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add670_test
INTEGER(4) FUNCTION add670_test(x0, x1)
INTEGER(1) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:addi [[reg2]], [[reg3]] : i32
add670_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add671_test
INTEGER(8) FUNCTION add671_test(x0, x1)
INTEGER(1) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:addi [[reg2]], [[reg3]] : i64
add671_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add672_test
INTEGER(16) FUNCTION add672_test(x0, x1)
INTEGER(1) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i128
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:addi [[reg2]], [[reg3]] : i128
add672_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add673_test
REAL(2) FUNCTION add673_test(x0, x1)
INTEGER(1) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.addf [[reg2]], [[reg3]] : f16
add673_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add674_test
REAL(4) FUNCTION add674_test(x0, x1)
INTEGER(1) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.addf [[reg2]], [[reg3]] : f32
add674_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add675_test
REAL(8) FUNCTION add675_test(x0, x1)
INTEGER(1) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.addf [[reg2]], [[reg3]] : f64
add675_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add676_test
REAL(10) FUNCTION add676_test(x0, x1)
INTEGER(1) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.addf [[reg2]], [[reg3]] : !fir.real<10>
add676_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add677_test
REAL(16) FUNCTION add677_test(x0, x1)
INTEGER(1) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.addf [[reg2]], [[reg3]] : !fir.real<16>
add677_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add678_test
INTEGER(2) FUNCTION add678_test(x0, x1)
INTEGER(2) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i16
! CHECK:addi [[reg1]], [[reg3]] : i16
add678_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add679_test
INTEGER(2) FUNCTION add679_test(x0, x1)
INTEGER(2) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:addi [[reg1]], [[reg2]] : i16
add679_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add680_test
INTEGER(4) FUNCTION add680_test(x0, x1)
INTEGER(2) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:addi [[reg2]], [[reg3]] : i32
add680_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add681_test
INTEGER(8) FUNCTION add681_test(x0, x1)
INTEGER(2) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:addi [[reg2]], [[reg3]] : i64
add681_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add682_test
INTEGER(16) FUNCTION add682_test(x0, x1)
INTEGER(2) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i128
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:addi [[reg2]], [[reg3]] : i128
add682_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add683_test
REAL(2) FUNCTION add683_test(x0, x1)
INTEGER(2) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.addf [[reg2]], [[reg3]] : f16
add683_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add684_test
REAL(4) FUNCTION add684_test(x0, x1)
INTEGER(2) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.addf [[reg2]], [[reg3]] : f32
add684_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add685_test
REAL(8) FUNCTION add685_test(x0, x1)
INTEGER(2) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.addf [[reg2]], [[reg3]] : f64
add685_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add686_test
REAL(10) FUNCTION add686_test(x0, x1)
INTEGER(2) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.addf [[reg2]], [[reg3]] : !fir.real<10>
add686_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add687_test
REAL(16) FUNCTION add687_test(x0, x1)
INTEGER(2) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.addf [[reg2]], [[reg3]] : !fir.real<16>
add687_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add688_test
INTEGER(4) FUNCTION add688_test(x0, x1)
INTEGER(4) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i32
! CHECK:addi [[reg1]], [[reg3]] : i32
add688_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add689_test
INTEGER(4) FUNCTION add689_test(x0, x1)
INTEGER(4) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i32
! CHECK:addi [[reg1]], [[reg3]] : i32
add689_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add690_test
INTEGER(4) FUNCTION add690_test(x0, x1)
INTEGER(4) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:addi [[reg1]], [[reg2]] : i32
add690_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add691_test
INTEGER(8) FUNCTION add691_test(x0, x1)
INTEGER(4) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:addi [[reg2]], [[reg3]] : i64
add691_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add692_test
INTEGER(16) FUNCTION add692_test(x0, x1)
INTEGER(4) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i128
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:addi [[reg2]], [[reg3]] : i128
add692_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add693_test
REAL(2) FUNCTION add693_test(x0, x1)
INTEGER(4) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.addf [[reg2]], [[reg3]] : f16
add693_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add694_test
REAL(4) FUNCTION add694_test(x0, x1)
INTEGER(4) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.addf [[reg2]], [[reg3]] : f32
add694_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add695_test
REAL(8) FUNCTION add695_test(x0, x1)
INTEGER(4) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.addf [[reg2]], [[reg3]] : f64
add695_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add696_test
REAL(10) FUNCTION add696_test(x0, x1)
INTEGER(4) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.addf [[reg2]], [[reg3]] : !fir.real<10>
add696_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add697_test
REAL(16) FUNCTION add697_test(x0, x1)
INTEGER(4) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.addf [[reg2]], [[reg3]] : !fir.real<16>
add697_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add698_test
INTEGER(8) FUNCTION add698_test(x0, x1)
INTEGER(8) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i64
! CHECK:addi [[reg1]], [[reg3]] : i64
add698_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add699_test
INTEGER(8) FUNCTION add699_test(x0, x1)
INTEGER(8) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i64
! CHECK:addi [[reg1]], [[reg3]] : i64
add699_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add700_test
INTEGER(8) FUNCTION add700_test(x0, x1)
INTEGER(8) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i64
! CHECK:addi [[reg1]], [[reg3]] : i64
add700_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add701_test
INTEGER(8) FUNCTION add701_test(x0, x1)
INTEGER(8) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:addi [[reg1]], [[reg2]] : i64
add701_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add702_test
INTEGER(16) FUNCTION add702_test(x0, x1)
INTEGER(8) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i128
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:addi [[reg2]], [[reg3]] : i128
add702_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add703_test
REAL(2) FUNCTION add703_test(x0, x1)
INTEGER(8) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.addf [[reg2]], [[reg3]] : f16
add703_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add704_test
REAL(4) FUNCTION add704_test(x0, x1)
INTEGER(8) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.addf [[reg2]], [[reg3]] : f32
add704_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add705_test
REAL(8) FUNCTION add705_test(x0, x1)
INTEGER(8) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.addf [[reg2]], [[reg3]] : f64
add705_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add706_test
REAL(10) FUNCTION add706_test(x0, x1)
INTEGER(8) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.addf [[reg2]], [[reg3]] : !fir.real<10>
add706_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add707_test
REAL(16) FUNCTION add707_test(x0, x1)
INTEGER(8) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.addf [[reg2]], [[reg3]] : !fir.real<16>
add707_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add708_test
INTEGER(16) FUNCTION add708_test(x0, x1)
INTEGER(16) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i128
! CHECK:addi [[reg1]], [[reg3]] : i128
add708_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add709_test
INTEGER(16) FUNCTION add709_test(x0, x1)
INTEGER(16) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i128
! CHECK:addi [[reg1]], [[reg3]] : i128
add709_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add710_test
INTEGER(16) FUNCTION add710_test(x0, x1)
INTEGER(16) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i128
! CHECK:addi [[reg1]], [[reg3]] : i128
add710_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add711_test
INTEGER(16) FUNCTION add711_test(x0, x1)
INTEGER(16) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i128
! CHECK:addi [[reg1]], [[reg3]] : i128
add711_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add712_test
INTEGER(16) FUNCTION add712_test(x0, x1)
INTEGER(16) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:addi [[reg1]], [[reg2]] : i128
add712_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add713_test
REAL(2) FUNCTION add713_test(x0, x1)
INTEGER(16) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.addf [[reg2]], [[reg3]] : f16
add713_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add714_test
REAL(4) FUNCTION add714_test(x0, x1)
INTEGER(16) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.addf [[reg2]], [[reg3]] : f32
add714_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add715_test
REAL(8) FUNCTION add715_test(x0, x1)
INTEGER(16) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.addf [[reg2]], [[reg3]] : f64
add715_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add716_test
REAL(10) FUNCTION add716_test(x0, x1)
INTEGER(16) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.addf [[reg2]], [[reg3]] : !fir.real<10>
add716_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add717_test
REAL(16) FUNCTION add717_test(x0, x1)
INTEGER(16) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.addf [[reg2]], [[reg3]] : !fir.real<16>
add717_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add718_test
REAL(2) FUNCTION add718_test(x0, x1)
REAL(2) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:fir.addf [[reg1]], [[reg3]] : f16
add718_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add719_test
REAL(2) FUNCTION add719_test(x0, x1)
REAL(2) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:fir.addf [[reg1]], [[reg3]] : f16
add719_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add720_test
REAL(2) FUNCTION add720_test(x0, x1)
REAL(2) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:fir.addf [[reg1]], [[reg3]] : f16
add720_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add721_test
REAL(2) FUNCTION add721_test(x0, x1)
REAL(2) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:fir.addf [[reg1]], [[reg3]] : f16
add721_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add722_test
REAL(2) FUNCTION add722_test(x0, x1)
REAL(2) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:fir.addf [[reg1]], [[reg3]] : f16
add722_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add723_test
REAL(2) FUNCTION add723_test(x0, x1)
REAL(2) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:fir.addf [[reg1]], [[reg2]] : f16
add723_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add724_test
REAL(4) FUNCTION add724_test(x0, x1)
REAL(2) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.addf [[reg2]], [[reg3]] : f32
add724_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add725_test
REAL(8) FUNCTION add725_test(x0, x1)
REAL(2) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.addf [[reg2]], [[reg3]] : f64
add725_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add726_test
REAL(10) FUNCTION add726_test(x0, x1)
REAL(2) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.addf [[reg2]], [[reg3]] : !fir.real<10>
add726_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add727_test
REAL(16) FUNCTION add727_test(x0, x1)
REAL(2) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.addf [[reg2]], [[reg3]] : !fir.real<16>
add727_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add728_test
REAL(4) FUNCTION add728_test(x0, x1)
REAL(4) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:fir.addf [[reg1]], [[reg3]] : f32
add728_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add729_test
REAL(4) FUNCTION add729_test(x0, x1)
REAL(4) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:fir.addf [[reg1]], [[reg3]] : f32
add729_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add730_test
REAL(4) FUNCTION add730_test(x0, x1)
REAL(4) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:fir.addf [[reg1]], [[reg3]] : f32
add730_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add731_test
REAL(4) FUNCTION add731_test(x0, x1)
REAL(4) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:fir.addf [[reg1]], [[reg3]] : f32
add731_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add732_test
REAL(4) FUNCTION add732_test(x0, x1)
REAL(4) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:fir.addf [[reg1]], [[reg3]] : f32
add732_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add733_test
REAL(4) FUNCTION add733_test(x0, x1)
REAL(4) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:fir.addf [[reg1]], [[reg3]] : f32
add733_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add734_test
REAL(4) FUNCTION add734_test(x0, x1)
REAL(4) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:fir.addf [[reg1]], [[reg2]] : f32
add734_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add735_test
REAL(8) FUNCTION add735_test(x0, x1)
REAL(4) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.addf [[reg2]], [[reg3]] : f64
add735_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add736_test
REAL(10) FUNCTION add736_test(x0, x1)
REAL(4) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.addf [[reg2]], [[reg3]] : !fir.real<10>
add736_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add737_test
REAL(16) FUNCTION add737_test(x0, x1)
REAL(4) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.addf [[reg2]], [[reg3]] : !fir.real<16>
add737_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add738_test
REAL(8) FUNCTION add738_test(x0, x1)
REAL(8) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:fir.addf [[reg1]], [[reg3]] : f64
add738_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add739_test
REAL(8) FUNCTION add739_test(x0, x1)
REAL(8) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:fir.addf [[reg1]], [[reg3]] : f64
add739_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add740_test
REAL(8) FUNCTION add740_test(x0, x1)
REAL(8) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:fir.addf [[reg1]], [[reg3]] : f64
add740_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add741_test
REAL(8) FUNCTION add741_test(x0, x1)
REAL(8) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:fir.addf [[reg1]], [[reg3]] : f64
add741_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add742_test
REAL(8) FUNCTION add742_test(x0, x1)
REAL(8) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:fir.addf [[reg1]], [[reg3]] : f64
add742_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add743_test
REAL(8) FUNCTION add743_test(x0, x1)
REAL(8) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:fir.addf [[reg1]], [[reg3]] : f64
add743_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add744_test
REAL(8) FUNCTION add744_test(x0, x1)
REAL(8) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:fir.addf [[reg1]], [[reg3]] : f64
add744_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add745_test
REAL(8) FUNCTION add745_test(x0, x1)
REAL(8) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:fir.addf [[reg1]], [[reg2]] : f64
add745_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add746_test
REAL(10) FUNCTION add746_test(x0, x1)
REAL(8) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.addf [[reg2]], [[reg3]] : !fir.real<10>
add746_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add747_test
REAL(16) FUNCTION add747_test(x0, x1)
REAL(8) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.addf [[reg2]], [[reg3]] : !fir.real<16>
add747_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add748_test
REAL(10) FUNCTION add748_test(x0, x1)
REAL(10) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:fir.addf [[reg1]], [[reg3]] : !fir.real<10>
add748_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add749_test
REAL(10) FUNCTION add749_test(x0, x1)
REAL(10) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:fir.addf [[reg1]], [[reg3]] : !fir.real<10>
add749_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add750_test
REAL(10) FUNCTION add750_test(x0, x1)
REAL(10) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:fir.addf [[reg1]], [[reg3]] : !fir.real<10>
add750_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add751_test
REAL(10) FUNCTION add751_test(x0, x1)
REAL(10) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:fir.addf [[reg1]], [[reg3]] : !fir.real<10>
add751_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add752_test
REAL(10) FUNCTION add752_test(x0, x1)
REAL(10) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:fir.addf [[reg1]], [[reg3]] : !fir.real<10>
add752_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add753_test
REAL(10) FUNCTION add753_test(x0, x1)
REAL(10) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:fir.addf [[reg1]], [[reg3]] : !fir.real<10>
add753_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add754_test
REAL(10) FUNCTION add754_test(x0, x1)
REAL(10) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:fir.addf [[reg1]], [[reg3]] : !fir.real<10>
add754_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add755_test
REAL(10) FUNCTION add755_test(x0, x1)
REAL(10) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:fir.addf [[reg1]], [[reg3]] : !fir.real<10>
add755_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add756_test
REAL(10) FUNCTION add756_test(x0, x1)
REAL(10) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:fir.addf [[reg1]], [[reg2]] : !fir.real<10>
add756_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add757_test
REAL(16) FUNCTION add757_test(x0, x1)
REAL(10) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.addf [[reg2]], [[reg3]] : !fir.real<16>
add757_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add758_test
REAL(16) FUNCTION add758_test(x0, x1)
REAL(16) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:fir.addf [[reg1]], [[reg3]] : !fir.real<16>
add758_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add759_test
REAL(16) FUNCTION add759_test(x0, x1)
REAL(16) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:fir.addf [[reg1]], [[reg3]] : !fir.real<16>
add759_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add760_test
REAL(16) FUNCTION add760_test(x0, x1)
REAL(16) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:fir.addf [[reg1]], [[reg3]] : !fir.real<16>
add760_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add761_test
REAL(16) FUNCTION add761_test(x0, x1)
REAL(16) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:fir.addf [[reg1]], [[reg3]] : !fir.real<16>
add761_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add762_test
REAL(16) FUNCTION add762_test(x0, x1)
REAL(16) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:fir.addf [[reg1]], [[reg3]] : !fir.real<16>
add762_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add763_test
REAL(16) FUNCTION add763_test(x0, x1)
REAL(16) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:fir.addf [[reg1]], [[reg3]] : !fir.real<16>
add763_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add764_test
REAL(16) FUNCTION add764_test(x0, x1)
REAL(16) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:fir.addf [[reg1]], [[reg3]] : !fir.real<16>
add764_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add765_test
REAL(16) FUNCTION add765_test(x0, x1)
REAL(16) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:fir.addf [[reg1]], [[reg3]] : !fir.real<16>
add765_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add766_test
REAL(16) FUNCTION add766_test(x0, x1)
REAL(16) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:fir.addf [[reg1]], [[reg3]] : !fir.real<16>
add766_test = x0 + x1
END FUNCTION

! CHECK-LABEL:add767_test
REAL(16) FUNCTION add767_test(x0, x1)
REAL(16) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:fir.addf [[reg1]], [[reg2]] : !fir.real<16>
add767_test = x0 + x1
END FUNCTION

! CHECK-LABEL:sub768_test
INTEGER(1) FUNCTION sub768_test(x0, x1)
INTEGER(1) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:subi [[reg1]], [[reg2]] : i8
sub768_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub769_test
INTEGER(2) FUNCTION sub769_test(x0, x1)
INTEGER(1) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:subi [[reg2]], [[reg3]] : i16
sub769_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub770_test
INTEGER(4) FUNCTION sub770_test(x0, x1)
INTEGER(1) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:subi [[reg2]], [[reg3]] : i32
sub770_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub771_test
INTEGER(8) FUNCTION sub771_test(x0, x1)
INTEGER(1) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:subi [[reg2]], [[reg3]] : i64
sub771_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub772_test
INTEGER(16) FUNCTION sub772_test(x0, x1)
INTEGER(1) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i128
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:subi [[reg2]], [[reg3]] : i128
sub772_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub773_test
REAL(2) FUNCTION sub773_test(x0, x1)
INTEGER(1) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.subf [[reg2]], [[reg3]] : f16
sub773_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub774_test
REAL(4) FUNCTION sub774_test(x0, x1)
INTEGER(1) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.subf [[reg2]], [[reg3]] : f32
sub774_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub775_test
REAL(8) FUNCTION sub775_test(x0, x1)
INTEGER(1) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.subf [[reg2]], [[reg3]] : f64
sub775_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub776_test
REAL(10) FUNCTION sub776_test(x0, x1)
INTEGER(1) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.subf [[reg2]], [[reg3]] : !fir.real<10>
sub776_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub777_test
REAL(16) FUNCTION sub777_test(x0, x1)
INTEGER(1) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.subf [[reg2]], [[reg3]] : !fir.real<16>
sub777_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub778_test
INTEGER(2) FUNCTION sub778_test(x0, x1)
INTEGER(2) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i16
! CHECK:subi [[reg1]], [[reg3]] : i16
sub778_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub779_test
INTEGER(2) FUNCTION sub779_test(x0, x1)
INTEGER(2) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:subi [[reg1]], [[reg2]] : i16
sub779_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub780_test
INTEGER(4) FUNCTION sub780_test(x0, x1)
INTEGER(2) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:subi [[reg2]], [[reg3]] : i32
sub780_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub781_test
INTEGER(8) FUNCTION sub781_test(x0, x1)
INTEGER(2) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:subi [[reg2]], [[reg3]] : i64
sub781_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub782_test
INTEGER(16) FUNCTION sub782_test(x0, x1)
INTEGER(2) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i128
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:subi [[reg2]], [[reg3]] : i128
sub782_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub783_test
REAL(2) FUNCTION sub783_test(x0, x1)
INTEGER(2) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.subf [[reg2]], [[reg3]] : f16
sub783_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub784_test
REAL(4) FUNCTION sub784_test(x0, x1)
INTEGER(2) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.subf [[reg2]], [[reg3]] : f32
sub784_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub785_test
REAL(8) FUNCTION sub785_test(x0, x1)
INTEGER(2) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.subf [[reg2]], [[reg3]] : f64
sub785_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub786_test
REAL(10) FUNCTION sub786_test(x0, x1)
INTEGER(2) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.subf [[reg2]], [[reg3]] : !fir.real<10>
sub786_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub787_test
REAL(16) FUNCTION sub787_test(x0, x1)
INTEGER(2) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.subf [[reg2]], [[reg3]] : !fir.real<16>
sub787_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub788_test
INTEGER(4) FUNCTION sub788_test(x0, x1)
INTEGER(4) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i32
! CHECK:subi [[reg1]], [[reg3]] : i32
sub788_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub789_test
INTEGER(4) FUNCTION sub789_test(x0, x1)
INTEGER(4) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i32
! CHECK:subi [[reg1]], [[reg3]] : i32
sub789_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub790_test
INTEGER(4) FUNCTION sub790_test(x0, x1)
INTEGER(4) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:subi [[reg1]], [[reg2]] : i32
sub790_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub791_test
INTEGER(8) FUNCTION sub791_test(x0, x1)
INTEGER(4) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:subi [[reg2]], [[reg3]] : i64
sub791_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub792_test
INTEGER(16) FUNCTION sub792_test(x0, x1)
INTEGER(4) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i128
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:subi [[reg2]], [[reg3]] : i128
sub792_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub793_test
REAL(2) FUNCTION sub793_test(x0, x1)
INTEGER(4) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.subf [[reg2]], [[reg3]] : f16
sub793_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub794_test
REAL(4) FUNCTION sub794_test(x0, x1)
INTEGER(4) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.subf [[reg2]], [[reg3]] : f32
sub794_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub795_test
REAL(8) FUNCTION sub795_test(x0, x1)
INTEGER(4) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.subf [[reg2]], [[reg3]] : f64
sub795_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub796_test
REAL(10) FUNCTION sub796_test(x0, x1)
INTEGER(4) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.subf [[reg2]], [[reg3]] : !fir.real<10>
sub796_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub797_test
REAL(16) FUNCTION sub797_test(x0, x1)
INTEGER(4) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.subf [[reg2]], [[reg3]] : !fir.real<16>
sub797_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub798_test
INTEGER(8) FUNCTION sub798_test(x0, x1)
INTEGER(8) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i64
! CHECK:subi [[reg1]], [[reg3]] : i64
sub798_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub799_test
INTEGER(8) FUNCTION sub799_test(x0, x1)
INTEGER(8) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i64
! CHECK:subi [[reg1]], [[reg3]] : i64
sub799_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub800_test
INTEGER(8) FUNCTION sub800_test(x0, x1)
INTEGER(8) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i64
! CHECK:subi [[reg1]], [[reg3]] : i64
sub800_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub801_test
INTEGER(8) FUNCTION sub801_test(x0, x1)
INTEGER(8) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:subi [[reg1]], [[reg2]] : i64
sub801_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub802_test
INTEGER(16) FUNCTION sub802_test(x0, x1)
INTEGER(8) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i128
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:subi [[reg2]], [[reg3]] : i128
sub802_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub803_test
REAL(2) FUNCTION sub803_test(x0, x1)
INTEGER(8) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.subf [[reg2]], [[reg3]] : f16
sub803_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub804_test
REAL(4) FUNCTION sub804_test(x0, x1)
INTEGER(8) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.subf [[reg2]], [[reg3]] : f32
sub804_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub805_test
REAL(8) FUNCTION sub805_test(x0, x1)
INTEGER(8) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.subf [[reg2]], [[reg3]] : f64
sub805_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub806_test
REAL(10) FUNCTION sub806_test(x0, x1)
INTEGER(8) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.subf [[reg2]], [[reg3]] : !fir.real<10>
sub806_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub807_test
REAL(16) FUNCTION sub807_test(x0, x1)
INTEGER(8) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.subf [[reg2]], [[reg3]] : !fir.real<16>
sub807_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub808_test
INTEGER(16) FUNCTION sub808_test(x0, x1)
INTEGER(16) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i128
! CHECK:subi [[reg1]], [[reg3]] : i128
sub808_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub809_test
INTEGER(16) FUNCTION sub809_test(x0, x1)
INTEGER(16) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i128
! CHECK:subi [[reg1]], [[reg3]] : i128
sub809_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub810_test
INTEGER(16) FUNCTION sub810_test(x0, x1)
INTEGER(16) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i128
! CHECK:subi [[reg1]], [[reg3]] : i128
sub810_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub811_test
INTEGER(16) FUNCTION sub811_test(x0, x1)
INTEGER(16) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i128
! CHECK:subi [[reg1]], [[reg3]] : i128
sub811_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub812_test
INTEGER(16) FUNCTION sub812_test(x0, x1)
INTEGER(16) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:subi [[reg1]], [[reg2]] : i128
sub812_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub813_test
REAL(2) FUNCTION sub813_test(x0, x1)
INTEGER(16) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.subf [[reg2]], [[reg3]] : f16
sub813_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub814_test
REAL(4) FUNCTION sub814_test(x0, x1)
INTEGER(16) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.subf [[reg2]], [[reg3]] : f32
sub814_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub815_test
REAL(8) FUNCTION sub815_test(x0, x1)
INTEGER(16) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.subf [[reg2]], [[reg3]] : f64
sub815_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub816_test
REAL(10) FUNCTION sub816_test(x0, x1)
INTEGER(16) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.subf [[reg2]], [[reg3]] : !fir.real<10>
sub816_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub817_test
REAL(16) FUNCTION sub817_test(x0, x1)
INTEGER(16) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.subf [[reg2]], [[reg3]] : !fir.real<16>
sub817_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub818_test
REAL(2) FUNCTION sub818_test(x0, x1)
REAL(2) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:fir.subf [[reg1]], [[reg3]] : f16
sub818_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub819_test
REAL(2) FUNCTION sub819_test(x0, x1)
REAL(2) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:fir.subf [[reg1]], [[reg3]] : f16
sub819_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub820_test
REAL(2) FUNCTION sub820_test(x0, x1)
REAL(2) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:fir.subf [[reg1]], [[reg3]] : f16
sub820_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub821_test
REAL(2) FUNCTION sub821_test(x0, x1)
REAL(2) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:fir.subf [[reg1]], [[reg3]] : f16
sub821_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub822_test
REAL(2) FUNCTION sub822_test(x0, x1)
REAL(2) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:fir.subf [[reg1]], [[reg3]] : f16
sub822_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub823_test
REAL(2) FUNCTION sub823_test(x0, x1)
REAL(2) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:fir.subf [[reg1]], [[reg2]] : f16
sub823_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub824_test
REAL(4) FUNCTION sub824_test(x0, x1)
REAL(2) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.subf [[reg2]], [[reg3]] : f32
sub824_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub825_test
REAL(8) FUNCTION sub825_test(x0, x1)
REAL(2) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.subf [[reg2]], [[reg3]] : f64
sub825_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub826_test
REAL(10) FUNCTION sub826_test(x0, x1)
REAL(2) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.subf [[reg2]], [[reg3]] : !fir.real<10>
sub826_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub827_test
REAL(16) FUNCTION sub827_test(x0, x1)
REAL(2) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.subf [[reg2]], [[reg3]] : !fir.real<16>
sub827_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub828_test
REAL(4) FUNCTION sub828_test(x0, x1)
REAL(4) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:fir.subf [[reg1]], [[reg3]] : f32
sub828_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub829_test
REAL(4) FUNCTION sub829_test(x0, x1)
REAL(4) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:fir.subf [[reg1]], [[reg3]] : f32
sub829_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub830_test
REAL(4) FUNCTION sub830_test(x0, x1)
REAL(4) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:fir.subf [[reg1]], [[reg3]] : f32
sub830_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub831_test
REAL(4) FUNCTION sub831_test(x0, x1)
REAL(4) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:fir.subf [[reg1]], [[reg3]] : f32
sub831_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub832_test
REAL(4) FUNCTION sub832_test(x0, x1)
REAL(4) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:fir.subf [[reg1]], [[reg3]] : f32
sub832_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub833_test
REAL(4) FUNCTION sub833_test(x0, x1)
REAL(4) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:fir.subf [[reg1]], [[reg3]] : f32
sub833_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub834_test
REAL(4) FUNCTION sub834_test(x0, x1)
REAL(4) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:fir.subf [[reg1]], [[reg2]] : f32
sub834_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub835_test
REAL(8) FUNCTION sub835_test(x0, x1)
REAL(4) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.subf [[reg2]], [[reg3]] : f64
sub835_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub836_test
REAL(10) FUNCTION sub836_test(x0, x1)
REAL(4) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.subf [[reg2]], [[reg3]] : !fir.real<10>
sub836_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub837_test
REAL(16) FUNCTION sub837_test(x0, x1)
REAL(4) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.subf [[reg2]], [[reg3]] : !fir.real<16>
sub837_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub838_test
REAL(8) FUNCTION sub838_test(x0, x1)
REAL(8) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:fir.subf [[reg1]], [[reg3]] : f64
sub838_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub839_test
REAL(8) FUNCTION sub839_test(x0, x1)
REAL(8) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:fir.subf [[reg1]], [[reg3]] : f64
sub839_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub840_test
REAL(8) FUNCTION sub840_test(x0, x1)
REAL(8) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:fir.subf [[reg1]], [[reg3]] : f64
sub840_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub841_test
REAL(8) FUNCTION sub841_test(x0, x1)
REAL(8) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:fir.subf [[reg1]], [[reg3]] : f64
sub841_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub842_test
REAL(8) FUNCTION sub842_test(x0, x1)
REAL(8) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:fir.subf [[reg1]], [[reg3]] : f64
sub842_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub843_test
REAL(8) FUNCTION sub843_test(x0, x1)
REAL(8) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:fir.subf [[reg1]], [[reg3]] : f64
sub843_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub844_test
REAL(8) FUNCTION sub844_test(x0, x1)
REAL(8) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:fir.subf [[reg1]], [[reg3]] : f64
sub844_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub845_test
REAL(8) FUNCTION sub845_test(x0, x1)
REAL(8) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:fir.subf [[reg1]], [[reg2]] : f64
sub845_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub846_test
REAL(10) FUNCTION sub846_test(x0, x1)
REAL(8) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.subf [[reg2]], [[reg3]] : !fir.real<10>
sub846_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub847_test
REAL(16) FUNCTION sub847_test(x0, x1)
REAL(8) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.subf [[reg2]], [[reg3]] : !fir.real<16>
sub847_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub848_test
REAL(10) FUNCTION sub848_test(x0, x1)
REAL(10) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:fir.subf [[reg1]], [[reg3]] : !fir.real<10>
sub848_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub849_test
REAL(10) FUNCTION sub849_test(x0, x1)
REAL(10) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:fir.subf [[reg1]], [[reg3]] : !fir.real<10>
sub849_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub850_test
REAL(10) FUNCTION sub850_test(x0, x1)
REAL(10) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:fir.subf [[reg1]], [[reg3]] : !fir.real<10>
sub850_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub851_test
REAL(10) FUNCTION sub851_test(x0, x1)
REAL(10) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:fir.subf [[reg1]], [[reg3]] : !fir.real<10>
sub851_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub852_test
REAL(10) FUNCTION sub852_test(x0, x1)
REAL(10) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:fir.subf [[reg1]], [[reg3]] : !fir.real<10>
sub852_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub853_test
REAL(10) FUNCTION sub853_test(x0, x1)
REAL(10) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:fir.subf [[reg1]], [[reg3]] : !fir.real<10>
sub853_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub854_test
REAL(10) FUNCTION sub854_test(x0, x1)
REAL(10) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:fir.subf [[reg1]], [[reg3]] : !fir.real<10>
sub854_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub855_test
REAL(10) FUNCTION sub855_test(x0, x1)
REAL(10) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:fir.subf [[reg1]], [[reg3]] : !fir.real<10>
sub855_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub856_test
REAL(10) FUNCTION sub856_test(x0, x1)
REAL(10) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:fir.subf [[reg1]], [[reg2]] : !fir.real<10>
sub856_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub857_test
REAL(16) FUNCTION sub857_test(x0, x1)
REAL(10) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.subf [[reg2]], [[reg3]] : !fir.real<16>
sub857_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub858_test
REAL(16) FUNCTION sub858_test(x0, x1)
REAL(16) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:fir.subf [[reg1]], [[reg3]] : !fir.real<16>
sub858_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub859_test
REAL(16) FUNCTION sub859_test(x0, x1)
REAL(16) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:fir.subf [[reg1]], [[reg3]] : !fir.real<16>
sub859_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub860_test
REAL(16) FUNCTION sub860_test(x0, x1)
REAL(16) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:fir.subf [[reg1]], [[reg3]] : !fir.real<16>
sub860_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub861_test
REAL(16) FUNCTION sub861_test(x0, x1)
REAL(16) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:fir.subf [[reg1]], [[reg3]] : !fir.real<16>
sub861_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub862_test
REAL(16) FUNCTION sub862_test(x0, x1)
REAL(16) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:fir.subf [[reg1]], [[reg3]] : !fir.real<16>
sub862_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub863_test
REAL(16) FUNCTION sub863_test(x0, x1)
REAL(16) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:fir.subf [[reg1]], [[reg3]] : !fir.real<16>
sub863_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub864_test
REAL(16) FUNCTION sub864_test(x0, x1)
REAL(16) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:fir.subf [[reg1]], [[reg3]] : !fir.real<16>
sub864_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub865_test
REAL(16) FUNCTION sub865_test(x0, x1)
REAL(16) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:fir.subf [[reg1]], [[reg3]] : !fir.real<16>
sub865_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub866_test
REAL(16) FUNCTION sub866_test(x0, x1)
REAL(16) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:fir.subf [[reg1]], [[reg3]] : !fir.real<16>
sub866_test = x0 - x1
END FUNCTION

! CHECK-LABEL:sub867_test
REAL(16) FUNCTION sub867_test(x0, x1)
REAL(16) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:fir.subf [[reg1]], [[reg2]] : !fir.real<16>
sub867_test = x0 - x1
END FUNCTION

! CHECK-LABEL:mult868_test
INTEGER(1) FUNCTION mult868_test(x0, x1)
INTEGER(1) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:muli [[reg1]], [[reg2]] : i8
mult868_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult869_test
INTEGER(2) FUNCTION mult869_test(x0, x1)
INTEGER(1) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:muli [[reg2]], [[reg3]] : i16
mult869_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult870_test
INTEGER(4) FUNCTION mult870_test(x0, x1)
INTEGER(1) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:muli [[reg2]], [[reg3]] : i32
mult870_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult871_test
INTEGER(8) FUNCTION mult871_test(x0, x1)
INTEGER(1) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:muli [[reg2]], [[reg3]] : i64
mult871_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult872_test
INTEGER(16) FUNCTION mult872_test(x0, x1)
INTEGER(1) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i128
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:muli [[reg2]], [[reg3]] : i128
mult872_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult873_test
REAL(2) FUNCTION mult873_test(x0, x1)
INTEGER(1) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.mulf [[reg2]], [[reg3]] : f16
mult873_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult874_test
REAL(4) FUNCTION mult874_test(x0, x1)
INTEGER(1) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.mulf [[reg2]], [[reg3]] : f32
mult874_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult875_test
REAL(8) FUNCTION mult875_test(x0, x1)
INTEGER(1) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.mulf [[reg2]], [[reg3]] : f64
mult875_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult876_test
REAL(10) FUNCTION mult876_test(x0, x1)
INTEGER(1) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.mulf [[reg2]], [[reg3]] : !fir.real<10>
mult876_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult877_test
REAL(16) FUNCTION mult877_test(x0, x1)
INTEGER(1) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.mulf [[reg2]], [[reg3]] : !fir.real<16>
mult877_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult878_test
INTEGER(2) FUNCTION mult878_test(x0, x1)
INTEGER(2) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i16
! CHECK:muli [[reg1]], [[reg3]] : i16
mult878_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult879_test
INTEGER(2) FUNCTION mult879_test(x0, x1)
INTEGER(2) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:muli [[reg1]], [[reg2]] : i16
mult879_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult880_test
INTEGER(4) FUNCTION mult880_test(x0, x1)
INTEGER(2) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:muli [[reg2]], [[reg3]] : i32
mult880_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult881_test
INTEGER(8) FUNCTION mult881_test(x0, x1)
INTEGER(2) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:muli [[reg2]], [[reg3]] : i64
mult881_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult882_test
INTEGER(16) FUNCTION mult882_test(x0, x1)
INTEGER(2) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i128
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:muli [[reg2]], [[reg3]] : i128
mult882_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult883_test
REAL(2) FUNCTION mult883_test(x0, x1)
INTEGER(2) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.mulf [[reg2]], [[reg3]] : f16
mult883_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult884_test
REAL(4) FUNCTION mult884_test(x0, x1)
INTEGER(2) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.mulf [[reg2]], [[reg3]] : f32
mult884_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult885_test
REAL(8) FUNCTION mult885_test(x0, x1)
INTEGER(2) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.mulf [[reg2]], [[reg3]] : f64
mult885_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult886_test
REAL(10) FUNCTION mult886_test(x0, x1)
INTEGER(2) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.mulf [[reg2]], [[reg3]] : !fir.real<10>
mult886_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult887_test
REAL(16) FUNCTION mult887_test(x0, x1)
INTEGER(2) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.mulf [[reg2]], [[reg3]] : !fir.real<16>
mult887_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult888_test
INTEGER(4) FUNCTION mult888_test(x0, x1)
INTEGER(4) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i32
! CHECK:muli [[reg1]], [[reg3]] : i32
mult888_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult889_test
INTEGER(4) FUNCTION mult889_test(x0, x1)
INTEGER(4) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i32
! CHECK:muli [[reg1]], [[reg3]] : i32
mult889_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult890_test
INTEGER(4) FUNCTION mult890_test(x0, x1)
INTEGER(4) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:muli [[reg1]], [[reg2]] : i32
mult890_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult891_test
INTEGER(8) FUNCTION mult891_test(x0, x1)
INTEGER(4) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:muli [[reg2]], [[reg3]] : i64
mult891_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult892_test
INTEGER(16) FUNCTION mult892_test(x0, x1)
INTEGER(4) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i128
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:muli [[reg2]], [[reg3]] : i128
mult892_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult893_test
REAL(2) FUNCTION mult893_test(x0, x1)
INTEGER(4) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.mulf [[reg2]], [[reg3]] : f16
mult893_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult894_test
REAL(4) FUNCTION mult894_test(x0, x1)
INTEGER(4) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.mulf [[reg2]], [[reg3]] : f32
mult894_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult895_test
REAL(8) FUNCTION mult895_test(x0, x1)
INTEGER(4) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.mulf [[reg2]], [[reg3]] : f64
mult895_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult896_test
REAL(10) FUNCTION mult896_test(x0, x1)
INTEGER(4) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.mulf [[reg2]], [[reg3]] : !fir.real<10>
mult896_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult897_test
REAL(16) FUNCTION mult897_test(x0, x1)
INTEGER(4) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.mulf [[reg2]], [[reg3]] : !fir.real<16>
mult897_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult898_test
INTEGER(8) FUNCTION mult898_test(x0, x1)
INTEGER(8) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i64
! CHECK:muli [[reg1]], [[reg3]] : i64
mult898_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult899_test
INTEGER(8) FUNCTION mult899_test(x0, x1)
INTEGER(8) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i64
! CHECK:muli [[reg1]], [[reg3]] : i64
mult899_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult900_test
INTEGER(8) FUNCTION mult900_test(x0, x1)
INTEGER(8) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i64
! CHECK:muli [[reg1]], [[reg3]] : i64
mult900_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult901_test
INTEGER(8) FUNCTION mult901_test(x0, x1)
INTEGER(8) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:muli [[reg1]], [[reg2]] : i64
mult901_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult902_test
INTEGER(16) FUNCTION mult902_test(x0, x1)
INTEGER(8) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i128
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:muli [[reg2]], [[reg3]] : i128
mult902_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult903_test
REAL(2) FUNCTION mult903_test(x0, x1)
INTEGER(8) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.mulf [[reg2]], [[reg3]] : f16
mult903_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult904_test
REAL(4) FUNCTION mult904_test(x0, x1)
INTEGER(8) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.mulf [[reg2]], [[reg3]] : f32
mult904_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult905_test
REAL(8) FUNCTION mult905_test(x0, x1)
INTEGER(8) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.mulf [[reg2]], [[reg3]] : f64
mult905_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult906_test
REAL(10) FUNCTION mult906_test(x0, x1)
INTEGER(8) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.mulf [[reg2]], [[reg3]] : !fir.real<10>
mult906_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult907_test
REAL(16) FUNCTION mult907_test(x0, x1)
INTEGER(8) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.mulf [[reg2]], [[reg3]] : !fir.real<16>
mult907_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult908_test
INTEGER(16) FUNCTION mult908_test(x0, x1)
INTEGER(16) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i128
! CHECK:muli [[reg1]], [[reg3]] : i128
mult908_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult909_test
INTEGER(16) FUNCTION mult909_test(x0, x1)
INTEGER(16) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i128
! CHECK:muli [[reg1]], [[reg3]] : i128
mult909_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult910_test
INTEGER(16) FUNCTION mult910_test(x0, x1)
INTEGER(16) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i128
! CHECK:muli [[reg1]], [[reg3]] : i128
mult910_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult911_test
INTEGER(16) FUNCTION mult911_test(x0, x1)
INTEGER(16) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i128
! CHECK:muli [[reg1]], [[reg3]] : i128
mult911_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult912_test
INTEGER(16) FUNCTION mult912_test(x0, x1)
INTEGER(16) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:muli [[reg1]], [[reg2]] : i128
mult912_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult913_test
REAL(2) FUNCTION mult913_test(x0, x1)
INTEGER(16) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.mulf [[reg2]], [[reg3]] : f16
mult913_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult914_test
REAL(4) FUNCTION mult914_test(x0, x1)
INTEGER(16) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.mulf [[reg2]], [[reg3]] : f32
mult914_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult915_test
REAL(8) FUNCTION mult915_test(x0, x1)
INTEGER(16) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.mulf [[reg2]], [[reg3]] : f64
mult915_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult916_test
REAL(10) FUNCTION mult916_test(x0, x1)
INTEGER(16) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.mulf [[reg2]], [[reg3]] : !fir.real<10>
mult916_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult917_test
REAL(16) FUNCTION mult917_test(x0, x1)
INTEGER(16) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.mulf [[reg2]], [[reg3]] : !fir.real<16>
mult917_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult918_test
REAL(2) FUNCTION mult918_test(x0, x1)
REAL(2) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:fir.mulf [[reg1]], [[reg3]] : f16
mult918_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult919_test
REAL(2) FUNCTION mult919_test(x0, x1)
REAL(2) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:fir.mulf [[reg1]], [[reg3]] : f16
mult919_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult920_test
REAL(2) FUNCTION mult920_test(x0, x1)
REAL(2) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:fir.mulf [[reg1]], [[reg3]] : f16
mult920_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult921_test
REAL(2) FUNCTION mult921_test(x0, x1)
REAL(2) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:fir.mulf [[reg1]], [[reg3]] : f16
mult921_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult922_test
REAL(2) FUNCTION mult922_test(x0, x1)
REAL(2) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:fir.mulf [[reg1]], [[reg3]] : f16
mult922_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult923_test
REAL(2) FUNCTION mult923_test(x0, x1)
REAL(2) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:fir.mulf [[reg1]], [[reg2]] : f16
mult923_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult924_test
REAL(4) FUNCTION mult924_test(x0, x1)
REAL(2) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.mulf [[reg2]], [[reg3]] : f32
mult924_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult925_test
REAL(8) FUNCTION mult925_test(x0, x1)
REAL(2) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.mulf [[reg2]], [[reg3]] : f64
mult925_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult926_test
REAL(10) FUNCTION mult926_test(x0, x1)
REAL(2) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.mulf [[reg2]], [[reg3]] : !fir.real<10>
mult926_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult927_test
REAL(16) FUNCTION mult927_test(x0, x1)
REAL(2) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.mulf [[reg2]], [[reg3]] : !fir.real<16>
mult927_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult928_test
REAL(4) FUNCTION mult928_test(x0, x1)
REAL(4) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:fir.mulf [[reg1]], [[reg3]] : f32
mult928_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult929_test
REAL(4) FUNCTION mult929_test(x0, x1)
REAL(4) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:fir.mulf [[reg1]], [[reg3]] : f32
mult929_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult930_test
REAL(4) FUNCTION mult930_test(x0, x1)
REAL(4) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:fir.mulf [[reg1]], [[reg3]] : f32
mult930_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult931_test
REAL(4) FUNCTION mult931_test(x0, x1)
REAL(4) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:fir.mulf [[reg1]], [[reg3]] : f32
mult931_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult932_test
REAL(4) FUNCTION mult932_test(x0, x1)
REAL(4) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:fir.mulf [[reg1]], [[reg3]] : f32
mult932_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult933_test
REAL(4) FUNCTION mult933_test(x0, x1)
REAL(4) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:fir.mulf [[reg1]], [[reg3]] : f32
mult933_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult934_test
REAL(4) FUNCTION mult934_test(x0, x1)
REAL(4) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:fir.mulf [[reg1]], [[reg2]] : f32
mult934_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult935_test
REAL(8) FUNCTION mult935_test(x0, x1)
REAL(4) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.mulf [[reg2]], [[reg3]] : f64
mult935_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult936_test
REAL(10) FUNCTION mult936_test(x0, x1)
REAL(4) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.mulf [[reg2]], [[reg3]] : !fir.real<10>
mult936_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult937_test
REAL(16) FUNCTION mult937_test(x0, x1)
REAL(4) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.mulf [[reg2]], [[reg3]] : !fir.real<16>
mult937_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult938_test
REAL(8) FUNCTION mult938_test(x0, x1)
REAL(8) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:fir.mulf [[reg1]], [[reg3]] : f64
mult938_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult939_test
REAL(8) FUNCTION mult939_test(x0, x1)
REAL(8) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:fir.mulf [[reg1]], [[reg3]] : f64
mult939_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult940_test
REAL(8) FUNCTION mult940_test(x0, x1)
REAL(8) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:fir.mulf [[reg1]], [[reg3]] : f64
mult940_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult941_test
REAL(8) FUNCTION mult941_test(x0, x1)
REAL(8) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:fir.mulf [[reg1]], [[reg3]] : f64
mult941_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult942_test
REAL(8) FUNCTION mult942_test(x0, x1)
REAL(8) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:fir.mulf [[reg1]], [[reg3]] : f64
mult942_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult943_test
REAL(8) FUNCTION mult943_test(x0, x1)
REAL(8) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:fir.mulf [[reg1]], [[reg3]] : f64
mult943_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult944_test
REAL(8) FUNCTION mult944_test(x0, x1)
REAL(8) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:fir.mulf [[reg1]], [[reg3]] : f64
mult944_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult945_test
REAL(8) FUNCTION mult945_test(x0, x1)
REAL(8) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:fir.mulf [[reg1]], [[reg2]] : f64
mult945_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult946_test
REAL(10) FUNCTION mult946_test(x0, x1)
REAL(8) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.mulf [[reg2]], [[reg3]] : !fir.real<10>
mult946_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult947_test
REAL(16) FUNCTION mult947_test(x0, x1)
REAL(8) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.mulf [[reg2]], [[reg3]] : !fir.real<16>
mult947_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult948_test
REAL(10) FUNCTION mult948_test(x0, x1)
REAL(10) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:fir.mulf [[reg1]], [[reg3]] : !fir.real<10>
mult948_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult949_test
REAL(10) FUNCTION mult949_test(x0, x1)
REAL(10) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:fir.mulf [[reg1]], [[reg3]] : !fir.real<10>
mult949_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult950_test
REAL(10) FUNCTION mult950_test(x0, x1)
REAL(10) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:fir.mulf [[reg1]], [[reg3]] : !fir.real<10>
mult950_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult951_test
REAL(10) FUNCTION mult951_test(x0, x1)
REAL(10) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:fir.mulf [[reg1]], [[reg3]] : !fir.real<10>
mult951_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult952_test
REAL(10) FUNCTION mult952_test(x0, x1)
REAL(10) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:fir.mulf [[reg1]], [[reg3]] : !fir.real<10>
mult952_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult953_test
REAL(10) FUNCTION mult953_test(x0, x1)
REAL(10) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:fir.mulf [[reg1]], [[reg3]] : !fir.real<10>
mult953_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult954_test
REAL(10) FUNCTION mult954_test(x0, x1)
REAL(10) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:fir.mulf [[reg1]], [[reg3]] : !fir.real<10>
mult954_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult955_test
REAL(10) FUNCTION mult955_test(x0, x1)
REAL(10) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:fir.mulf [[reg1]], [[reg3]] : !fir.real<10>
mult955_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult956_test
REAL(10) FUNCTION mult956_test(x0, x1)
REAL(10) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:fir.mulf [[reg1]], [[reg2]] : !fir.real<10>
mult956_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult957_test
REAL(16) FUNCTION mult957_test(x0, x1)
REAL(10) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.mulf [[reg2]], [[reg3]] : !fir.real<16>
mult957_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult958_test
REAL(16) FUNCTION mult958_test(x0, x1)
REAL(16) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:fir.mulf [[reg1]], [[reg3]] : !fir.real<16>
mult958_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult959_test
REAL(16) FUNCTION mult959_test(x0, x1)
REAL(16) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:fir.mulf [[reg1]], [[reg3]] : !fir.real<16>
mult959_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult960_test
REAL(16) FUNCTION mult960_test(x0, x1)
REAL(16) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:fir.mulf [[reg1]], [[reg3]] : !fir.real<16>
mult960_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult961_test
REAL(16) FUNCTION mult961_test(x0, x1)
REAL(16) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:fir.mulf [[reg1]], [[reg3]] : !fir.real<16>
mult961_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult962_test
REAL(16) FUNCTION mult962_test(x0, x1)
REAL(16) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:fir.mulf [[reg1]], [[reg3]] : !fir.real<16>
mult962_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult963_test
REAL(16) FUNCTION mult963_test(x0, x1)
REAL(16) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:fir.mulf [[reg1]], [[reg3]] : !fir.real<16>
mult963_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult964_test
REAL(16) FUNCTION mult964_test(x0, x1)
REAL(16) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:fir.mulf [[reg1]], [[reg3]] : !fir.real<16>
mult964_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult965_test
REAL(16) FUNCTION mult965_test(x0, x1)
REAL(16) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:fir.mulf [[reg1]], [[reg3]] : !fir.real<16>
mult965_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult966_test
REAL(16) FUNCTION mult966_test(x0, x1)
REAL(16) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:fir.mulf [[reg1]], [[reg3]] : !fir.real<16>
mult966_test = x0 * x1
END FUNCTION

! CHECK-LABEL:mult967_test
REAL(16) FUNCTION mult967_test(x0, x1)
REAL(16) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:fir.mulf [[reg1]], [[reg2]] : !fir.real<16>
mult967_test = x0 * x1
END FUNCTION

! CHECK-LABEL:div968_test
INTEGER(1) FUNCTION div968_test(x0, x1)
INTEGER(1) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:divi_signed [[reg1]], [[reg2]] : i8
div968_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div969_test
INTEGER(2) FUNCTION div969_test(x0, x1)
INTEGER(1) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:divi_signed [[reg2]], [[reg3]] : i16
div969_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div970_test
INTEGER(4) FUNCTION div970_test(x0, x1)
INTEGER(1) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:divi_signed [[reg2]], [[reg3]] : i32
div970_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div971_test
INTEGER(8) FUNCTION div971_test(x0, x1)
INTEGER(1) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:divi_signed [[reg2]], [[reg3]] : i64
div971_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div972_test
INTEGER(16) FUNCTION div972_test(x0, x1)
INTEGER(1) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i128
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:divi_signed [[reg2]], [[reg3]] : i128
div972_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div973_test
REAL(2) FUNCTION div973_test(x0, x1)
INTEGER(1) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.divf [[reg2]], [[reg3]] : f16
div973_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div974_test
REAL(4) FUNCTION div974_test(x0, x1)
INTEGER(1) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.divf [[reg2]], [[reg3]] : f32
div974_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div975_test
REAL(8) FUNCTION div975_test(x0, x1)
INTEGER(1) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.divf [[reg2]], [[reg3]] : f64
div975_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div976_test
REAL(10) FUNCTION div976_test(x0, x1)
INTEGER(1) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.divf [[reg2]], [[reg3]] : !fir.real<10>
div976_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div977_test
REAL(16) FUNCTION div977_test(x0, x1)
INTEGER(1) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.divf [[reg2]], [[reg3]] : !fir.real<16>
div977_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div978_test
INTEGER(2) FUNCTION div978_test(x0, x1)
INTEGER(2) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i16
! CHECK:divi_signed [[reg1]], [[reg3]] : i16
div978_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div979_test
INTEGER(2) FUNCTION div979_test(x0, x1)
INTEGER(2) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:divi_signed [[reg1]], [[reg2]] : i16
div979_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div980_test
INTEGER(4) FUNCTION div980_test(x0, x1)
INTEGER(2) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:divi_signed [[reg2]], [[reg3]] : i32
div980_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div981_test
INTEGER(8) FUNCTION div981_test(x0, x1)
INTEGER(2) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:divi_signed [[reg2]], [[reg3]] : i64
div981_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div982_test
INTEGER(16) FUNCTION div982_test(x0, x1)
INTEGER(2) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i128
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:divi_signed [[reg2]], [[reg3]] : i128
div982_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div983_test
REAL(2) FUNCTION div983_test(x0, x1)
INTEGER(2) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.divf [[reg2]], [[reg3]] : f16
div983_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div984_test
REAL(4) FUNCTION div984_test(x0, x1)
INTEGER(2) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.divf [[reg2]], [[reg3]] : f32
div984_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div985_test
REAL(8) FUNCTION div985_test(x0, x1)
INTEGER(2) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.divf [[reg2]], [[reg3]] : f64
div985_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div986_test
REAL(10) FUNCTION div986_test(x0, x1)
INTEGER(2) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.divf [[reg2]], [[reg3]] : !fir.real<10>
div986_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div987_test
REAL(16) FUNCTION div987_test(x0, x1)
INTEGER(2) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.divf [[reg2]], [[reg3]] : !fir.real<16>
div987_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div988_test
INTEGER(4) FUNCTION div988_test(x0, x1)
INTEGER(4) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i32
! CHECK:divi_signed [[reg1]], [[reg3]] : i32
div988_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div989_test
INTEGER(4) FUNCTION div989_test(x0, x1)
INTEGER(4) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i32
! CHECK:divi_signed [[reg1]], [[reg3]] : i32
div989_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div990_test
INTEGER(4) FUNCTION div990_test(x0, x1)
INTEGER(4) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:divi_signed [[reg1]], [[reg2]] : i32
div990_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div991_test
INTEGER(8) FUNCTION div991_test(x0, x1)
INTEGER(4) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:divi_signed [[reg2]], [[reg3]] : i64
div991_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div992_test
INTEGER(16) FUNCTION div992_test(x0, x1)
INTEGER(4) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i128
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:divi_signed [[reg2]], [[reg3]] : i128
div992_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div993_test
REAL(2) FUNCTION div993_test(x0, x1)
INTEGER(4) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.divf [[reg2]], [[reg3]] : f16
div993_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div994_test
REAL(4) FUNCTION div994_test(x0, x1)
INTEGER(4) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.divf [[reg2]], [[reg3]] : f32
div994_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div995_test
REAL(8) FUNCTION div995_test(x0, x1)
INTEGER(4) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.divf [[reg2]], [[reg3]] : f64
div995_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div996_test
REAL(10) FUNCTION div996_test(x0, x1)
INTEGER(4) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.divf [[reg2]], [[reg3]] : !fir.real<10>
div996_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div997_test
REAL(16) FUNCTION div997_test(x0, x1)
INTEGER(4) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.divf [[reg2]], [[reg3]] : !fir.real<16>
div997_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div998_test
INTEGER(8) FUNCTION div998_test(x0, x1)
INTEGER(8) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i64
! CHECK:divi_signed [[reg1]], [[reg3]] : i64
div998_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div999_test
INTEGER(8) FUNCTION div999_test(x0, x1)
INTEGER(8) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i64
! CHECK:divi_signed [[reg1]], [[reg3]] : i64
div999_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1000_test
INTEGER(8) FUNCTION div1000_test(x0, x1)
INTEGER(8) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i64
! CHECK:divi_signed [[reg1]], [[reg3]] : i64
div1000_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1001_test
INTEGER(8) FUNCTION div1001_test(x0, x1)
INTEGER(8) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:divi_signed [[reg1]], [[reg2]] : i64
div1001_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1002_test
INTEGER(16) FUNCTION div1002_test(x0, x1)
INTEGER(8) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> i128
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:divi_signed [[reg2]], [[reg3]] : i128
div1002_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1003_test
REAL(2) FUNCTION div1003_test(x0, x1)
INTEGER(8) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.divf [[reg2]], [[reg3]] : f16
div1003_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1004_test
REAL(4) FUNCTION div1004_test(x0, x1)
INTEGER(8) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.divf [[reg2]], [[reg3]] : f32
div1004_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1005_test
REAL(8) FUNCTION div1005_test(x0, x1)
INTEGER(8) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.divf [[reg2]], [[reg3]] : f64
div1005_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1006_test
REAL(10) FUNCTION div1006_test(x0, x1)
INTEGER(8) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.divf [[reg2]], [[reg3]] : !fir.real<10>
div1006_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1007_test
REAL(16) FUNCTION div1007_test(x0, x1)
INTEGER(8) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.divf [[reg2]], [[reg3]] : !fir.real<16>
div1007_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1008_test
INTEGER(16) FUNCTION div1008_test(x0, x1)
INTEGER(16) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i128
! CHECK:divi_signed [[reg1]], [[reg3]] : i128
div1008_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1009_test
INTEGER(16) FUNCTION div1009_test(x0, x1)
INTEGER(16) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i128
! CHECK:divi_signed [[reg1]], [[reg3]] : i128
div1009_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1010_test
INTEGER(16) FUNCTION div1010_test(x0, x1)
INTEGER(16) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i128
! CHECK:divi_signed [[reg1]], [[reg3]] : i128
div1010_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1011_test
INTEGER(16) FUNCTION div1011_test(x0, x1)
INTEGER(16) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> i128
! CHECK:divi_signed [[reg1]], [[reg3]] : i128
div1011_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1012_test
INTEGER(16) FUNCTION div1012_test(x0, x1)
INTEGER(16) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:divi_signed [[reg1]], [[reg2]] : i128
div1012_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1013_test
REAL(2) FUNCTION div1013_test(x0, x1)
INTEGER(16) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f16
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.divf [[reg2]], [[reg3]] : f16
div1013_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1014_test
REAL(4) FUNCTION div1014_test(x0, x1)
INTEGER(16) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.divf [[reg2]], [[reg3]] : f32
div1014_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1015_test
REAL(8) FUNCTION div1015_test(x0, x1)
INTEGER(16) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.divf [[reg2]], [[reg3]] : f64
div1015_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1016_test
REAL(10) FUNCTION div1016_test(x0, x1)
INTEGER(16) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.divf [[reg2]], [[reg3]] : !fir.real<10>
div1016_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1017_test
REAL(16) FUNCTION div1017_test(x0, x1)
INTEGER(16) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.divf [[reg2]], [[reg3]] : !fir.real<16>
div1017_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1018_test
REAL(2) FUNCTION div1018_test(x0, x1)
REAL(2) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:fir.divf [[reg1]], [[reg3]] : f16
div1018_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1019_test
REAL(2) FUNCTION div1019_test(x0, x1)
REAL(2) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:fir.divf [[reg1]], [[reg3]] : f16
div1019_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1020_test
REAL(2) FUNCTION div1020_test(x0, x1)
REAL(2) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:fir.divf [[reg1]], [[reg3]] : f16
div1020_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1021_test
REAL(2) FUNCTION div1021_test(x0, x1)
REAL(2) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:fir.divf [[reg1]], [[reg3]] : f16
div1021_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1022_test
REAL(2) FUNCTION div1022_test(x0, x1)
REAL(2) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f16
! CHECK:fir.divf [[reg1]], [[reg3]] : f16
div1022_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1023_test
REAL(2) FUNCTION div1023_test(x0, x1)
REAL(2) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:fir.divf [[reg1]], [[reg2]] : f16
div1023_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1024_test
REAL(4) FUNCTION div1024_test(x0, x1)
REAL(2) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f32
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.divf [[reg2]], [[reg3]] : f32
div1024_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1025_test
REAL(8) FUNCTION div1025_test(x0, x1)
REAL(2) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.divf [[reg2]], [[reg3]] : f64
div1025_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1026_test
REAL(10) FUNCTION div1026_test(x0, x1)
REAL(2) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.divf [[reg2]], [[reg3]] : !fir.real<10>
div1026_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1027_test
REAL(16) FUNCTION div1027_test(x0, x1)
REAL(2) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.divf [[reg2]], [[reg3]] : !fir.real<16>
div1027_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1028_test
REAL(4) FUNCTION div1028_test(x0, x1)
REAL(4) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:fir.divf [[reg1]], [[reg3]] : f32
div1028_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1029_test
REAL(4) FUNCTION div1029_test(x0, x1)
REAL(4) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:fir.divf [[reg1]], [[reg3]] : f32
div1029_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1030_test
REAL(4) FUNCTION div1030_test(x0, x1)
REAL(4) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:fir.divf [[reg1]], [[reg3]] : f32
div1030_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1031_test
REAL(4) FUNCTION div1031_test(x0, x1)
REAL(4) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:fir.divf [[reg1]], [[reg3]] : f32
div1031_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1032_test
REAL(4) FUNCTION div1032_test(x0, x1)
REAL(4) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:fir.divf [[reg1]], [[reg3]] : f32
div1032_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1033_test
REAL(4) FUNCTION div1033_test(x0, x1)
REAL(4) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f32
! CHECK:fir.divf [[reg1]], [[reg3]] : f32
div1033_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1034_test
REAL(4) FUNCTION div1034_test(x0, x1)
REAL(4) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:fir.divf [[reg1]], [[reg2]] : f32
div1034_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1035_test
REAL(8) FUNCTION div1035_test(x0, x1)
REAL(4) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> f64
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.divf [[reg2]], [[reg3]] : f64
div1035_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1036_test
REAL(10) FUNCTION div1036_test(x0, x1)
REAL(4) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.divf [[reg2]], [[reg3]] : !fir.real<10>
div1036_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1037_test
REAL(16) FUNCTION div1037_test(x0, x1)
REAL(4) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.divf [[reg2]], [[reg3]] : !fir.real<16>
div1037_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1038_test
REAL(8) FUNCTION div1038_test(x0, x1)
REAL(8) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:fir.divf [[reg1]], [[reg3]] : f64
div1038_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1039_test
REAL(8) FUNCTION div1039_test(x0, x1)
REAL(8) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:fir.divf [[reg1]], [[reg3]] : f64
div1039_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1040_test
REAL(8) FUNCTION div1040_test(x0, x1)
REAL(8) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:fir.divf [[reg1]], [[reg3]] : f64
div1040_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1041_test
REAL(8) FUNCTION div1041_test(x0, x1)
REAL(8) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:fir.divf [[reg1]], [[reg3]] : f64
div1041_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1042_test
REAL(8) FUNCTION div1042_test(x0, x1)
REAL(8) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:fir.divf [[reg1]], [[reg3]] : f64
div1042_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1043_test
REAL(8) FUNCTION div1043_test(x0, x1)
REAL(8) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:fir.divf [[reg1]], [[reg3]] : f64
div1043_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1044_test
REAL(8) FUNCTION div1044_test(x0, x1)
REAL(8) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> f64
! CHECK:fir.divf [[reg1]], [[reg3]] : f64
div1044_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1045_test
REAL(8) FUNCTION div1045_test(x0, x1)
REAL(8) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:fir.divf [[reg1]], [[reg2]] : f64
div1045_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1046_test
REAL(10) FUNCTION div1046_test(x0, x1)
REAL(8) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<10>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.divf [[reg2]], [[reg3]] : !fir.real<10>
div1046_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1047_test
REAL(16) FUNCTION div1047_test(x0, x1)
REAL(8) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.divf [[reg2]], [[reg3]] : !fir.real<16>
div1047_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1048_test
REAL(10) FUNCTION div1048_test(x0, x1)
REAL(10) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:fir.divf [[reg1]], [[reg3]] : !fir.real<10>
div1048_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1049_test
REAL(10) FUNCTION div1049_test(x0, x1)
REAL(10) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:fir.divf [[reg1]], [[reg3]] : !fir.real<10>
div1049_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1050_test
REAL(10) FUNCTION div1050_test(x0, x1)
REAL(10) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:fir.divf [[reg1]], [[reg3]] : !fir.real<10>
div1050_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1051_test
REAL(10) FUNCTION div1051_test(x0, x1)
REAL(10) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:fir.divf [[reg1]], [[reg3]] : !fir.real<10>
div1051_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1052_test
REAL(10) FUNCTION div1052_test(x0, x1)
REAL(10) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:fir.divf [[reg1]], [[reg3]] : !fir.real<10>
div1052_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1053_test
REAL(10) FUNCTION div1053_test(x0, x1)
REAL(10) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:fir.divf [[reg1]], [[reg3]] : !fir.real<10>
div1053_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1054_test
REAL(10) FUNCTION div1054_test(x0, x1)
REAL(10) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:fir.divf [[reg1]], [[reg3]] : !fir.real<10>
div1054_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1055_test
REAL(10) FUNCTION div1055_test(x0, x1)
REAL(10) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<10>
! CHECK:fir.divf [[reg1]], [[reg3]] : !fir.real<10>
div1055_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1056_test
REAL(10) FUNCTION div1056_test(x0, x1)
REAL(10) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:fir.divf [[reg1]], [[reg2]] : !fir.real<10>
div1056_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1057_test
REAL(16) FUNCTION div1057_test(x0, x1)
REAL(10) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.convert [[reg1]] {{.*}} -> !fir.real<16>
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.load %arg1
! CHECK:fir.divf [[reg2]], [[reg3]] : !fir.real<16>
div1057_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1058_test
REAL(16) FUNCTION div1058_test(x0, x1)
REAL(16) :: x0
INTEGER(1) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:fir.divf [[reg1]], [[reg3]] : !fir.real<16>
div1058_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1059_test
REAL(16) FUNCTION div1059_test(x0, x1)
REAL(16) :: x0
INTEGER(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:fir.divf [[reg1]], [[reg3]] : !fir.real<16>
div1059_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1060_test
REAL(16) FUNCTION div1060_test(x0, x1)
REAL(16) :: x0
INTEGER(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:fir.divf [[reg1]], [[reg3]] : !fir.real<16>
div1060_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1061_test
REAL(16) FUNCTION div1061_test(x0, x1)
REAL(16) :: x0
INTEGER(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:fir.divf [[reg1]], [[reg3]] : !fir.real<16>
div1061_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1062_test
REAL(16) FUNCTION div1062_test(x0, x1)
REAL(16) :: x0
INTEGER(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:fir.divf [[reg1]], [[reg3]] : !fir.real<16>
div1062_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1063_test
REAL(16) FUNCTION div1063_test(x0, x1)
REAL(16) :: x0
REAL(2) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:fir.divf [[reg1]], [[reg3]] : !fir.real<16>
div1063_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1064_test
REAL(16) FUNCTION div1064_test(x0, x1)
REAL(16) :: x0
REAL(4) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:fir.divf [[reg1]], [[reg3]] : !fir.real<16>
div1064_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1065_test
REAL(16) FUNCTION div1065_test(x0, x1)
REAL(16) :: x0
REAL(8) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:fir.divf [[reg1]], [[reg3]] : !fir.real<16>
div1065_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1066_test
REAL(16) FUNCTION div1066_test(x0, x1)
REAL(16) :: x0
REAL(10) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK-DAG:[[reg3:%[0-9]+]] = fir.convert [[reg2]] {{.*}} -> !fir.real<16>
! CHECK:fir.divf [[reg1]], [[reg3]] : !fir.real<16>
div1066_test = x0 / x1
END FUNCTION

! CHECK-LABEL:div1067_test
REAL(16) FUNCTION div1067_test(x0, x1)
REAL(16) :: x0
REAL(16) :: x1
! CHECK-DAG:[[reg1:%[0-9]+]] = fir.load %arg0
! CHECK-DAG:[[reg2:%[0-9]+]] = fir.load %arg1
! CHECK:fir.divf [[reg1]], [[reg2]] : !fir.real<16>
div1067_test = x0 / x1
END FUNCTION

! End of generated test file
