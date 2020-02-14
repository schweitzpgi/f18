// RUN: tco %s | FileCheck %s

// CHECK: @g_i0 = global i32 0
fir.global @g_i0 : i32 {
  constant 0 : i32
}

// CHECK: @g_i2 = global i32 2
fir.global @g_i2 : i32 {
  constant 2 : i32
}

// CHECK: @g_ci5 = constant i32 5
fir.global @g_ci5 constant : i32 {
  constant 5 : i32
}
