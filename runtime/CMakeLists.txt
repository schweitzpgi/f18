set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DFORTRAN_IN_RUNTIME")

add_library(FortranRuntime
  ISO_Fortran_binding.cpp
  buffer.cpp
  connection.cpp
  derived-type.cpp
  descriptor.cpp
  edit-input.cpp
  edit-output.cpp
  environment.cpp
  file.cpp
  format.cpp
  internal-unit.cpp
  iostat.cpp
  io-api.cpp
  io-error.cpp
  io-stmt.cpp
  main.cpp
  memory.cpp
  stop.cpp
  terminator.cpp
  tools.cpp
  transformational.cpp
  type-code.cpp
  unit.cpp
  unit-map.cpp
)

target_link_libraries(FortranRuntime
  FortranDecimal
)
