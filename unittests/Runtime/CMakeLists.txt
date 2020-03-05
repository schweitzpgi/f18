if(CMAKE_COMPILER_IS_GNUCXX OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")
endif()

add_flang_test(format format-test
  FortranRuntime
  RuntimeTesting
  LLVMSupport
)

add_flang_test(hello HelloWorld
  FortranRuntime
  RuntimeTesting
  LLVMSupport
)

add_flang_test(external-hello ExternalHelloWorld
  FortranRuntime
  LLVMSupport
)

add_executable(list-input-test
  list-input.cpp
)

target_link_libraries(list-input-test
  FortranRuntime
  RuntimeTesting
  LLVMSupport
)

add_test(ListInput list-input-test)
