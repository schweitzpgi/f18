set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")

set(LIBS
  FIRDialect
  FIROptimizer
  FIRTransforms
  MLIRAllDialects
  MLIRIR
  MLIRLLVMIR
  MLIRPass
  MLIRStandardToLLVM
  MLIRTransforms
  MLIRAffineToStandard
  MLIRAnalysis
  MLIRLoopToStandard
  MLIREDSC
  MLIRParser
  MLIRStandardToLLVM
  MLIRSupport
  MLIRVectorToLLVM
)

add_llvm_tool(tco tco.cpp)
llvm_update_compile_flags(tco)
target_link_libraries(tco PRIVATE ${LIBS})
