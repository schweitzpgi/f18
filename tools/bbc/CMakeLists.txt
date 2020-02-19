set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")

set(LIBS
  FIRDialect
  FIROptimizer
  FIRTransforms
  MLIRAllDialects
  MLIRLLVMIR
  MLIRAffineToStandard
  FortranCommon
  FortranParser
  FortranEvaluate
  FortranSemantics
  FortranLower
)

add_llvm_tool(bbc bbc.cpp)
llvm_update_compile_flags(bbc)
target_link_libraries(bbc PRIVATE ${LIBS})
