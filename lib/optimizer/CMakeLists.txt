set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error -Wno-unused-parameter")

add_subdirectory(Transforms)

add_llvm_library(FortranOptimizer
  CodeGen.cpp
  IteratedDominanceFrontier.cpp
  StdConverter.cpp
)

target_link_libraries(FortranOptimizer
  FIR
)

install (TARGETS FortranOptimizer
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)
