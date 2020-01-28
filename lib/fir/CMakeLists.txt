set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error -Wno-unused-parameter")

add_subdirectory(Dialect)

add_llvm_library(FIR
  InternalNames.cpp
  KindMapping.cpp
)

target_link_libraries(FIR
  FIRDialect
)

install (TARGETS FIR
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)
