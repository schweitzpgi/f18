set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error")

add_library(FortranLower
  ASTBuilder.cpp
)

target_link_libraries(FortranLower
  LLVMSupport
)

install (TARGETS FortranLower
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
