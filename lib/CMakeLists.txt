if ((CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
  if (APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-string-conversion -Wno-covered-switch-default")
  endif()
endif()

add_subdirectory(Common)
add_subdirectory(Evaluate)
add_subdirectory(Decimal)
add_subdirectory(Lower)
add_subdirectory(Optimizer)
add_subdirectory(Parser)
add_subdirectory(Semantics)

# Install header files to mirror the directory structure that follows
# aspects of the LLVM/Clang layout (for full, not toolchain,
# installs).
if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  install(DIRECTORY common evaluate decimal parser semantics 
    DESTINATION include/flang
    COMPONENT flang-headers
    FILES_MATCHING
    PATTERN "*.def"
    PATTERN "*.h"
    PATTERN "config.h" EXCLUDE
    PATTERN ".git"     EXCLUDE
    PATTERN ".svn"     EXCLUDE)
endif()

