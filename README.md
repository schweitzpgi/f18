<!--
Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
-->

# FIR

Working branch for FIR development.

## Monorepo

This is quite similar to the old way, but with a few subtle differences.

1. Get the stuff.

```
  git clone git@github.com:flang-compiler/f18-llvm-project.git
  git clone git@github.com:flang-compiler/f18-mlir.git
  git clone git@github.com:schweitzpgi/f18.git 
```

2. Get "on" the right branches.

```
  (cd f18-llvm-project; git checkout master)
  (cd f18-mlir; git checkout f18)
  (cd f18; git checkout f18)
```
             
3. Setup the LLVM space for in-tree builds.
   
``` 
  cd f18-llvm-project/llvm/projects ; ln -s ../../../f18-mlir mlir
  cd f18-llvm-project ; ln -s ../f18 flang
```

4. Create a build space for cmake and make/ninja

```
  mkdir build; cd build; cmake ../f18-llvm-project/llvm -DCMAKE_BUILD_TYPE=Debug -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_ENABLE_PROJECTS=flang -DCMAKE_CXX_STANDARD=17 <other-arguments>
```


## Directions for building with the old repositories...

1. Get the stuff.

```
  git clone http://llvm.org/git/llvm.git
  git clone git@github.com:schweitzpgi/mlir.git
  git clone git@github.com:schweitzpgi/f18.git 
```

2. Get "on" the right branches.

```
  (cd llvm; git checkout master)
  (cd mlir; git checkout f18)
  (cd f18; git checkout f18)
```
             
3. Setup the LLVM space for in-tree builds.
   
``` 
  cd llvm/projects ; ln -s ../../mlir .
  cd llvm/tools ; ln -s ../../f18 flang
```

4. Create a build space for cmake and make/ninja

```
  mkdir build; cd build; cmake ../llvm -DCMAKE_BUILD_TYPE=Debug -DLLVM_TARGETS_TO_BUILD=X86 <other-arguments>
```


