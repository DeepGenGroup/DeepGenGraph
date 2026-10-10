add_deepgengraph_library(FriskToBase
  FriskToBasePass.cpp
  NormalizeAttentionKLayout.cpp
  # FriskBaseToTheadlevelIR.cpp
  # FriskBaseToTheadlevelIR_selfTry.cpp
  FriskBaseToTheadlevelIR_plan2.cpp
  FriskFragIRReorder.cpp
  PackSharedMemory.cpp
  
  DEPENDS
  FriskToBaseConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRAffineDialect
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRVectorDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  DeepgengraphAnalysis
  FriskIR 
)
