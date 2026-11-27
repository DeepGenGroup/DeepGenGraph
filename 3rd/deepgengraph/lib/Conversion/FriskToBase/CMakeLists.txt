add_deepgengraph_library(FriskToBase
  FriskToBasePass.cpp
  ReorderWorkgroups.cpp
  NormalizeAttentionKLayout.cpp
  TileAttention.cpp
  CheckedGlobalIO.cpp
  WmmaLayoutConversion.cpp
  # FriskBaseToTheadlevelIR.cpp
  # FriskBaseToTheadlevelIR_selfTry.cpp
  FriskBaseToTheadlevelIR_plan2.cpp
  FriskFragIRReorder.cpp
  PrefetchFragmentLDS.cpp
  RetainInvariantFragments.cpp
  FragmentMaskAddBypass.cpp
  FragmentDeferLaneSum.cpp
  FragmentLazySoftmaxBase.cpp
  PackSharedMemory.cpp
  PromoteGlobalToSharedAsync.cpp
  SinkWaitBufferReady.cpp
  
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
