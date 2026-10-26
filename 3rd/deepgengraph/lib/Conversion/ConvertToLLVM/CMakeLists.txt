add_deepgengraph_library(ConvertThreadToLLVMIR
  TheadlevelIRLegalize.cpp
  GPUToNvvmRocdl.cpp
  TranslateToLLVM.cpp
  LowerPipe.cpp
  LowerFriskAsyncCopy.cpp
  CoalesceWaitsAndBarriers.cpp
  IROptimize.cpp
  FuseFragmentAccumulator.cpp
  FuseMMACAccumulator.cpp
  MaskAddBypass.cpp
  LocalReduction.cpp
  DeferLaneSum.cpp
  CanonicalizeLDSGEPs.cpp
  OptimizeRegisterLifetimes.cpp
  OptimizeRegisterMath.cpp
  PrefetchGlobalPackets.cpp
  PrefetchMMACFragments.cpp
  ReuseLDSFragments.cpp

  DEPENDS
  ThreadToLLVMPasseIncGen


  LINK_LIBS PUBLIC
  DeepgengraphAnalysis
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRAMDGPUDialect
  MLIRROCDLDialect
  MLIRLLVMCommonConversion
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRNVVMToLLVMIRTranslation
  MLIRROCDLToLLVMIRTranslation
  MLIRTargetLLVMIRExport
  FriskIR 
)
