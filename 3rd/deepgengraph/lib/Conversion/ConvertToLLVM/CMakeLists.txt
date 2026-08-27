add_deepgengraph_library(ConvertThreadToLLVMIR
  TheadlevelIRLegalize.cpp
  GPUToNvvmRocdl.cpp
  TranslateToLLVM.cpp
  LowerPipe.cpp

  DEPENDS
  ThreadToLLVMPasseIncGen


  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRNVVMToLLVMIRTranslation
  MLIRROCDLToLLVMIRTranslation
  MLIRTargetLLVMIRExport
  FriskIR 
)
