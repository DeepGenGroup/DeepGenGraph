add_deepgengraph_library(DeepgengraphAnalysis
  Parallelism.cpp
  ThreadAnalysis.cpp
  ThreadBounds.cpp
  LowerInfo.cpp
  ShmDataIODesc.cpp
  HardwareSpecification.cpp
  LivelinessAnalyze.cpp
  RegisterPressure.cpp

  DEPENDS
  DeepgengraphTableGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRPass
  MLIRTransforms
  MLIRDataLayoutInterfaces
  MLIRControlFlowInterfaces
  MLIRFunctionInterfaces
  MLIRLoopLikeInterface
  MLIRLLVMDialect
  MLIRIR
  MLIRAffineDialect
  MLIRArithDialect
  MLIRGPUDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRVectorDialect
  MLIRViewLikeInterface
  FriskIR
  DeepgengraphTritonIR
  DeepgengraphIR 
)
