add_deepgengraph_library(DeepgengraphAnalysis
  Parallelism.cpp
  ThreadAnalysis.cpp
  LowerInfo.cpp
  HardwareSpecification.cpp
  LivelinessAnalyze.cpp

  DEPENDS
  DeepgengraphTableGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRIR
  MLIRAffineDialect
  MLIRArithDialect
  MLIRGPUDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRVectorDialect
  MLIRViewLikeInterface
  FriskIR
  DeepgengraphIR 
)
