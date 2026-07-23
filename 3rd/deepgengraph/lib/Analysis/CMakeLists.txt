add_deepgengraph_library(DeepgengraphAnalysis
  Parallelism.cpp
  ThreadAnalysis.cpp
  LowerInfo.cpp
  HardwareSpecification.cpp

  DEPENDS
  DeepgengraphTableGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRIR
  DeepgengraphIR 
)