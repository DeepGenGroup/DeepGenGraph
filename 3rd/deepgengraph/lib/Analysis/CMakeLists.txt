add_deepgengraph_library(DeepgengraphAnalysis
  Parallelism.cpp
  ThreadAnalysis.cpp

  DEPENDS
  DeepgengraphTableGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRIR
  DeepgengraphIR 
)