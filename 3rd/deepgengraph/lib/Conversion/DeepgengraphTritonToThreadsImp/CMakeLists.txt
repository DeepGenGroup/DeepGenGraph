add_deepgengraph_library(DeepgengraphTritonToThreadsImp
  DeepgengraphTritonToThreadsImpPass.cpp

  DEPENDS
  DeepgengraphTritonToThreadsImpPassIncgen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  DeepgengraphIR 
  DeepgengraphTritonIR
)