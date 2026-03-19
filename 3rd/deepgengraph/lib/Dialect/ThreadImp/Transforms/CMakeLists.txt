add_deepgengraph_library(ThreadImpTransforms
    DeepgengraphToThreadImp.cpp
  DEPENDS
    ThreadImpTransformsIncGen

  LINK_LIBS PUBLIC
    MLIRPass
    MLIRTransformUtils
    DeepgengraphIR
    
    ThreadImpIR
)