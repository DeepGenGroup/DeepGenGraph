add_deepgengraph_library(ThreadImpTransforms
  CalculateOpConversion.cpp
  MemOpConversion.cpp
  InlineDeviceKernel.cpp
    
  DEPENDS
    ThreadImpTransformsIncGen

  LINK_LIBS PUBLIC
    MLIRPass
    MLIRTransformUtils
    DeepgengraphIR
    
    ThreadImpIR
)