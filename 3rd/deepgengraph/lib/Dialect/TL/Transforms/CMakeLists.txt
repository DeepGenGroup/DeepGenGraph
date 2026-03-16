add_deepgengraph_library(TilelangTransforms
    DeepgengraphToTilelang.cpp
  DEPENDS
    TilelangTransformsIncGen

  LINK_LIBS PUBLIC
    MLIRPass
    MLIRTransformUtils
    DeepgengraphIR
    TilelangIR
)