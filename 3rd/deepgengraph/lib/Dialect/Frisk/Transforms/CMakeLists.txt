add_deepgengraph_library(FriskTransforms
    DeepgengraphToFrisk.cpp
    DeepgengraphSimplify.cpp
    DgMemspaceAnalyze.cpp
    FriskCanonicalize.cpp
    overlap.cpp
    
    DEPENDS FriskTransformsIncGen

    LINK_LIBS PUBLIC
    MLIRPass
    MLIRTransformUtils
    FriskIR
)