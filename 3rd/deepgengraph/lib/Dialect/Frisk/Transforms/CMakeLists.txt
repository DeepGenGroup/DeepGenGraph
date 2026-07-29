add_deepgengraph_library(FriskTransforms
    DeepgengraphToFrisk.cpp
    DeepgengraphSimplify.cpp
    DgMemspaceAnalyze.cpp
    FriskCanonicalize.cpp
    FriskLayoutInfer.cpp
    overlap.cpp
    
    DEPENDS FriskTransformsIncGen

    LINK_LIBS PUBLIC
    MLIRPass
    MLIRTransformUtils
    DeepgengraphAnalysis
    FriskIR
)
