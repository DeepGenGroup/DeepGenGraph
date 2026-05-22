add_deepgengraph_library(FriskTransforms
    DeepgengraphToFrisk.cpp
    DeepgengraphSimplify.cpp
    DgCalcOpToFrisk.cpp
    overlap.cpp
    DgDataflowAnalyze.cpp
    
    DEPENDS FriskTransformsIncGen

    LINK_LIBS PUBLIC
    MLIRPass
    MLIRTransformUtils
    FriskIR
)