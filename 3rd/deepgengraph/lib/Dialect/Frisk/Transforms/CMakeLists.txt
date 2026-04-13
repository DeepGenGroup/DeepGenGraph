add_deepgengraph_library(FriskTransforms

    DEPENDS FriskTransformsIncGen


    LINK_LIBS PUBLIC
    MLIRPass
    MLIRTransformUtils
    FriskIR
)