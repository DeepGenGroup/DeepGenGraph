add_deepgengraph_library(FriskPipeline
    PipelineSchedule.cpp
    AutoPipeline.cpp
    SimplifyLoopControl.cpp

    DEPENDS PipelineIncGen

    LINK_LIBS PUBLIC
    MLIRPass
    MLIRTransformUtils
    MLIRTransforms
    MLIRAffineUtils
    MLIRSCFUtils
    MLIRAffineDialect
    MLIRArithDialect
    MLIRSCFDialect
    MLIRFuncDialect
    MLIRMemRefDialect
    FriskIR
    DeepgengraphAnalysis
)

add_executable(FriskPipelineTest
    test/pipeline_schedule_test.cpp
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

target_link_libraries(FriskPipelineTest PRIVATE
    ${dialect_libs}

    FriskPipeline
    FriskIR

    MLIROptLib
    MLIRPass
    MLIRTransforms
    MLIRParser
    MLIRIR
    MLIRSupport
    LLVMSupport
)
