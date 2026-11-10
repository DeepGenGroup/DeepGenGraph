add_deepgengraph_library(FriskPipeline
    PipelineSchedule.cpp
    ScheduleJSON.cpp
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

add_executable(FriskScheduleJSONTest test/schedule_json_test.cpp)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
target_link_libraries(FriskScheduleJSONTest PRIVATE
    ${dialect_libs} MLIRRegisterAllDialects FriskPipeline FriskIR DeepgengraphIR
    MLIRParser MLIRIR MLIRSupport LLVMSupport)

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
