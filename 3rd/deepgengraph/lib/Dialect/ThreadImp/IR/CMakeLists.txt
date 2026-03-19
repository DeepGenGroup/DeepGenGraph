add_deepgengraph_library(ThreadImpIR
  ThreadImpOps.cpp
  ThreadImpTypes.cpp
  
  DEPENDS
  DeepgengraphTritonTableGen
  ThreadImpIncGen
  ThreadImpTransformsIncGen


  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRMathDialect
  MLIRSCFDialect
)


# add_mlir_dialect_library(DeepgengraphIR
#   DeepgengraphDialect.cpp
#   DeepgengraphOps.cpp

#   DEPENDS
#   DeepgengraphTableGen

#   LINK_LIBS PUBLIC
#   MLIRIR
#   MLIRArithDialect
#   MLIRMathDialect
#   MLIRSCFDialect
# )