add_deepgengraph_library(FriskIR
  FriskAttributes.cpp
  FriskEnums.cpp
  FriskOps.cpp
  FriskOps_Reduce.cpp
  FriskTypes.cpp
  ../Utils/LayoutUtils.cpp
  ../Utils/Utils.cpp

  DEPENDS
  FriskTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRMathDialect
  MLIRSCFDialect
)
