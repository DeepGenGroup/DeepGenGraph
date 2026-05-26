add_deepgengraph_library(FriskToBase
  FriskToBasePass.cpp
  FriskBaseToTheadlevelIR.cpp
  
  DEPENDS
  FriskToBaseConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  FriskIR 
)