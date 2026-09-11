add_deepgengraph_library(FriskToBase
  FriskToBasePass.cpp
  # FriskBaseToTheadlevelIR.cpp
  FriskBaseToTheadlevelIR_selfTry.cpp
  
  DEPENDS
  FriskToBaseConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  FriskIR 
)