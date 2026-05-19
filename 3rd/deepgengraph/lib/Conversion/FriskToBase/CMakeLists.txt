add_deepgengraph_library(FriskToBase
  FriskToBasePass.cpp

  DEPENDS
  FriskToBaseConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  FriskIR 
)