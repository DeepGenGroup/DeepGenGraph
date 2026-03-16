# set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

# set(LLVM_TARGET_DEFINITIONS DeepgengraphDialect.td)
# mlir_tablegen(DeepgengraphDialect.h.inc -gen-dialect-decls)
# mlir_tablegen(DeepgengraphDialect.cpp.inc -gen-dialect-defs)
# add_mlir_doc(DeepgengraphDialect DeepgengraphDialect Deepgengraph/ -gen-dialect-doc)


# add_mlir_dialect(DeepgengraphOps deepgengraph)
# add_mlir_doc(DeepgengraphDialect DeepgengraphDialect Deepgengraph/ -gen-dialect-doc)
# # add_mlir_doc(DeepgengraphOps DeepgengraphOps Deepgengraph/ -gen-op-doc)


set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS TilelangDialect.td)
mlir_tablegen(TilelangDialect.h.inc -gen-dialect-decls)
mlir_tablegen(TilelangDialect.cpp.inc -gen-dialect-defs)

set(LLVM_TARGET_DEFINITIONS TilelangOps.td)
mlir_tablegen(TilelangOps.h.inc -gen-op-decls)
mlir_tablegen(TilelangOps.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS TilelangTypes.td)
mlir_tablegen(TilelangTypes.h.inc -gen-typedef-decls)
mlir_tablegen(TilelangTypes.cpp.inc -gen-typedef-defs)

set(LLVM_TARGET_DEFINITIONS TilelangAttrDefs.td)
mlir_tablegen(TilelangEnums.h.inc -gen-enum-decls)
mlir_tablegen(TilelangEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(TilelangAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(TilelangAttrs.cpp.inc -gen-attrdef-defs)


set(LLVM_TARGET_DEFINITIONS TilelangOpInterface.td)
mlir_tablegen(TilelangOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(TilelangOpInterfaces.cpp.inc -gen-op-interface-defs)


add_public_tablegen_target(TilelangIncGen)