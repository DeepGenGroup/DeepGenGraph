# set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

# set(LLVM_TARGET_DEFINITIONS DeepgengraphDialect.td)
# mlir_tablegen(DeepgengraphDialect.h.inc -gen-dialect-decls)
# mlir_tablegen(DeepgengraphDialect.cpp.inc -gen-dialect-defs)
# add_mlir_doc(DeepgengraphDialect DeepgengraphDialect Deepgengraph/ -gen-dialect-doc)


# add_mlir_dialect(DeepgengraphOps deepgengraph)
# add_mlir_doc(DeepgengraphDialect DeepgengraphDialect Deepgengraph/ -gen-dialect-doc)
# # add_mlir_doc(DeepgengraphOps DeepgengraphOps Deepgengraph/ -gen-op-doc)


set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS ThreadImpDialect.td)
mlir_tablegen(ThreadImpDialect.h.inc -gen-dialect-decls)
mlir_tablegen(ThreadImpDialect.cpp.inc -gen-dialect-defs)

set(LLVM_TARGET_DEFINITIONS ThreadImpOps.td)
mlir_tablegen(ThreadImpOps.h.inc -gen-op-decls)
mlir_tablegen(ThreadImpOps.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS ThreadImpTypes.td)
mlir_tablegen(ThreadImpTypes.h.inc -gen-typedef-decls)
mlir_tablegen(ThreadImpTypes.cpp.inc -gen-typedef-defs)

set(LLVM_TARGET_DEFINITIONS ThreadImpAttrDefs.td)
mlir_tablegen(ThreadImpEnums.h.inc -gen-enum-decls)
mlir_tablegen(ThreadImpEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(ThreadImpAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(ThreadImpAttrs.cpp.inc -gen-attrdef-defs)


set(LLVM_TARGET_DEFINITIONS ThreadImpOpInterface.td)
mlir_tablegen(ThreadImpOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(ThreadImpOpInterfaces.cpp.inc -gen-op-interface-defs)


add_public_tablegen_target(ThreadImpIncGen)