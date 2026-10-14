add_compile_options(-g)
add_executable(MyTest test_src.cpp)
add_executable(MyTestAsync test_src_async.cpp)
llvm_update_compile_flags(MyTestAsync)
target_link_libraries(MyTestAsync PRIVATE
  MLIRParser MLIRROCDLDialect MLIRLLVMDialect MLIRTargetLLVMIRExport
  MLIRBuiltinToLLVMIRTranslation MLIRLLVMToLLVMIRTranslation
  MLIRROCDLToLLVMIRTranslation)
add_executable(FragmentOptTest fragment_opt_test.cpp)
add_executable(ThreadTilingTest thread_tiling_test.cpp)
add_executable(LegacyLLVMExportTest legacy_llvm_export_test.cpp)
llvm_update_compile_flags(ThreadTilingTest)
llvm_update_compile_flags(LegacyLLVMExportTest)
# add_executable(MyTest friskTest.cpp)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
# get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)
get_property(deepgengraph_libs GLOBAL PROPERTY DEEPGENGRAPH_LIBS)
target_link_libraries(ThreadTilingTest PRIVATE
  MLIRRegisterAllDialects
  ${dialect_libs} ${conversion_libs} ${extension_libs}
  FriskToBase DeepgengraphAnalysis FriskIR DeepgengraphIR
  DeepgengraphTritonIR TilelangIR ConvertThreadToLLVMIR FriskPipeline MLIROptLib)
target_link_libraries(FragmentOptTest PRIVATE
  ${dialect_libs} ${conversion_libs} ${extension_libs}
  ConvertThreadToLLVMIR MLIROptLib)
target_link_libraries(LegacyLLVMExportTest PRIVATE
  ${dialect_libs} ${conversion_libs} ${extension_libs}
  ConvertThreadToLLVMIR LLVMIRReader)

target_link_libraries(MyTest PRIVATE
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
#  ${triton_libs}
  ${deepgengraph_libs}

  # MLIR core
  MLIROptLib
  MLIRPass
  MLIRTransforms
  # LLVM IR 导出/导入核心库
  MLIRTargetLLVMIRExport
  MLIRTargetLLVMIRImport
  # Dialect 到 LLVM IR 的 Translation 库
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMIRToLLVMTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRNVVMToLLVMIRTranslation
  MLIRLLVMIRToNVVMTranslation
  MLIRROCDLToLLVMIRTranslation

  DeepgengraphIR
  DeepgengraphTritonIR
  TilelangIR
  FriskIR
)
