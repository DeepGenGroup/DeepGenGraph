#!/bin/bash

frisk=/data2/xsl/DeepGenGraph/3rd/deepgengraph/build/test/MyTest
input_ir=/data2/xsl/DeepGenGraph/3rd/deepgengraph/test/test_input.mlir
final_ll=/data2/xsl/DeepGenGraph/3rd/deepgengraph/build/final.ll
# /data2/xsl/install/bin/mlir-translate --mlir-to-llvmir /data2/xsl/DeepGenGraph/3rd/deepgengraph/build/log.log

$frisk $input_ir > $final_ll 2>&1 
# python /data2/xsl/DeepGenGraph/3rd/deepgengraph/legalizeLLVMText.py $final_ll
