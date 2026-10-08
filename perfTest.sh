#!/bin/bash
# 用于 性能测试
kernelfile=$1
python /data2/xsl/DeepGenGraph/kernelLauncher.py --hsaco ${kernelfile}  --kernel Attn_p2   --grid 32,64,1 --block 128,1,1