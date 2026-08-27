#!/bin/bash
buffer=$1
file=/data2/xsl/DeepGenGraph/3rd/deepgengraph/build/frisk.mlir
op=$2
python affine_buffer_visualizer.py ${file} \
  --buffer ${buffer} \
  --op ${op} \
  --thread 0:1 \
  --set block_id_x=0 \
  --set block_id_y=0 \
  --label-regions \
  --region-ticks-only \
  --output /data2/xsl/DeepGenGraph/layout_$buffer.png \
  # --where iterK=0 \

  # --list 查看访问点