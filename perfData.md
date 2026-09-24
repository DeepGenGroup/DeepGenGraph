# 性能测量结果

baseline = gemm + casual_mask + softmax + gemm
ours = 融合kernel 

## nopipe + O3 (BM=64, BN=32) speedup=0.97
```
{
  "amdhsa.kernels": [
    {
      ".args": [
        {
          ".address_space": "global",
          ".offset": 0,
          ".size": 8,
          ".value_kind": "global_buffer"
        },
        {
          ".address_space": "global",
          ".offset": 8,
          ".size": 8,
          ".value_kind": "global_buffer"
        },
        {
          ".address_space": "global",
          ".offset": 16,
          ".size": 8,
          ".value_kind": "global_buffer"
        },
        {
          ".address_space": "global",
          ".offset": 24,
          ".size": 8,
          ".value_kind": "global_buffer"
        }
      ],
      ".fp64_status": 0,
      ".group_segment_fixed_size": 45824,
      ".kernarg_segment_align": 8,
      ".kernarg_segment_size": 32,
      ".language": "OpenCL C",
      ".language_version": [
        2,
        0
      ],
      ".max_flat_workgroup_size": 128,
      ".name": "Attn_p2",
      ".private_segment_fixed_size": 0,
      ".reqd_workgroup_size": [
        128,
        1,
        1
      ],
      ".sgpr_count": 72,
      ".sgpr_spill_count": 0,
      ".symbol": "Attn_p2.kd",
      ".uniform_work_group_size": 1,
      ".uses_dynamic_stack": false,
      ".vgpr_count": 241,
      ".vgpr_spill_count": 0,
      ".wavefront_size": 64
    }
  ],
  "amdhsa.target": "amdgcn-amd-amdhsa--gfx936",
  "amdhsa.version": [
    1,
    2
  ]
}

RESULT: PASS
========================================
time_our_mid=63.94410705566406, time_base_mid=62.33962821960449, speedup=0.974908104750562x
```



## pipelinesched + O3 (BM=64, BN=16) speedup=1.46
```
{
  "amdhsa.kernels": [
    {
      ".args": [
        {
          ".address_space": "global",
          ".offset": 0,
          ".size": 8,
          ".value_kind": "global_buffer"
        },
        {
          ".address_space": "global",
          ".offset": 8,
          ".size": 8,
          ".value_kind": "global_buffer"
        },
        {
          ".address_space": "global",
          ".offset": 16,
          ".size": 8,
          ".value_kind": "global_buffer"
        },
        {
          ".address_space": "global",
          ".offset": 24,
          ".size": 8,
          ".value_kind": "global_buffer"
        }
      ],
      ".fp64_status": 0,
      ".group_segment_fixed_size": 43776,
      ".kernarg_segment_align": 8,
      ".kernarg_segment_size": 32,
      ".language": "OpenCL C",
      ".language_version": [
        2,
        0
      ],
      ".max_flat_workgroup_size": 128,
      ".name": "Attn_p2",
      ".private_segment_fixed_size": 148,
      ".reqd_workgroup_size": [
        128,
        1,
        1
      ],
      ".sgpr_count": 76,
      ".sgpr_spill_count": 0,
      ".symbol": "Attn_p2.kd",
      ".uniform_work_group_size": 1,
      ".uses_dynamic_stack": false,
      ".vgpr_count": 256,
      ".vgpr_spill_count": 70,
      ".wavefront_size": 64
    }
  ],
  "amdhsa.target": "amdgcn-amd-amdhsa--gfx936",
  "amdhsa.version": [
    1,
    2
  ]
}

RESULT: PASS
========================================
time_our_mid=62.745723724365234, time_base_mid=91.94274520874023, speedup=1.465322889773878x
```

## pipeline-sched + O3 + shm reuse (BM=64,BN=32) speedup=0.44
```
{
  "amdhsa.kernels": [
    {
      ".args": [
        {
          ".address_space": "global",
          ".offset": 0,
          ".size": 8,
          ".value_kind": "global_buffer"
        },
        {
          ".address_space": "global",
          ".offset": 8,
          ".size": 8,
          ".value_kind": "global_buffer"
        },
        {
          ".address_space": "global",
          ".offset": 16,
          ".size": 8,
          ".value_kind": "global_buffer"
        },
        {
          ".address_space": "global",
          ".offset": 24,
          ".size": 8,
          ".value_kind": "global_buffer"
        }
      ],
      ".fp64_status": 0,
      ".group_segment_fixed_size": 53760,
      ".kernarg_segment_align": 8,
      ".kernarg_segment_size": 32,
      ".language": "OpenCL C",
      ".language_version": [
        2,
        0
      ],
      ".max_flat_workgroup_size": 128,
      ".name": "Attn_p2",
      ".private_segment_fixed_size": 524,
      ".reqd_workgroup_size": [
        128,
        1,
        1
      ],
      ".sgpr_count": 76,
      ".sgpr_spill_count": 0,
      ".symbol": "Attn_p2.kd",
      ".uniform_work_group_size": 1,
      ".uses_dynamic_stack": false,
      ".vgpr_count": 256,
      ".vgpr_spill_count": 200,
      ".wavefront_size": 64
    }
  ],
  "amdhsa.target": "amdgcn-amd-amdhsa--gfx936",
  "amdhsa.version": [
    1,
    2
  ]
}

RESULT: PASS
========================================
time_our_mid=51.52022933959961, time_base_mid=22.956395149230957, speedup=0.44558022049770174x
```

## pipeline + O3 BM64 BN32, baseline=sdpa  
```
HSACO (ms): min=49.814209, median=49.840771, p90=49.852127, max=49.857407
PyTorch SDPA (ms): min=1.467039, median=1.478079, p90=1.489136, max=1.493598
time_our_mid=49.84077072143555, time_base_mid=1.4780794978141785, speedup=0.029656032128301042x

```
## no pipe + unroll wmma kloop + O3 , BM64 BN32, base=sdpa
```
HSACO (ms): min=24.746740, median=24.836818, p90=24.892656, max=25.054260
PyTorch SDPA (ms): min=1.463201, median=1.475440, p90=1.484321, max=1.494401
time_our_mid=24.836817741394043, time_base_mid=1.4754400253295898, speedup=0.059405357026498684
相比于不unroll wmma中的loop ，加速效果显著 49->25 ms
{
  "amdhsa.kernels": [
    {
      ".args": [
        {
          ".address_space": "global",
          ".offset": 0,
          ".size": 8,
          ".value_kind": "global_buffer"
        },
        {
          ".address_space": "global",
          ".offset": 8,
          ".size": 8,
          ".value_kind": "global_buffer"
        },
        {
          ".address_space": "global",
          ".offset": 16,
          ".size": 8,
          ".value_kind": "global_buffer"
        },
        {
          ".address_space": "global",
          ".offset": 24,
          ".size": 8,
          ".value_kind": "global_buffer"
        }
      ],
      ".fp64_status": 0,
      ".group_segment_fixed_size": 29184,
      ".kernarg_segment_align": 8,
      ".kernarg_segment_size": 32,
      ".language": "OpenCL C",
      ".language_version": [
        2,
        0
      ],
      ".max_flat_workgroup_size": 128,
      ".name": "Attn_p2",
      ".private_segment_fixed_size": 0,
      ".reqd_workgroup_size": [
        128,
        1,
        1
      ],
      ".sgpr_count": 72,
      ".sgpr_spill_count": 0,
      ".symbol": "Attn_p2.kd",
      ".uniform_work_group_size": 1,
      ".uses_dynamic_stack": false,
      ".vgpr_count": 241,
      ".vgpr_spill_count": 0,
      ".wavefront_size": 64
    }
  ],
  "amdhsa.target": "amdgcn-amd-amdhsa--gfx936",
  "amdhsa.version": [
    1,
    2
  ]
}

```

## unroll-more + nopipe O3 
```
HSACO (ms): min=24.711811, median=24.843408, p90=24.897107, max=24.931808
PyTorch SDPA (ms): min=1.458078, median=1.474158, p90=1.486414, max=1.505116
time_our_mid=24.84340763092041, time_base_mid=1.474157989025116, speedup=0.05933799464733497x
相比于仅unroll wmma中的loop ，加速效果不显著 耗时基本不变
```


## unroll + better shm reuse O3 
```
HSACO (ms): min=13.818686, median=13.852525, p90=13.883103, max=13.916609
PyTorch SDPA (ms): min=1.466077, median=1.476317, p90=1.483693, max=1.485757
time_our_mid=13.852525234222412, time_base_mid=1.476316511631012, speedup=0.10657381861206063x

{
  "amdhsa.kernels": [
    {
      ".args": [
        {
          ".address_space": "global",
          ".offset": 0,
          ".size": 8,
          ".value_kind": "global_buffer"
        },
        {
          ".address_space": "global",
          ".offset": 8,
          ".size": 8,
          ".value_kind": "global_buffer"
        },
        {
          ".address_space": "global",
          ".offset": 16,
          ".size": 8,
          ".value_kind": "global_buffer"
        },
        {
          ".address_space": "global",
          ".offset": 24,
          ".size": 8,
          ".value_kind": "global_buffer"
        }
      ],
      ".fp64_status": 0,
      ".group_segment_fixed_size": 16384,
      ".kernarg_segment_align": 8,
      ".kernarg_segment_size": 32,
      ".language": "OpenCL C",
      ".language_version": [
        2,
        0
      ],
      ".max_flat_workgroup_size": 128,
      ".name": "Attn_p2",
      ".private_segment_fixed_size": 0,
      ".reqd_workgroup_size": [
        128,
        1,
        1
      ],
      ".sgpr_count": 72,
      ".sgpr_spill_count": 0,
      ".symbol": "Attn_p2.kd",
      ".uniform_work_group_size": 1,
      ".uses_dynamic_stack": false,
      ".vgpr_count": 242,
      ".vgpr_spill_count": 0,
      ".wavefront_size": 64
    }
  ],
  "amdhsa.target": "amdgcn-amd-amdhsa--gfx936",
  "amdhsa.version": [
    1,
    2
  ]
}

```
## betterreuse + barrierOpt , O3
```
HSACO (ms): min=13.735995, median=13.780714, p90=13.807450, max=13.825595
PyTorch SDPA (ms): min=1.459520, median=1.472560, p90=1.478448, max=1.487199
time_our_mid=13.78071403503418, time_base_mid=1.4725595116615295, speedup=0.10685654661419561x

```
