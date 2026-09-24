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
