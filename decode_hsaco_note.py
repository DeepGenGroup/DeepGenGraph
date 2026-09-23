import json

raw_hex_data = """
83 ae 61 6d 64 68 73 61 2e 6b 65 72 6e 65 6c 73 91 de 00 11 a5 2e 61 72 67 73 94 84 ae 2e 61 64 64 72 65 73 73 5f 73 70 61 63 65 a6 67 6c 6f 62 61 6c a7 2e 6f 66 66 73 65 74 00 a5 2e 73 69 7a 65 08 ab 2e 76 61 6c 75 65 5f 6b 69 6e 64 ad 67 6c 6f 62 61 6c 5f 62 75 66 66 65 72 84 ae 2e 61 64 64 72 65 73 73 5f 73 70 61 63 65 a6 67 6c 6f 62 61 6c a7 2e 6f 66 66 73 65 74 08 a5 2e 73 69 7a 65 08 ab 2e 76 61 6c 75 65 5f 6b 69 6e 64 ad 67 6c 6f 62 61 6c 5f 62 75 66 66 65 72 84 ae 2e 61 64 64 72 65 73 73 5f 73 70 61 63 65 a6 67 6c 6f 62 61 6c a7 2e 6f 66 66 73 65 74 10 a5 2e 73 69 7a 65 08 ab 2e 76 61 6c 75 65 5f 6b 69 6e 64 ad 67 6c 6f 62 61 6c 5f 62 75 66 66 65 72 84 ae 2e 61 64 64 72 65 73 73 5f 73 70 61 63 65 a6 67 6c 6f 62 61 6c a7 2e 6f 66 66 73 65 74 18 a5 2e 73 69 7a 65 08 ab 2e 76 61 6c 75 65 5f 6b 69 6e 64 ad 67 6c 6f 62 61 6c 5f 62 75 66 66 65 72 ac 2e 66 70 36 34 5f 73 74 61 74 75 73 00 b9 2e 67 72 6f 75 70 5f 73 65 67 6d 65 6e 74 5f 66 69 78 65 64 5f 73 69 7a 65 cd c1 00 b6 2e 6b 65 72 6e 61 72 67 5f 73 65 67 6d 65 6e 74 5f 61 6c 69 67 6e 08 b5 2e 6b 65 72 6e 61 72 67 5f 73 65 67 6d 65 6e 74 5f 73 69 7a 65 20 a9 2e 6c 61 6e 67 75 61 67 65 a8 4f 70 65 6e 43 4c 20 43 b1 2e 6c 61 6e 67 75 61 67 65 5f 76 65 72 73 69 6f 6e 92 02 00 b8 2e 6d 61 78 5f 66 6c 61 74 5f 77 6f 72 6b 67 72 6f 75 70 5f 73 69 7a 65 cc 80 a5 2e 6e 61 6d 65 a7 41 74 74 6e 5f 70 32 bb 2e 70 72 69 76 61 74 65 5f 73 65 67 6d 65 6e 74 5f 66 69 78 65 64 5f 73 69 7a 65 cd 46 98 b4 2e 72 65 71 64 5f 77 6f 72 6b 67 72 6f 75 70 5f 73 69 7a 65 93 cc 80 01 01 ab 2e 73 67 70 72 5f 63 6f 75 6e 74 6a b1 2e 73 67 70 72 5f 73 70 69 6c 6c 5f 63 6f 75 6e 74 cd 02 9c a7 2e 73 79 6d 62 6f 6c aa 41 74 74 6e 5f 70 32 2e 6b 64 ab 2e 76 67 70 72 5f 63 6f 75 6e 74 cc e1 b1 2e 76 67 70 72 5f 73 70 69 6c 6c 5f 63 6f 75 6e 74 cd 0d 79 af 2e 77 61 76 65 66 72 6f 6e 74 5f 73 69 7a 65 40 ad 61 6d 64 68 73 61 2e 74 61 72 67 65 74 b9 61 6d 64 67 63 6e 2d 61 6d 64 2d 61 6d 64 68 73 61 2d 2d 67 66 78 39 33 36 ae 61 6d 64 68 73 61 2e 76 65 72 73 69 6f 6e 92 01 01 
"""  # O0 fragment 优化前
# === 2. MsgPack 还原结构化数据 ===
# {
#   "amdhsa.kernels": [
#     {
#       ".args": [
#         {
#           ".address_space": "global",
#           ".offset": 0,
#           ".size": 8,
#           ".value_kind": "global_buffer"
#         },
#         {
#           ".address_space": "global",
#           ".offset": 8,
#           ".size": 8,
#           ".value_kind": "global_buffer"
#         },
#         {
#           ".address_space": "global",
#           ".offset": 16,
#           ".size": 8,
#           ".value_kind": "global_buffer"
#         },
#         {
#           ".address_space": "global",
#           ".offset": 24,
#           ".size": 8,
#           ".value_kind": "global_buffer"
#         }
#       ],
#       ".fp64_status": 0,
#       ".group_segment_fixed_size": 49408,
#       ".kernarg_segment_align": 8,
#       ".kernarg_segment_size": 32,
#       ".language": "OpenCL C",
#       ".language_version": [
#         2,
#         0
#       ],
#       ".max_flat_workgroup_size": 128,
#       ".name": "Attn_p2",
#       ".private_segment_fixed_size": 18072,
#       ".reqd_workgroup_size": [
#         128,
#         1,
#         1
#       ],
#       ".sgpr_count": 106,
#       ".sgpr_spill_count": 668,
#       ".symbol": "Attn_p2.kd",
#       ".vgpr_count": 225,
#       ".vgpr_spill_count": 3449,
#       ".wavefront_size": 64
#     }
#   ],
#   "amdhsa.target": "amdgcn-amd-amdhsa--gfx936",
#   "amdhsa.version": [
#     1,
#     1
#   ]
# }


raw_hex_data='''
83 ae 61 6d 64 68 73 61 2e 6b 65 72 6e 65 6c 73 91 de 00 11 a5 2e 61 72 67 73 94 84 ae 2e 61 64 64 72 65 73 73 5f 73 70 61 63 65 a6 67 6c 6f 62 61 6c a7 2e 6f 66 66 73 65 74 00 a5 2e 73 69 7a 65 08 ab 2e 76 61 6c 75 65 5f 6b 69 6e 64 ad 67 6c 6f 62 61 6c 5f 62 75 66 66 65 72 84 ae 2e 61 64 64 72 65 73 73 5f 73 70 61 63 65 a6 67 6c 6f 62 61 6c a7 2e 6f 66 66 73 65 74 08 a5 2e 73 69 7a 65 08 ab 2e 76 61 6c 75 65 5f 6b 69 6e 64 ad 67 6c 6f 62 61 6c 5f 62 75 66 66 65 72 84 ae 2e 61 64 64 72 65 73 73 5f 73 70 61 63 65 a6 67 6c 6f 62 61 6c a7 2e 6f 66 66 73 65 74 10 a5 2e 73 69 7a 65 08 ab 2e 76 61 6c 75 65 5f 6b 69 6e 64 ad 67 6c 6f 62 61 6c 5f 62 75 66 66 65 72 84 ae 2e 61 64 64 72 65 73 73 5f 73 70 61 63 65 a6 67 6c 6f 62 61 6c a7 2e 6f 66 66 73 65 74 18 a5 2e 73 69 7a 65 08 ab 2e 76 61 6c 75 65 5f 6b 69 6e 64 ad 67 6c 6f 62 61 6c 5f 62 75 66 66 65 72 ac 2e 66 70 36 34 5f 73 74 61 74 75 73 00 b9 2e 67 72 6f 75 70 5f 73 65 67 6d 65 6e 74 5f 66 69 78 65 64 5f 73 69 7a 65 cd c1 00 b6 2e 6b 65 72 6e 61 72 67 5f 73 65 67 6d 65 6e 74 5f 61 6c 69 67 6e 08 b5 2e 6b 65 72 6e 61 72 67 5f 73 65 67 6d 65 6e 74 5f 73 69 7a 65 20 a9 2e 6c 61 6e 67 75 61 67 65 a8 4f 70 65 6e 43 4c 20 43 b1 2e 6c 61 6e 67 75 61 67 65 5f 76 65 72 73 69 6f 6e 92 02 00 b8 2e 6d 61 78 5f 66 6c 61 74 5f 77 6f 72 6b 67 72 6f 75 70 5f 73 69 7a 65 cc 80 a5 2e 6e 61 6d 65 a7 41 74 74 6e 5f 70 32 bb 2e 70 72 69 76 61 74 65 5f 73 65 67 6d 65 6e 74 5f 66 69 78 65 64 5f 73 69 7a 65 cd 32 28 b4 2e 72 65 71 64 5f 77 6f 72 6b 67 72 6f 75 70 5f 73 69 7a 65 93 cc 80 01 01 ab 2e 73 67 70 72 5f 63 6f 75 6e 74 6c b1 2e 73 67 70 72 5f 73 70 69 6c 6c 5f 63 6f 75 6e 74 cd 01 c5 a7 2e 73 79 6d 62 6f 6c aa 41 74 74 6e 5f 70 32 2e 6b 64 ab 2e 76 67 70 72 5f 63 6f 75 6e 74 cc a6 b1 2e 76 67 70 72 5f 73 70 69 6c 6c 5f 63 6f 75 6e 74 cd 0a 2f af 2e 77 61 76 65 66 72 6f 6e 74 5f 73 69 7a 65 40 ad 61 6d 64 68 73 61 2e 74 61 72 67 65 74 b9 61 6d 64 67 63 6e 2d 61 6d 64 2d 61 6d 64 68 73 61 2d 2d 67 66 78 39 33 36 ae 61 6d 64 68 73 61 2e 76 65 72 73 69 6f 6e 92 01 01 
'''  # O0 fragment优化后
# === 2. MsgPack 还原结构化数据 ===
# {
#   "amdhsa.kernels": [
#     {
#       ".args": [
#         {
#           ".address_space": "global",
#           ".offset": 0,
#           ".size": 8,
#           ".value_kind": "global_buffer"
#         },
#         {
#           ".address_space": "global",
#           ".offset": 8,
#           ".size": 8,
#           ".value_kind": "global_buffer"
#         },
#         {
#           ".address_space": "global",
#           ".offset": 16,
#           ".size": 8,
#           ".value_kind": "global_buffer"
#         },
#         {
#           ".address_space": "global",
#           ".offset": 24,
#           ".size": 8,
#           ".value_kind": "global_buffer"
#         }
#       ],
#       ".fp64_status": 0,
#       ".group_segment_fixed_size": 49408,
#       ".kernarg_segment_align": 8,
#       ".kernarg_segment_size": 32,
#       ".language": "OpenCL C",
#       ".language_version": [
#         2,
#         0
#       ],
#       ".max_flat_workgroup_size": 128,
#       ".name": "Attn_p2",
#       ".private_segment_fixed_size": 12840,
#       ".reqd_workgroup_size": [
#         128,
#         1,
#         1
#       ],
#       ".sgpr_count": 108,
#       ".sgpr_spill_count": 453,
#       ".symbol": "Attn_p2.kd",
#       ".vgpr_count": 166,
#       ".vgpr_spill_count": 2607,
#       ".wavefront_size": 64
#     }
#   ],
#   "amdhsa.target": "amdgcn-amd-amdhsa--gfx936",
#   "amdhsa.version": [
#     1,
#     1
#   ]
# }

raw_hex_data='''
 83 ae 61 6d 64 68 73 61 2e 6b 65 72 6e 65 6c 73 91 de 00 11 a5 2e 61 72 67 73 94 84 ae 2e 61 64 64 72 65 73 73 5f 73 70 61 63 65 a6 67 6c 6f 62 61 6c a7 2e 6f 66 66 73 65 74 00 a5 2e 73 69 7a 65 08 ab 2e 76 61 6c 75 65 5f 6b 69 6e 64 ad 67 6c 6f 62 61 6c 5f 62 75 66 66 65 72 84 ae 2e 61 64 64 72 65 73 73 5f 73 70 61 63 65 a6 67 6c 6f 62 61 6c a7 2e 6f 66 66 73 65 74 08 a5 2e 73 69 7a 65 08 ab 2e 76 61 6c 75 65 5f 6b 69 6e 64 ad 67 6c 6f 62 61 6c 5f 62 75 66 66 65 72 84 ae 2e 61 64 64 72 65 73 73 5f 73 70 61 63 65 a6 67 6c 6f 62 61 6c a7 2e 6f 66 66 73 65 74 10 a5 2e 73 69 7a 65 08 ab 2e 76 61 6c 75 65 5f 6b 69 6e 64 ad 67 6c 6f 62 61 6c 5f 62 75 66 66 65 72 84 ae 2e 61 64 64 72 65 73 73 5f 73 70 61 63 65 a6 67 6c 6f 62 61 6c a7 2e 6f 66 66 73 65 74 18 a5 2e 73 69 7a 65 08 ab 2e 76 61 6c 75 65 5f 6b 69 6e 64 ad 67 6c 6f 62 61 6c 5f 62 75 66 66 65 72 ac 2e 66 70 36 34 5f 73 74 61 74 75 73 00 b9 2e 67 72 6f 75 70 5f 73 65 67 6d 65 6e 74 5f 66 69 78 65 64 5f 73 69 7a 65 cd c1 00 b6 2e 6b 65 72 6e 61 72 67 5f 73 65 67 6d 65 6e 74 5f 61 6c 69 67 6e 08 b5 2e 6b 65 72 6e 61 72 67 5f 73 65 67 6d 65 6e 74 5f 73 69 7a 65 20 a9 2e 6c 61 6e 67 75 61 67 65 a8 4f 70 65 6e 43 4c 20 43 b1 2e 6c 61 6e 67 75 61 67 65 5f 76 65 72 73 69 6f 6e 92 02 00 b8 2e 6d 61 78 5f 66 6c 61 74 5f 77 6f 72 6b 67 72 6f 75 70 5f 73 69 7a 65 cc 80 a5 2e 6e 61 6d 65 a7 41 74 74 6e 5f 70 32 bb 2e 70 72 69 76 61 74 65 5f 73 65 67 6d 65 6e 74 5f 66 69 78 65 64 5f 73 69 7a 65 cc f4 b4 2e 72 65 71 64 5f 77 6f 72 6b 67 72 6f 75 70 5f 73 69 7a 65 93 cc 80 01 01 ab 2e 73 67 70 72 5f 63 6f 75 6e 74 4a b1 2e 73 67 70 72 5f 73 70 69 6c 6c 5f 63 6f 75 6e 74 00 a7 2e 73 79 6d 62 6f 6c aa 41 74 74 6e 5f 70 32 2e 6b 64 ab 2e 76 67 70 72 5f 63 6f 75 6e 74 cd 01 00 b1 2e 76 67 70 72 5f 73 70 69 6c 6c 5f 63 6f 75 6e 74 3c af 2e 77 61 76 65 66 72 6f 6e 74 5f 73 69 7a 65 40 ad 61 6d 64 68 73 61 2e 74 61 72 67 65 74 b9 61 6d 64 67 63 6e 2d 61 6d 64 2d 61 6d 64 68 73 61 2d 2d 67 66 78 39 33 36 ae 61 6d 64 68 73 61 2e 76 65 72 73 69 6f 6e 92 01 01
'''  # O3 fragment优化后

raw_hex_data='''
83 ae 61 6d 64 68 73 61 2e 6b 65 72 6e 65 6c 73 91 de 00 13 a5 2e 61 72 67 73 94 84 ae 2e 61 64 64 72 65 73 73 5f 73 70 61 63 65 a6 67 6c 6f 62 61 6c a7 2e 6f 66 66 73 65 74 00 a5 2e 73 69 7a 65 08 ab 2e 76 61 6c 75 65 5f 6b 69 6e 64 ad 67 6c 6f 62 61 6c 5f 62 75 66 66 65 72 84 ae 2e 61 64 64 72 65 73 73 5f 73 70 61 63 65 a6 67 6c 6f 62 61 6c a7 2e 6f 66 66 73 65 74 08 a5 2e 73 69 7a 65 08 ab 2e 76 61 6c 75 65 5f 6b 69 6e 64 ad 67 6c 6f 62 61 6c 5f 62 75 66 66 65 72 84 ae 2e 61 64 64 72 65 73 73 5f 73 70 61 63 65 a6 67 6c 6f 62 61 6c a7 2e 6f 66 66 73 65 74 10 a5 2e 73 69 7a 65 08 ab 2e 76 61 6c 75 65 5f 6b 69 6e 64 ad 67 6c 6f 62 61 6c 5f 62 75 66 66 65 72 84 ae 2e 61 64 64 72 65 73 73 5f 73 70 61 63 65 a6 67 6c 6f 62 61 6c a7 2e 6f 66 66 73 65 74 18 a5 2e 73 69 7a 65 08 ab 2e 76 61 6c 75 65 5f 6b 69 6e 64 ad 67 6c 6f 62 61 6c 5f 62 75 66 66 65 72 ac 2e 66 70 36 34 5f 73 74 61 74 75 73 00 b9 2e 67 72 6f 75 70 5f 73 65 67 6d 65 6e 74 5f 66 69 78 65 64 5f 73 69 7a 65 cd 92 00 b6 2e 6b 65 72 6e 61 72 67 5f 73 65 67 6d 65 6e 74 5f 61 6c 69 67 6e 08 b5 2e 6b 65 72 6e 61 72 67 5f 73 65 67 6d 65 6e 74 5f 73 69 7a 65 20 a9 2e 6c 61 6e 67 75 61 67 65 a8 4f 70 65 6e 43 4c 20 43 b1 2e 6c 61 6e 67 75 61 67 65 5f 76 65 72 73 69 6f 6e 92 02 00 b8 2e 6d 61 78 5f 66 6c 61 74 5f 77 6f 72 6b 67 72 6f 75 70 5f 73 69 7a 65 cc 80 a5 2e 6e 61 6d 65 a7 41 74 74 6e 5f 70 32 bb 2e 70 72 69 76 61 74 65 5f 73 65 67 6d 65 6e 74 5f 66 69 78 65 64 5f 73 69 7a 65 00 b4 2e 72 65 71 64 5f 77 6f 72 6b 67 72 6f 75 70 5f 73 69 7a 65 93 cc 80 01 01 ab 2e 73 67 70 72 5f 63 6f 75 6e 74 48 b1 2e 73 67 70 72 5f 73 70 69 6c 6c 5f 63 6f 75 6e 74 00 a7 2e 73 79 6d 62 6f 6c aa 41 74 74 6e 5f 70 32 2e 6b 64 b8 2e 75 6e 69 66 6f 72 6d 5f 77 6f 72 6b 5f 67 72 6f 75 70 5f 73 69 7a 65 01 b3 2e 75 73 65 73 5f 64 79 6e 61 6d 69 63 5f 73 74 61 63 6b c2 ab 2e 76 67 70 72 5f 63 6f 75 6e 74 cc f3 b1 2e 76 67 70 72 5f 73 70 69 6c 6c 5f 63 6f 75 6e 74 00 af 2e 77 61 76 65 66 72 6f 6e 74 5f 73 69 7a 65 40 ad 61 6d 64 68 73 61 2e 74 61 72 67 65 74 b9 61 6d 64 67 63 6e 2d 61 6d 64 2d 61 6d 64 68 73 61 2d 2d 67 66 78 39 33 36 ae 61 6d 64 68 73 61 2e 76 65 72 73 69 6f 6e 92 01 02
'''  # 更换 bt->tt pipeline后， spill 消失


# 1. 提取并清理 Byte 数组
clean_hex = "".join(raw_hex_data.split())
binary_data = bytes.fromhex(clean_hex)

try:
    import msgpack
    print("\n=== 2. MsgPack 还原结构化数据 ===")
    parsed_data = msgpack.unpackb(binary_data, raw=False)
    print(json.dumps(parsed_data, indent=2, ensure_ascii=False))
except ImportError:
    print("\n提示：如需完全还原成结构化 JSON 字典，请先执行 `pip install msgpack`。")
