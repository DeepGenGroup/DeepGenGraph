import json
import sys

raw_hex_data = sys.argv[1]

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
