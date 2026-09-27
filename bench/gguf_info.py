#!/usr/bin/env python3
# Read GGUF header metadata without loading the whole file.
# Usage: gguf_info.py <file.gguf> [key-substring ...]
import struct
import sys


def read_string(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", "replace")


def skip_value(f, vtype):
    # types that are fixed size: 0-7,10,11,12 = 1/1/2/2/4/4/4/1/8/8/8 bytes
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if vtype in sizes:
        f.read(sizes[vtype])
    elif vtype == 8:  # string
        n = struct.unpack("<Q", f.read(8))[0]
        f.read(n)
    elif vtype == 9:  # array
        et = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            skip_value(f, et)
    else:
        raise RuntimeError(f"unknown type {vtype}")


def read_value(f, vtype):
    if vtype == 0:
        return struct.unpack("<B", f.read(1))[0]
    if vtype == 1:
        return struct.unpack("<b", f.read(1))[0]
    if vtype == 2:
        return struct.unpack("<H", f.read(2))[0]
    if vtype == 3:
        return struct.unpack("<h", f.read(2))[0]
    if vtype == 4:
        return struct.unpack("<I", f.read(4))[0]
    if vtype == 5:
        return struct.unpack("<i", f.read(4))[0]
    if vtype == 6:
        return struct.unpack("<f", f.read(4))[0]
    if vtype == 7:
        return struct.unpack("<B", f.read(1))[0] != 0
    if vtype == 8:
        return read_string(f)
    if vtype == 9:  # array
        et = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            skip_value(f, et)
        return f"[array of {n} x type {et}]"
    if vtype == 10:
        return struct.unpack("<Q", f.read(8))[0]
    if vtype == 11:
        return struct.unpack("<q", f.read(8))[0]
    if vtype == 12:
        return struct.unpack("<d", f.read(8))[0]
    return f"<unknown type {vtype}>"


def main():
    path = sys.argv[1]
    filters = sys.argv[2:]
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"GGUF", f"not a GGUF file: {magic}"
        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        print(f"version={version} tensors={n_tensors} metadata_kv={n_kv}")
        for _ in range(n_kv):
            key = read_string(f)
            vtype = struct.unpack("<I", f.read(4))[0]
            val = read_value(f, vtype)
            if not filters or any(fl in key for fl in filters):
                print(f"{key} = {val}")


if __name__ == "__main__":
    main()
