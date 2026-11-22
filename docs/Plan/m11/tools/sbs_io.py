"""Minimal read-only parser for gemma.cpp SBS (BlobStore V1/V2) files. M11."""
import struct, numpy as np
MAGIC = 0x0A534253

def read_dir(path):
    with open(path, 'rb') as f:
        magic, n, fbytes = struct.unpack('<IIQ', f.read(16))
        assert magic == MAGIC
        f.seek(0, 2); size = f.tell()
        if n == 0:  # V2: header + directory at end
            f.seek(size - 16); magic, n, fbytes = struct.unpack('<IIQ', f.read(16))
            f.seek(size - 16 - 32 * n)
        assert fbytes == size, (fbytes, size)
        d = f.read(32 * n)
    keys = [d[16*i:16*i+16].split(b'\0')[0].decode() for i in range(n)]
    ranges = [struct.unpack('<QQ', d[16*(n+i):16*(n+i)+16]) for i in range(n)]
    return {k: r for k, r in zip(keys, ranges)}, size

def blob(path, rng):
    off, nbytes = rng
    return np.memmap(path, dtype=np.uint8, mode='r', offset=off, shape=(nbytes,))

def q4_decode(raw):
    """raw uint8 Q4_0Stream bytes -> (levels uint8 [nblk,32] in 0..15, scale float32 [nblk]); value=(L-8)*scale."""
    b = np.asarray(raw).reshape(-1, 18)
    sc = (b[:, :2].copy().view(np.uint16).astype(np.uint32) << 16).view(np.float32)[:, 0]
    qs = b[:, 2:]
    L = np.concatenate([qs & 15, qs >> 4], axis=1)
    return L, sc
