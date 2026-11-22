"""Q4_0 group-32 exact grid helpers (shared by export and, later, MLX/JAX trainers).

MLX affine pack stores unsigned levels L in [0, 15] packed 8-per-uint32 (low nibble first),
with per-group scale d (bf16) and bias b. Official / our exact packs use b = -8d, so the
signed quantization points are q = L-8 in [-8, 7] and the dequant is (L)*d + b = q*d.

`levels_fixed(w, d)` snaps a dense weight onto a *given* official scale (no closed-form
scale reproduces the official grid; see docs/Plan/08).
"""
from __future__ import annotations
import numpy as np

GROUP = 32


def unpack_u32_nibbles(w: np.ndarray) -> np.ndarray:
    """[out, in/8] uint32/int32 -> levels uint8 [out, in], low nibble first (MLX layout)."""
    w = np.asarray(w).view(np.uint32)
    sh = (np.arange(8, dtype=np.uint32) * 4)
    return ((w[..., None] >> sh) & 15).reshape(w.shape[0], -1).astype(np.uint8)


def pack_u32_nibbles(levels: np.ndarray) -> np.ndarray:
    """Inverse of unpack_u32_nibbles: uint8 [out, in] -> uint32 [out, in/8]."""
    L = np.asarray(levels, dtype=np.uint8)
    rows, cols = L.shape
    assert cols % 8 == 0, cols
    x = L.reshape(rows, cols // 8, 8).astype(np.uint32)
    sh = (np.arange(8, dtype=np.uint32) * 4)
    return np.bitwise_or.reduce(x << sh, axis=-1)


def dequant_affine(levels: np.ndarray, scales: np.ndarray, biases: np.ndarray, group: int = GROUP) -> np.ndarray:
    """levels uint8 [out, in], scales/biases [out, in/group] -> float32 dense."""
    L = np.asarray(levels, dtype=np.float32)
    rows, cols = L.shape
    s = np.asarray(scales, dtype=np.float32)[..., None]
    b = np.asarray(biases, dtype=np.float32)[..., None]
    return (L.reshape(rows, cols // group, group) * s + b).reshape(rows, cols)


def levels_fixed(w: np.ndarray, d: np.ndarray, group: int = GROUP) -> np.ndarray:
    """Snap dense w [out,in] onto fixed scale d [out,in/group] -> signed levels int8 in [-8, 7]."""
    w = np.asarray(w, dtype=np.float32)
    d = np.asarray(d, dtype=np.float32)
    rows, cols = w.shape
    blk = w.reshape(rows, cols // group, group)
    d32 = np.where(d[..., None] == 0, 1.0, d[..., None])
    return np.clip(np.rint(blk / d32), -8, 7).astype(np.int8).reshape(rows, cols)


def q4_fixed(w: np.ndarray, d: np.ndarray, group: int = GROUP) -> np.ndarray:
    L = levels_fixed(w, d, group).reshape(w.shape[0], w.shape[1] // group, group).astype(np.float32)
    return (L * np.asarray(d, dtype=np.float32)[..., None]).reshape(w.shape)


def bias_is_minus_8d(scales, biases) -> bool:
    s = np.asarray(scales, dtype=np.float32)
    b = np.asarray(biases, dtype=np.float32)
    # Compare in bf16-equivalent (IEEE float16 is not bf16; use rounded float32 check at 2**-7 ulp of typical scales).
    return bool(np.all(np.abs(b + 8 * s) <= 1e-3 * np.maximum(1e-6, np.abs(s))))


try:
    import torch

    def unpack_mlx(w_u32: "torch.Tensor") -> "torch.Tensor":
        w = w_u32.view(torch.int32)
        sh = torch.arange(8, dtype=torch.int32) * 4
        return ((w.unsqueeze(-1) >> sh) & 15).reshape(w.shape[0], -1).to(torch.uint8)

    def requant_sym(wf: "torch.Tensor", g: int):
        """wf float32 [rows, cols] -> (q int8 [-8,7], scale bf16 [rows, cols/g]) by per-group MSE scale search."""
        r, c = wf.shape
        x = wf.reshape(r, c // g, g)
        amax = x.abs().amax(-1, keepdim=True).clamp_min(1e-12)
        best_e = best_s = best_q = None
        for div in (7.0, 7.25, 7.5, 7.75, 8.0):
            s = (amax / div).to(torch.bfloat16).float()
            q = torch.clamp(torch.round(x / s), -8, 7)
            e = ((q * s - x) ** 2).sum(-1, keepdim=True)
            if best_e is None:
                best_e, best_s, best_q = e, s, q
            else:
                m = e < best_e
                best_e = torch.where(m, e, best_e)
                best_s = torch.where(m, s, best_s)
                best_q = torch.where(m, q, best_q)
        return best_q.reshape(r, c).to(torch.int8), best_s[..., 0].to(torch.bfloat16)

    def dequant_affine_torch(levels, scales, biases, group: int = GROUP):
        L = levels.float(); rows, cols = L.shape
        return (L.reshape(rows, cols // group, group) * scales.float()[..., None] + biases.float()[..., None]).reshape(rows, cols)

except ImportError:  # pragma: no cover
    pass
