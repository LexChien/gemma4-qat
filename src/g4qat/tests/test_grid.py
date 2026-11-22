"""C-1 smoke tests for g4qat.grid (no model weights required)."""
import numpy as np
from g4qat.grid import unpack_u32_nibbles, pack_u32_nibbles, dequant_affine, levels_fixed, q4_fixed, GROUP

def test_pack_unpack_nibbles_roundtrip():
    rng = np.random.default_rng(0)
    L = rng.integers(0, 16, size=(17, 64), dtype=np.uint8)
    pk = pack_u32_nibbles(L)
    assert pk.shape == (17, 8) and pk.dtype == np.uint32
    assert np.array_equal(unpack_u32_nibbles(pk), L)

def test_dequant_symmetric_bias():
    # L=0..15, d=0.5, b=-8d => values -4, -3.5, ..., 3.5
    rows, cols = 2, 32
    L = np.arange(32, dtype=np.uint8)[None, :].repeat(rows, 0)
    d = np.full((rows, 1), 0.5, dtype=np.float32)
    b = -8 * d
    w = dequant_affine(L, d, b)
    assert w.shape == (rows, cols)
    assert np.allclose(w[0], (np.arange(32) - 8) * 0.5)

def test_levels_fixed_is_fixed_point():
    rng = np.random.default_rng(1)
    d = rng.random((4, 2)).astype(np.float32) * 0.1 + 0.01
    q = rng.integers(-8, 8, size=(4, 64), dtype=np.int8)
    w = (q.reshape(4, 2, GROUP).astype(np.float32) * d[..., None]).reshape(4, 64)
    got = levels_fixed(w, d)
    assert np.array_equal(got, q)
    assert np.allclose(q4_fixed(w, d), w)

def test_levels_fixed_clips():
    d = np.ones((1, 1), dtype=np.float32)
    w = np.array([[100.0] + [0.0] * 31], dtype=np.float32)
    L = levels_fixed(w, d)
    assert L[0, 0] == 7 and L[0, 1] == 0
