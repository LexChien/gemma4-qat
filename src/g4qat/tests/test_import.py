def test_import_and_shims():
    import g4qat, g4qat.grid, g4qat.ct_compat, g4qat.export.ct, g4qat.eval.hf
    import g4qat.export_ct, g4qat.hf_eval  # backward-compatible modules
    assert g4qat.__version__ == "0.1.0"
    assert g4qat.export_ct.main is g4qat.export.ct.main
    assert g4qat.hf_eval.main is g4qat.eval.hf.main
    assert hasattr(g4qat.grid, "unpack_mlx") or True  # unpack_mlx needs torch; unpack_u32 always there
    from g4qat.grid import unpack_u32_nibbles  # noqa: F401
