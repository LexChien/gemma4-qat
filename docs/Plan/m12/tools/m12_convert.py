"""M12: run the ORIGINAL gemma.cpp converter (ALife work copy, unmodified file) with two in-memory overrides:
  M12_Q4=c_embedding,ple_model_proj  -> these tensors are written as the target type (Q4_0) instead of bf16
  M12_VOCAB=<n>                       -> ModelConfig.vocab_size override (vocab-trimmed model)
Everything else (layout, writer, tokenizer packing) is the converter's own code. Never overwrites.
usage: m12_convert.py <in.safetensors> <tokenizer.json> <out.sbs> <out.csv>"""
import sys, os, importlib.util
CONV = '/Users/lexchien/Documents/AI/ALife-Platform/Google-Gemma4/work/source/gemma.cpp/python/convert_from_safetensors.py'
os.environ.setdefault('GEMMA_CPP_BUILD_DIR', '/Users/lexchien/Documents/AI/ALife-Platform/Google-Gemma4/runtime/bindings')
src, tok, out, csv = sys.argv[1:5]
assert not os.path.exists(out), 'refuse overwrite'
spec = importlib.util.spec_from_file_location('cfs', CONV); cfs = importlib.util.module_from_spec(spec); spec.loader.exec_module(cfs)
Q4 = {x for x in os.environ.get('M12_Q4', '').split(',') if x}
orig = cfs._is_bf16_param
cfs._is_bf16_param = lambda n: False if n in Q4 else orig(n)
V = int(os.environ.get('M12_VOCAB', '0'))
if V:
    real = cfs.configs
    class Proxy:
        def __getattr__(self, k): return getattr(real, k)
        @staticmethod
        def ModelConfig(*a):
            c = real.ModelConfig(*a); c.vocab_size = V; return c
    cfs.configs = Proxy()
print('M12 overrides: Q4', sorted(Q4), 'vocab', V or 'unchanged', flush=True)
sys.argv = ['convert', '--model_specifier=gemma4-2b-lm-q4_0-it', f'--load_path={src}', f'--tokenizer_file={tok}',
            f'--sbs_file={out}.partial', f'--metadata_file={csv}']
try:
    cfs.app.run(cfs.main)
except SystemExit as e:
    if e.code not in (0, None): raise
os.rename(out + '.partial', out); print('CONVERT_DONE', out)
