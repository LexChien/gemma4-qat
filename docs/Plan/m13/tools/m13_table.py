"""Per-item correctness matrix across native (set_*.jsonl) and MLX (mlx_*.jsonl) runs + per-language totals."""
import json, glob, os, sys
sys.path.insert(0, 'docs/Plan/m13/tools'); import m13_sets as S
R = 'docs/Plan/m13/raw'; runs = {}; ITEMS = {i[0]: i for i in S.ALL}
for f in sorted(glob.glob(f'{R}/set_*.jsonl')) + sorted(glob.glob(f'{R}/setenv_*.jsonl')) + sorted(glob.glob(f'{R}/settxt_*.jsonl')) + sorted(glob.glob(f'{R}/mlx_*.jsonl')):
    t = os.path.basename(f)[:-6].replace('setenv_', 'W:').replace('settxt_', 'T:').replace('set_', 'N:').replace('mlx_', 'X:'); L = {json.loads(l)['id']: json.loads(l) for l in open(f)}
    if len(L) >= 80:
        for i, r in L.items(): r['correct'], r['lang_ok'], _ = S.grade(ITEMS[i], r['answer'])
        runs[t] = L
names = list(runs)
for n in names:
    L = runs[n]; b = lambda p, lo, hi: sum(L[i]['correct'] for i in L if i[0] == p and lo <= int(i[1:]) <= hi)
    print(f"{n:20s} base zh {b('z',1,40):2d}/40 en {b('e',1,40):2d}/40 | hard zh {b('z',41,60):2d}/20 en {b('e',41,60):2d}/20 | total zh {b('z',1,60):2d}/60 en {b('e',1,60):2d}/60 (n={len(L)})")
print('\nitems wrong in any run:'); print(' ' * 5 + ' '.join(f'{k:>3d}' for k in range(len(names))))
for it in S.ALL:
    row = [runs[n][it[0]]['correct'] if it[0] in runs[n] else None for n in names]
    if not all(r is not False for r in row) : print(f"{it[0]:5s}" + ' '.join('  .' if c else '  -' if c is None else '  X' for c in row))
for k, n in enumerate(names): print(k, n)
