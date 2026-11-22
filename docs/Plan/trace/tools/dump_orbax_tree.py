# 讀取 orbax OCDBT checkpoint 的每個陣列 shape/dtype（唯讀，不載入數值）
import os, json, sys, ast, tensorstore as ts
base = sys.argv[1]; out = sys.argv[2]
meta = json.load(open(f'{base}/_METADATA'))['tree_metadata']
kv = ts.KvStore.open({'driver':'ocdbt','base':{'driver':'file','path':os.path.abspath(base)+'/'}}).result()
keys = [k.decode() for k in kv.list().result()]
zarr = sorted({k.rsplit('/',1)[0] for k in keys if k.endswith('.zarray')})
res = []
for k in meta:
    path = ast.literal_eval(k); name = '.'.join(path)
    vt = meta[k]['value_metadata']['value_type']
    if name in zarr:
        za = json.loads(kv.read(name + '/.zarray').result().value)
        res.append({'path':name,'shape':za['shape'],'dtype':za['dtype'],'value_type':vt})
    else:
        res.append({'path':name,'shape':None,'dtype':None,'value_type':vt})
json.dump(res, open(out,'w'), indent=1)
print(len(res), sum(1 for r in res if r['shape'] is not None), len(zarr))
