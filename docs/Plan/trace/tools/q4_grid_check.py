# 檢查 HF safetensors 權重是否落在 Q4_0 格點（每 32 個一組，16 級）。唯讀 memmap。
import json, struct, sys, numpy as np
def load(path):
    f=open(path,'rb'); n=struct.unpack('<Q',f.read(8))[0]; h=json.loads(f.read(n)); h.pop('__metadata__',None)
    mm=np.memmap(path,dtype=np.uint8,mode='r',offset=8+n); return h,mm
def get(h,mm,k):
    v=h[k]; a,b=v['data_offsets']; u=np.frombuffer(mm[a:b],dtype=np.uint16).astype(np.uint32)<<16
    return u.view(np.float32).reshape(v['shape'])
def grid_stats(w, axis, nblk=4096, seed=0):
    x=np.moveaxis(w,axis,-1); x=x.reshape(-1,32) if x.shape[-1]%32==0 else None
    if x is None: return None
    rng=np.random.default_rng(seed); idx=rng.choice(len(x),min(nblk,len(x)),replace=False); x=x[idx]
    uniq=np.array([len(np.unique(r)) for r in x])
    amax=np.abs(x).max(1,keepdims=True); d=np.where(amax==0,1,amax/8)
    q=x/d; err=np.abs(q-np.round(q)).max(1)
    return dict(blocks=len(x), uniq_mean=float(uniq.mean()), uniq_max=int(uniq.max()), frac_le16=float((uniq<=16).mean()),
                int_err_med=float(np.median(err)), frac_on_grid=float((err<0.02).mean()))
it_h,it_m=load('weights/gemma-4-E2B-it/model.safetensors')
q_h,q_m=load('weights/gemma-4-E2B-it-qat-q4_0-unquantized/model.safetensors')
keys=sys.argv[1:]
out={}
for k in keys:
    r={}
    for tag,(h,m) in {'it':(it_h,it_m),'qat':(q_h,q_m)}.items():
        if k not in h: continue
        w=get(h,m,k)
        r[tag]={'shape':list(w.shape),'mean':float(w.mean()),'std':float(w.std()),'absmax':float(np.abs(w).max()),
                'grid_last_axis':grid_stats(w,-1),'grid_first_axis':grid_stats(w,0)}
    if 'it' in r and 'qat' in r:
        a=get(it_h,it_m,k); b=get(q_h,q_m,k); dlt=b-a
        r['diff']={'rel_l2':float(np.linalg.norm(dlt)/np.linalg.norm(a)),'max_abs':float(np.abs(dlt).max()),'cos':float((a*b).sum()/np.linalg.norm(a)/np.linalg.norm(b))}
    out[k]=r; print(k, json.dumps(r)[:1500], flush=True)
json.dump(out,open('docs/Plan/trace/raw/q4_grid_check.json','w'),indent=1)
