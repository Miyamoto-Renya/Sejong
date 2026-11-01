import ctypes as c
import hashlib,json,os,statistics as stats,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
P=c.c_void_p;I=c.c_int;U=c.c_uint;Z=c.c_size_t
hip=c.CDLL(str(Path(os.environ['ROCM_PATH'])/'lib/libamdhip64.so.7'))
specs={'hipInit':([U],I),'hipSetDevice':([I],I),'hipDeviceGetName':([P,I,I],I),'hipGetErrorString':([I],c.c_char_p),'hipMalloc':([c.POINTER(P),Z],I),'hipFree':([P],I),'hipMemcpy':([P,P,Z,I],I),'hipStreamCreate':([c.POINTER(P)],I),'hipStreamSynchronize':([P],I),'hipStreamDestroy':([P],I),'hipEventCreate':([c.POINTER(P)],I),'hipEventRecord':([P,P],I),'hipEventElapsedTime':([c.POINTER(c.c_float),P,P],I),'hipEventDestroy':([P],I),'hipModuleLoadData':([c.POINTER(P),P],I),'hipModuleGetFunction':([c.POINTER(P),P,c.c_char_p],I),'hipModuleUnload':([P],I),'hipModuleLaunchKernel':([P,U,U,U,U,U,U,U,P,c.POINTER(P),c.POINTER(P)],I)}
for name,(args,result) in specs.items():f=getattr(hip,name);f.argtypes=args;f.restype=result

def check(value,op):
 if value:raise RuntimeError(f'{op}: {value} {hip.hipGetErrorString(value).decode()}')
def call(name,*args):check(getattr(hip,name)(*args),name)
started=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
call('hipInit',0);call('hipSetDevice',0)
name=c.create_string_buffer(256);call('hipDeviceGetName',name,256,0)
stream=P();call('hipStreamCreate',c.byref(stream))
events=[P(),P()]
for ev in events:call('hipEventCreate',c.byref(ev))
metadata=json.loads((ROOT/'dx-compile.json').read_text())
modules={}
for row in metadata:
 blob=c.create_string_buffer((ROOT/(row['key']+'.hsaco')).read_bytes())
 module=P();fn=P();call('hipModuleLoadData',c.byref(module),blob);call('hipModuleGetFunction',c.byref(fn),module,row['symbol'].encode())
 modules[row['key']]=(module,fn,row)
BATCH=128;ROUNDS=6;WARMUP=24
rng=np.random.default_rng(20261101)
results=[]
launch=hip.hipModuleLaunchKernel
for M,N,K in [(512,128,384),(32,512,384)]:
 inputs=[rng.normal(0,.5,(M,K)).astype(np.float32),rng.normal(0,.5,(M,K)).astype(np.float32),rng.normal(0,1/np.sqrt(K),(N,K)).astype(np.float16),rng.normal(0,1/np.sqrt(K),(N,K)).astype(np.float16)]
 allocations=[]
 for size in [*(x.nbytes for x in inputs),*([M*N*4]*4)]:
  p=P();call('hipMalloc',c.byref(p),size);allocations.append(p)
 for host,p in zip(inputs,allocations):call('hipMemcpy',p,P(host.ctypes.data),host.nbytes,1)
 def bind(key,args,grid):
  module,fn,meta=modules[key]
  values=[*args,P(),P()]
  pointers=(P*len(values))(*[c.cast(c.byref(v),P) for v in values])
  return fn,meta['shared'],meta['threads'],values,pointers,grid
 grid=((M+15)//16,(N+31)//32,1)
 base=[bind('matmul_dx',[allocations[0],allocations[2],allocations[4],I(M),I(N),I(K)],grid),bind('matmul_dx',[allocations[1],allocations[3],allocations[5],I(M),I(N),I(K)],grid),bind('add_dx',[allocations[4],allocations[5],allocations[6],I(M*N),c.c_float(1.)],((M*N+255)//256,1,1))]
 fused=[bind('gate_up_dx',[allocations[0],allocations[1],allocations[2],allocations[3],allocations[7],I(M),I(N),I(K)],grid)]
 variants=[base,fused]
 def run(index,count):
  for _ in range(count):
   for fn,shared,threads,values,ptrs,g in variants[index]:
    code=launch(fn,*g,threads,1,1,shared,stream,ptrs,None)
    if code:check(code,'hipModuleLaunchKernel')
 for ix in [0,1,1,0]:run(ix,WARMUP)
 call('hipStreamSynchronize',stream)
 records=[[],[]]
 for r in range(ROUNDS):
  for ix in [0,1,1,0]:
   call('hipEventRecord',events[0],stream)
   before=time.perf_counter();run(ix,BATCH);after=time.perf_counter()
   call('hipEventRecord',events[1],stream);call('hipStreamSynchronize',stream)
   elapsed=c.c_float();call('hipEventElapsedTime',c.byref(elapsed),*events)
   records[ix].append({'event_us_per_sequence':elapsed.value*1000/BATCH,'enqueue_us_per_sequence':(after-before)*1e6/BATCH})
 output=[]
 for p in allocations[6:]:
  x=np.empty((M,N),dtype=np.float32);call('hipMemcpy',P(x.ctypes.data),p,x.nbytes,2);output.append(x.astype(np.float64))
 a,b=output;d=a-b
 errors={'dx':{'max_abs':float(np.abs(d).max()),'relative_l2':float(np.linalg.norm(d)/np.linalg.norm(a)),'different_elements':int(np.count_nonzero(a!=b))}}
 summary=[]
 for rows in records:
  summary.append({key:{'min':min(r[key] for r in rows),'median':stats.median(r[key] for r in rows),'max':max(r[key] for r in rows)} for key in rows[0]})
 dg,du,wg,wu=inputs;sr=rng.integers(M,size=64);sc=rng.integers(N,size=64)
 ref=np.array([np.dot(dg[i].astype(np.float64),wg[j].astype(np.float64))+np.dot(du[i].astype(np.float64),wu[j].astype(np.float64)) for i,j in zip(sr,sc)])
 cpu={'samples':64,'baseline_max_abs':float(np.abs(output[0][sr,sc]-ref).max()),'fused_max_abs':float(np.abs(output[1][sr,sc]-ref).max())}
 result={'shape_MNK':[M,N,K],'baseline_launches':3,'fused_launches':1,'baseline':summary[0],'fused':summary[1],'median_speedup':summary[0]['event_us_per_sequence']['median']/summary[1]['event_us_per_sequence']['median'],'errors':errors,'cpu_fp64_samples':cpu,'samples':records}
 print(json.dumps({k:v for k,v in result.items() if k!='samples'}),flush=True);results.append(result)
 for p in allocations:call('hipFree',p)
for module,fn,meta in modules.values():call('hipModuleUnload',module)
for ev in events:call('hipEventDestroy',ev)
call('hipStreamDestroy',stream)
report={'device':name.value.decode(),'started_utc':started,'finished_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'dy_dtype':'f32','weight_dtype':'f16','output_dtype':'f32','dot_operands':'fp32 IEEE','accumulator':'f32','tile_MNK':[16,32,32],'warps':4,'batch':BATCH,'ABBA_rounds':ROUNDS,'warmup_per_slot':WARMUP,'measurement':'HIP events bracket a sequence batch; host launch gaps are included. Allocation/copies/compilation occur outside timing. Baseline sequence=DG WG^T,DU WU^T,add; fused sequence=separate FP32 accumulators and one final add, storing DX.','runs':results,'artifact_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.hsaco')}}
(ROOT/'dx-microbench.json').write_text(json.dumps(report,indent=2)+'\n')
