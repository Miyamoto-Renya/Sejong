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
metadata=json.loads((ROOT/'qkv-conditional-compile.json').read_text())
modules={}
for row in metadata:
 blob=c.create_string_buffer((ROOT/(row['key']+'.hsaco')).read_bytes())
 module=P();fn=P();call('hipModuleLoadData',c.byref(module),blob);call('hipModuleGetFunction',c.byref(fn),module,row['symbol'].encode())
 modules[row['key']]=(module,fn,row)
BATCH=256;ROUNDS=8;WARMUP=256
rng=np.random.default_rng(20261101);results=[]
launch=hip.hipModuleLaunchKernel
for M,D,KV in [(512,128,128),(64,512,256)]:
 inputs=[rng.normal(0,.5,(M,D)).astype(np.float16),rng.normal(0,1/np.sqrt(D),(D,D)).astype(np.float16),rng.normal(0,1/np.sqrt(D),(D,KV)).astype(np.float16),rng.normal(0,1/np.sqrt(D),(D,KV)).astype(np.float16)]
 shapes=[(M,D),(M,KV),(M,KV)]*3
 allocations=[]
 for size in [*(x.nbytes for x in inputs),*(a*b*2 for a,b in shapes)]:
  p=P();call('hipMalloc',c.byref(p),size);allocations.append(p)
 for host,p in zip(inputs,allocations):call('hipMemcpy',p,P(host.ctypes.data),host.nbytes,1)
 def bind(key,args,grid):
  module,fn,meta=modules[key];vals=[*args,P(),P()]
  pointers=(P*len(vals))(*[c.cast(c.byref(v),P) for v in vals])
  return fn,meta['shared'],meta['threads'],vals,pointers,grid
 base=[bind('matmul_qkv',[allocations[0],allocations[1+i],allocations[4+i],I(M),I(n),I(D)],((M+15)//16,(n+31)//32,1)) for i,n in enumerate([D,KV,KV])]
 variants=[base]
 for v,bn in enumerate([32,16],1):
  variants.append([bind('qkv_conditional_'+str(bn),[allocations[0],allocations[1],allocations[2],allocations[3],allocations[4+v*3],allocations[5+v*3],allocations[6+v*3],I(M),I(D),I(KV)],((M+15)//16,(D+bn-1)//bn,1))])
 def run(ix,count):
  for _ in range(count):
   for fn,shared,threads,vals,ptrs,g in variants[ix]:
    code=launch(fn,*g,threads,1,1,shared,stream,ptrs,None)
    if code:check(code,'hipModuleLaunchKernel')
 for ix in [0,1,2,2,1,0]:run(ix,WARMUP)
 call('hipStreamSynchronize',stream)
 records=[[],[],[]]
 for r in range(ROUNDS):
  for ix in [0,1,2,2,1,0]:
   call('hipEventRecord',events[0],stream)
   before=time.perf_counter();run(ix,BATCH);after=time.perf_counter()
   call('hipEventRecord',events[1],stream);call('hipStreamSynchronize',stream)
   elapsed=c.c_float();call('hipEventElapsedTime',c.byref(elapsed),*events)
   records[ix].append({'event_us_per_sequence':elapsed.value*1000/BATCH,'enqueue_us_per_sequence':(after-before)*1e6/BATCH})
 outputs=[]
 for p,shape in zip(allocations[4:],shapes):
  arr=np.empty(shape,dtype=np.float16);call('hipMemcpy',P(arr.ctypes.data),p,arr.nbytes,2);outputs.append(arr.astype(np.float64))
 summary=[]
 for rows in records:
  summary.append({key:{'min':min(r[key] for r in rows),'median':stats.median(r[key] for r in rows),'max':max(r[key] for r in rows)} for key in rows[0]})
 errs=[]
 for v in [1,2]:
  er={}
  for i,key in enumerate(['q','k','v']):
   a,b=outputs[i],outputs[v*3+i];dd=a-b
   er[key]={'max_abs':float(np.abs(dd).max()),'relative_l2':float(np.linalg.norm(dd)/np.linalg.norm(a)),'different_elements':int(np.count_nonzero(dd))}
  errs.append(er)
 cpu={}
 for i,key in enumerate(['q','k','v']):
  width=[D,KV,KV][i];rr=rng.integers(M,size=64);cc=rng.integers(width,size=64)
  ref=np.array([np.dot(inputs[0][r].astype(np.float64),inputs[i+1][:,col].astype(np.float64)) for r,col in zip(rr,cc)]).astype(np.float16).astype(np.float64)
  cpu[key]={'samples':64,'baseline_max_abs':float(np.abs(outputs[i][rr,cc]-ref).max()),'fused32_max_abs':float(np.abs(outputs[3+i][rr,cc]-ref).max()),'fused16_max_abs':float(np.abs(outputs[6+i][rr,cc]-ref).max())}
 result={'shape_M_D_KV':[M,D,KV],'baseline_launches':3,'fused_launches':1,'baseline':summary[0],'fused32':summary[1],'fused16':summary[2],'median_speedup32':summary[0]['event_us_per_sequence']['median']/summary[1]['event_us_per_sequence']['median'],'median_speedup16':summary[0]['event_us_per_sequence']['median']/summary[2]['event_us_per_sequence']['median'],'errors32':errs[0],'errors16':errs[1],'cpu_fp64_samples':cpu,'samples':records}
 print(json.dumps({k:v for k,v in result.items() if k!='samples'}),flush=True);results.append(result)
 for p in allocations:call('hipFree',p)
for module,fn,meta in modules.values():call('hipModuleUnload',module)
for ev in events:call('hipEventDestroy',ev)
call('hipStreamDestroy',stream)
report={'device':name.value.decode(),'started_utc':started,'finished_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'input_output_dtype':'f16','dot_operands':'fp32 IEEE','accumulator':'f32','warps':4,'batch':BATCH,'ABCCBA_rounds':ROUNDS,'warmup_per_slot':WARMUP,'measurement':'HIP events bracket a sequence batch; host launch gaps are included. Allocation/copies/compilation occur outside timing. Baseline is three distinct matmuls. Conditional fused executes Q/K/V accumulators below KWIDTH and Q-only above KWIDTH, with separate contiguous outputs.','runs':results,'artifact_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.hsaco')}}
(ROOT/'qkv-microbench.json').write_text(json.dumps(report,indent=2)+'\n')
