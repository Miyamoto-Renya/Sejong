import ctypes as c
import hashlib
import json
import os
import statistics as stats
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
hip=c.CDLL(str(Path(os.environ['ROCM_PATH'])/'lib/libamdhip64.so.7'))
P=c.c_void_p; I=c.c_int; U=c.c_uint; Z=c.c_size_t
specs={
 'hipInit':([U],I),'hipSetDevice':([I],I),'hipDeviceGetName':([P,I,I],I),
 'hipGetErrorString':([I],c.c_char_p),'hipMalloc':([c.POINTER(P),Z],I),'hipFree':([P],I),
 'hipMemcpy':([P,P,Z,I],I),'hipStreamCreate':([c.POINTER(P)],I),
 'hipStreamSynchronize':([P],I),'hipStreamDestroy':([P],I),
 'hipEventCreate':([c.POINTER(P)],I),'hipEventRecord':([P,P],I),
 'hipEventElapsedTime':([c.POINTER(c.c_float),P,P],I),'hipEventDestroy':([P],I),
 'hipModuleLoadData':([c.POINTER(P),P],I),
 'hipModuleGetFunction':([c.POINTER(P),P,c.c_char_p],I),'hipModuleUnload':([P],I),
 'hipModuleLaunchKernel':([P,U,U,U,U,U,U,U,P,c.POINTER(P),c.POINTER(P)],I)
}
for name,(args,result) in specs.items():
 f=getattr(hip,name); f.argtypes=args; f.restype=result

def check(value,op):
 if value: raise RuntimeError(f'{op}: {value} {hip.hipGetErrorString(value).decode()}')

def call(name,*args):check(getattr(hip,name)(*args),name)

call('hipInit',0);call('hipSetDevice',0)
name=c.create_string_buffer(256);call('hipDeviceGetName',name,256,0)
stream=P();call('hipStreamCreate',c.byref(stream))
events=[P(),P()]
for event in events:call('hipEventCreate',c.byref(event))
metadata=json.loads((ROOT/'dot2-compile.json').read_text())
modules=[]
for key,symbol in [('matmul-0-0','matmul'),('native_half_matmul-0-0','native_half_matmul')]:
 row=next(x for x in metadata if x['kernel']==key)
 blob=c.create_string_buffer((ROOT/f'{key}.hsaco').read_bytes())
 module=P();function=P()
 call('hipModuleLoadData',c.byref(module),blob)
 call('hipModuleGetFunction',c.byref(function),module,symbol.encode())
 modules.append((module,function,row['shared_bytes']))

BATCH=256;ROUNDS=8;WARMUP=24
results=[]
rng=np.random.default_rng(20261101)
for M,N,K in [(512,384,128),(512,128,384),(64,512,512)]:
 a=rng.normal(0,.5,(M,K)).astype(np.float16)
 b=rng.normal(0,.5,(K,N)).astype(np.float16)
 allocations=[]
 for size in [a.nbytes,b.nbytes,M*N*2,M*N*2]:
  pointer=P();call('hipMalloc',c.byref(pointer),size);allocations.append(pointer)
 call('hipMemcpy',allocations[0],P(a.ctypes.data),a.nbytes,1)
 call('hipMemcpy',allocations[1],P(b.ctypes.data),b.nbytes,1)
 variants=[]
 for index,(module,function,shared) in enumerate(modules):
  arguments=[allocations[0],allocations[1],allocations[2+index],I(M),I(N),I(K),P(),P()]
  pointers=(P*len(arguments))(*[c.cast(c.byref(x),P) for x in arguments])
  variants.append((function,shared,arguments,pointers))
 launch=hip.hipModuleLaunchKernel
 def run(index,count):
  function,shared,arguments,pointers=variants[index]
  for _ in range(count):
   code=launch(function,(M+15)//16,(N+31)//32,1,128,1,1,shared,stream,pointers,None)
   if code:check(code,'hipModuleLaunchKernel')
 for index in [0,1,1,0]:run(index,WARMUP)
 call('hipStreamSynchronize',stream)
 samples=[[],[]]
 sequence=[]
 for round_no in range(ROUNDS):
  for index in [0,1,1,0]:
   started=time.perf_counter()
   call('hipEventRecord',events[0],stream)
   enqueue_started=time.perf_counter()
   run(index,BATCH)
   enqueue_finished=time.perf_counter()
   call('hipEventRecord',events[1],stream)
   call('hipStreamSynchronize',stream)
   finished=time.perf_counter()
   elapsed=c.c_float();call('hipEventElapsedTime',c.byref(elapsed),*events)
   row={'event_us_per_launch':elapsed.value*1000/BATCH,'enqueue_us_per_launch':(enqueue_finished-enqueue_started)*1e6/BATCH,'wall_us_per_launch':(finished-started)*1e6/BATCH}
   samples[index].append(row);sequence.append({'round':round_no,'variant':['baseline','native_half'][index],**row})
 output=[]
 for pointer in allocations[2:]:
  host=np.empty((M,N),dtype=np.float16)
  call('hipMemcpy',P(host.ctypes.data),pointer,host.nbytes,2);output.append(host)
 base,native=[x.astype(np.float64) for x in output]
 delta=native-base
 rows=rng.integers(M,size=64);cols=rng.integers(N,size=64)
 reference=np.array([np.dot(a[i].astype(np.float64),b[:,j].astype(np.float64)) for i,j in zip(rows,cols)])
 sampled=[x[rows,cols] for x in [base,native]]
 summary=[]
 for values in samples:
  summary.append({key:{'minimum':min(row[key] for row in values),'median':stats.median(row[key] for row in values),'maximum':max(row[key] for row in values)} for key in values[0]})
 result={'shape_MNK':[M,N,K],'variants':{'baseline':summary[0],'native_half':summary[1]},'native_vs_baseline':{'max_abs':float(np.abs(delta).max()),'relative_l2':float(np.linalg.norm(delta)/np.linalg.norm(base))},'cpu_fp64_dot_samples':{'count':64,'baseline_max_abs':float(np.abs(sampled[0]-reference).max()),'baseline_relative_l2':float(np.linalg.norm(sampled[0]-reference)/np.linalg.norm(reference)),'native_max_abs':float(np.abs(sampled[1]-reference).max()),'native_relative_l2':float(np.linalg.norm(sampled[1]-reference)/np.linalg.norm(reference))},'samples':sequence}
 result['native_median_speedup']=summary[0]['event_us_per_launch']['median']/summary[1]['event_us_per_launch']['median']
 results.append(result)
 print(json.dumps({key:value for key,value in result.items() if key!='samples'}),flush=True)
 for pointer in allocations:call('hipFree',pointer)
for module,_,_ in modules:call('hipModuleUnload',module)
for event in events:call('hipEventDestroy',event)
call('hipStreamDestroy',stream)
report={'started_utc':started_utc,'finished_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'measurement_context':'Sequential GPU ownership after root benchmark completion','device':name.value.decode(),'target':'gfx1032','input_dtype':'f16','output_dtype':'f16','accumulator':'f32','tile_MNK':[16,32,32],'warps':4,'wave_size':32,'warmup_per_ABBA_slot':WARMUP,'batch_launches':BATCH,'ABBA_rounds':ROUNDS,'measurement':'HIP events bracket each batch on a dedicated stream; event time includes device idle gaps from host submission. enqueue is CPU launch-loop wall time; wall includes event calls and final stream synchronization. H2D/D2H, allocation and compilation occur outside timing.','seed':20261101,'runs':results}
report['artifact_sha256'] = {
 path.name: hashlib.sha256(path.read_bytes()).hexdigest()
 for path in [ROOT/'matmul-0-0.hsaco', ROOT/'native_half_matmul-0-0.hsaco',
              ROOT/'baseline_kernel.py', ROOT/'dot2-compile-probe.py', Path(__file__)]
}
(ROOT/'dot2-microbench.json').write_text(json.dumps(report,indent=2)+'\n')
