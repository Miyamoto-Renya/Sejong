"""현재 Sejong GEMM과 HIP 전송을 독립적으로 측정한다."""
import ctypes as c
import importlib.metadata as metadata
import json, statistics, time
from pathlib import Path
from sejong_kernels.bridge.compiler import compile_request
P=c.c_void_p; I=c.c_int; U=c.c_uint; Z=c.c_size_t
hip=c.CDLL(str(metadata.distribution('rocm-sdk-core').locate_file('_rocm_sdk_core/lib/libamdhip64.so.7')))
spec={'hipInit':([U],I),'hipSetDevice':([I],I),'hipMalloc':([c.POINTER(P),Z],I),'hipFree':([P],I),'hipMemset':([P,I,Z],I),'hipMemcpy':([P,P,Z,I],I),'hipDeviceSynchronize':([],I),'hipEventCreate':([c.POINTER(P)],I),'hipEventRecord':([P,P],I),'hipEventElapsedTime':([c.POINTER(c.c_float),P,P],I),'hipEventDestroy':([P],I),'hipModuleLoad':([c.POINTER(P),c.c_char_p],I),'hipModuleGetFunction':([c.POINTER(P),P,c.c_char_p],I),'hipModuleLaunchKernel':([P,U,U,U,U,U,U,U,P,c.POINTER(P),c.POINTER(P)],I),'hipModuleUnload':([P],I)}
for name,(args,result) in spec.items():
 fn=getattr(hip,name);fn.argtypes=args;fn.restype=result
def call(name,*args):
 result=getattr(hip,name)(*args)
 if result:raise RuntimeError((name,result))
call('hipInit',0);call('hipSetDevice',0)
events=[P(),P()]
for event in events:call('hipEventCreate',c.byref(event))
modules={};rows=[]
def bench(label,M,N,K,ta=False,tb=False,ad='fp16',bd='fp16',cd='fp16'):
 operation='matmul64' if cd=='fp64' else 'matmul'
 key=(operation,ta,tb,ad,bd,cd)
 if key not in modules:
  constants={'TA':ta,'TB':tb,**({'BLOCK':32} if operation=='matmul64' else {'BM':16,'BN':32,'BK':32})}
  r=compile_request({'operation':operation,'signature':{'A':'*'+ad,'B':'*'+bd,'C':'*'+cd,'M':'i32','N':'i32','K':'i32'},'constants':constants,'architecture':'gfx1032','wavefront':32,'warps':4})
  module=P();fn=P();call('hipModuleLoad',c.byref(module),r['binary'].encode());call('hipModuleGetFunction',c.byref(fn),module,r['symbol'].encode());modules[key]=(module,fn,r)
 module,fn,r=modules[key]
 sizes={'fp16':2,'fp32':4,'fp64':8}
 buffers=[]
 for n,dt in [(M*K,ad),(K*N,bd),(M*N,cd)]:
  p=P();call('hipMalloc',c.byref(p),n*sizes[dt]);call('hipMemset',p,0,n*sizes[dt]);buffers.append(p)
 args=[*buffers,I(M),I(N),I(K),P(),P()]
 pointers=(P*len(args))(*[c.cast(c.byref(v),P) for v in args])
 grid=(M,(N+31)//32) if operation=='matmul64' else ((M+15)//16,(N+31)//32)
 def launch():call('hipModuleLaunchKernel',fn,*grid,1,r['threads'],1,1,r['shared'],None,pointers,None)
 launch();call('hipDeviceSynchronize')
 times=[]
 for _ in range(3):
  call('hipEventRecord',events[0],None);launch();call('hipEventRecord',events[1],None);call('hipDeviceSynchronize')
  elapsed=c.c_float();call('hipEventElapsedTime',c.byref(elapsed),*events);times.append(elapsed.value)
 row={'label':label,'MNK':[M,N,K],'TA':ta,'TB':tb,'dtype':[ad,bd,cd],'ms':times,'median_ms':statistics.median(times),'logical_TF':2*M*N*K/statistics.median(times)/1e9}
 rows.append(row);print(json.dumps(row),flush=True)
 for p in buffers:call('hipFree',p)
 Path('.cache/composite-cost/probe.json').write_text(json.dumps({'kernels':rows},indent=2)+'\n')
# Byte NativeValue: physical B1, sequence1024, global128, E8/top2 mean32 expert rows.
for tag,m,k,n in [('local',1024,256,768),('global',128,1024,1024),('expert',32,1024,768)]:
 bench(tag+'-forward',m,n,k)
 bench(tag+'-dX',m,k,n,tb=True,ad='fp32',cd='fp32')
 bench(tag+'-dW',k,n,m,ta=True,bd='fp32',cd='fp32')
# Memory fact phase: query dimension1024, K/V hidden109088, batch32.
bench('memory-K-forward',32,109088,1024,ad='fp32',bd='fp32',cd='fp32')
bench('memory-V-forward',32,1024,109088,ad='fp32',bd='fp32',cd='fp32')
bench('memory-K-dW',1024,109088,32,ta=True,ad='fp32',bd='fp32',cd='fp32')
# Direct reading, original FP64 scalar GEMM.
bench('direct-K',1,109088,1024,ad='fp64',bd='fp64',cd='fp64')
bench('direct-V',1,1024,109088,ad='fp64',bd='fp64',cd='fp64')
size=128*1024*1024;host=c.create_string_buffer(size);dev=P();call('hipMalloc',c.byref(dev),size)
transfers=[]
for _ in range(4):
 start=time.perf_counter();call('hipMemcpy',dev,c.cast(host,P),size,1);call('hipDeviceSynchronize');transfers.append(time.perf_counter()-start)
call('hipFree',dev)
result={'scope':'individual original kernels; zero-filled operands; one warmup and three HIP-event samples; allocation and compilation outside timing','kernels':rows,'pageable_H2D':{'bytes':size,'seconds':transfers,'steady_GB_s':size/statistics.median(transfers[1:])/1e9}}
Path('.cache/composite-cost/probe.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'pageable_H2D':result['pageable_H2D']}),flush=True)
for module,_,_ in modules.values():call('hipModuleUnload',module)
for event in events:call('hipEventDestroy',event)
