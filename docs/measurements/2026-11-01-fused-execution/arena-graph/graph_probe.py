import ctypes as c
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import statistics
import time
import numpy as np
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from baseline_kernel import add

ROOT=Path(__file__).resolve().parent
P=c.c_void_p; I=c.c_int; U=c.c_uint; Z=c.c_size_t
class Dim3(c.Structure):_fields_=[('x',U),('y',U),('z',U)]
class KernelParams(c.Structure):_fields_=[('blockDim',Dim3),('extra',P),('func',P),('gridDim',Dim3),('kernelParams',c.POINTER(P)),('sharedMemBytes',U)]
SDK=Path(metadata.distribution('rocm-sdk-core').locate_file('_rocm_sdk_core'))
hip=c.CDLL(str(SDK/'lib/libamdhip64.so.7'))
specs={
 'hipInit':([U],I),'hipSetDevice':([I],I),'hipDeviceGetName':([P,I,I],I),'hipRuntimeGetVersion':([c.POINTER(I)],I),
 'hipGetErrorString':([I],c.c_char_p),'hipMalloc':([c.POINTER(P),Z],I),'hipFree':([P],I),
 'hipMemcpy':([P,P,Z,I],I),'hipStreamCreate':([c.POINTER(P)],I),'hipStreamSynchronize':([P],I),'hipStreamDestroy':([P],I),
 'hipEventCreate':([c.POINTER(P)],I),'hipEventRecord':([P,P],I),'hipEventElapsedTime':([c.POINTER(c.c_float),P,P],I),'hipEventDestroy':([P],I),
 'hipModuleLoadData':([c.POINTER(P),P],I),'hipModuleGetFunction':([c.POINTER(P),P,c.c_char_p],I),'hipModuleUnload':([P],I),
 'hipModuleLaunchKernel':([P,U,U,U,U,U,U,U,P,c.POINTER(P),c.POINTER(P)],I),
 'hipStreamBeginCapture':([P,I],I),'hipStreamEndCapture':([P,c.POINTER(P)],I),
 'hipGraphInstantiate':([c.POINTER(P),P,c.POINTER(P),P,Z],I),'hipGraphLaunch':([P,P],I),
 'hipGraphDestroy':([P],I),'hipGraphExecDestroy':([P],I),'hipGraphGetNodes':([P,c.POINTER(P),c.POINTER(Z)],I),
 'hipGraphNodeSetEnabled':([P,P,U],I),'hipGraphExecKernelNodeSetParams':([P,P,c.POINTER(KernelParams)],I),
 'hipGraphKernelNodeGetParams':([P,c.POINTER(KernelParams)],I)
}
for name,(args,result) in specs.items():getattr(hip,name).argtypes=args;getattr(hip,name).restype=result

def ck(value,op):
 if value:raise RuntimeError(f'{op}: {value}: {hip.hipGetErrorString(value).decode()}')
def call(name,*args):ck(getattr(hip,name)(*args),name)

compiled=triton.compile(ASTSource(add,signature={'X':'*fp32','Z':'*fp32','Y':'*fp32','n':'i32','alpha':'fp32'},constexprs={'BLOCK':512}),target=GPUTarget('hip','gfx1032',32),options={'num_warps':4,'enable_fp_fusion':True})
(ROOT/'add.hsaco').write_bytes(compiled.asm['hsaco'])
if os.environ.get('SEJONG_PROBE_COMPILE_ONLY'):
 print('compiled');raise SystemExit
started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
call('hipInit',0);call('hipSetDevice',0)
name=c.create_string_buffer(256);call('hipDeviceGetName',name,256,0)
version=I();call('hipRuntimeGetVersion',c.byref(version))
stream=P();call('hipStreamCreate',c.byref(stream))
events=[P(),P()]
for e in events:call('hipEventCreate',c.byref(e))
module=P();function=P();blob=c.create_string_buffer(compiled.asm['hsaco'])
call('hipModuleLoadData',c.byref(module),blob);call('hipModuleGetFunction',c.byref(function),module,compiled.metadata.name.encode())
results=[]
for n,nodes in [(256,128),(256,1024),(256,4096),(256,14260),(65536,4096)]:
 state=P();delta=P();call('hipMalloc',c.byref(state),n*4);call('hipMalloc',c.byref(delta),n*4)
 zero=np.zeros(n,np.float32);one=np.ones(n,np.float32)
 call('hipMemcpy',delta,P(one.ctypes.data),one.nbytes,1)
 args=[state,delta,state,I(n),c.c_float(1),P(),P()]
 ptrs=(P*len(args))(*[c.cast(c.byref(x),P) for x in args])
 grid=((n+511)//512,1,1)
 def direct():
  for _ in range(nodes):
   ck(hip.hipModuleLaunchKernel(function,*grid,128,1,1,compiled.metadata.shared,stream,ptrs,None),'hipModuleLaunchKernel')
 call('hipMemcpy',state,P(zero.ctypes.data),zero.nbytes,1)
 direct();call('hipStreamSynchronize',stream)
 graph=P();executable=P();t=time.perf_counter()
 call('hipStreamBeginCapture',stream,0);direct();call('hipStreamEndCapture',stream,c.byref(graph));capture_seconds=time.perf_counter()-t
 count=Z();call('hipGraphGetNodes',graph,None,c.byref(count))
 t=time.perf_counter();call('hipGraphInstantiate',c.byref(executable),graph,None,None,0);instantiate_seconds=time.perf_counter()-t
 handles=(P*count.value)();call('hipGraphGetNodes',graph,handles,c.byref(count))
 call('hipGraphLaunch',executable,stream);call('hipStreamSynchronize',stream)
 samples=[[],[]]
 for repeat in range(2):
  for variant in [0,1,1,0]:
   call('hipMemcpy',state,P(zero.ctypes.data),zero.nbytes,1)
   call('hipEventRecord',events[0],stream);t=time.perf_counter()
   if variant:call('hipGraphLaunch',executable,stream)
   else:direct()
   enqueued=time.perf_counter();call('hipEventRecord',events[1],stream);call('hipStreamSynchronize',stream);finished=time.perf_counter()
   ms=c.c_float();call('hipEventElapsedTime',c.byref(ms),*events)
   samples[variant].append({'enqueue_seconds':enqueued-t,'synchronized_seconds':finished-t,'event_seconds':ms.value/1000})
 output=np.empty_like(zero);call('hipMemcpy',P(output.ctypes.data),state,output.nbytes,2)
 result={'elements':n,'nodes':nodes,'captured_nodes':count.value,'capture_seconds':capture_seconds,'instantiate_seconds':instantiate_seconds,'output_expected':nodes,'output_min':float(output.min()),'output_max':float(output.max()),'variants':{}}
 for label,rows in zip(['direct','graph'],samples):result['variants'][label]={key:{'minimum':min(x[key] for x in rows),'median':statistics.median(x[key] for x in rows),'maximum':max(x[key] for x in rows)} for key in rows[0]}
 if n==256 and nodes==128:
  call('hipGraphNodeSetEnabled',executable,handles[0],0);call('hipMemcpy',state,P(zero.ctypes.data),zero.nbytes,1);call('hipGraphLaunch',executable,stream);call('hipStreamSynchronize',stream);call('hipMemcpy',P(output.ctypes.data),state,output.nbytes,2)
  result['node_disabled_output_min_max']=[float(output.min()),float(output.max())]
  call('hipGraphNodeSetEnabled',executable,handles[0],1)
  changed=KernelParams();call('hipGraphKernelNodeGetParams',handles[0],c.byref(changed))
  new_args=[state,delta,state,I(n),c.c_float(2),P(),P()];new_ptrs=(P*len(new_args))(*[c.cast(c.byref(x),P) for x in new_args]);changed.kernelParams=new_ptrs
  t=time.perf_counter();call('hipGraphExecKernelNodeSetParams',executable,handles[0],c.byref(changed));result['single_node_parameter_update_seconds']=time.perf_counter()-t
  call('hipMemcpy',state,P(zero.ctypes.data),zero.nbytes,1);call('hipGraphLaunch',executable,stream);call('hipStreamSynchronize',stream);call('hipMemcpy',P(output.ctypes.data),state,output.nbytes,2)
  result['node_alpha_updated_output_min_max']=[float(output.min()),float(output.max())]
 result['synchronized_median_speedup']=result['variants']['direct']['synchronized_seconds']['median']/result['variants']['graph']['synchronized_seconds']['median']
 results.append(result);print(json.dumps(result),flush=True)
 call('hipGraphExecDestroy',executable);call('hipGraphDestroy',graph);call('hipFree',state);call('hipFree',delta)
call('hipModuleUnload',module)
for e in events:call('hipEventDestroy',e)
call('hipStreamDestroy',stream)
report={'started_utc':started_utc,'finished_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'device':name.value.decode(),'hip_runtime_version':version.value,'triton':metadata.version('triton'),'kernel':'actual Sejong add FP32, in-place accumulation of ones','measurement':'HIP event and host synchronized wall per captured chain; direct ctypes submissions vs graph replay; compilation and H2D reset outside timing; 2 ABBA rounds; warm direct and graph before samples','artifact_sha256':hashlib.sha256(compiled.asm['hsaco']).hexdigest(),'results':results}
(ROOT/'result.json').write_text(json.dumps(report,indent=2)+'\n')
