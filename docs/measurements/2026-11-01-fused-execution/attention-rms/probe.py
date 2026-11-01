import ctypes as c
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import triton as tr
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from baseline_kernels import fill, rmsnorm_backward
from baseline_kernels import hierarchy_attention, hierarchy_attention_dq, hierarchy_attention_dkv

ROOT = Path(__file__).resolve().parent
EPS = 2.0 ** -23

@tr.jit
def rms_partial(X,W,DY,DX,PART,R,D:tl.constexpr,EPS:tl.constexpr,BR:tl.constexpr,BD:tl.constexpr):
    group=tl.program_id(0)
    r=group*BR+tl.arange(0,BR)
    j=tl.arange(0,BD)
    x=tl.load(X+r[:,None]*D+j[None,:],(r[:,None]<R)&(j[None,:]<D),0).to(tl.float32)
    w=tl.load(W+j,j<D,0).to(tl.float32)
    dy=tl.load(DY+r[:,None]*D+j[None,:],(r[:,None]<R)&(j[None,:]<D),0).to(tl.float32)
    inv=tl.rsqrt(tl.sum(x*x,1)/D+EPS)
    d=dy*w[None,:]
    dx=inv[:,None]*(d-x*(tl.sum(d*x,1)/D)[:,None]*inv[:,None]*inv[:,None])
    tl.store(DX+r[:,None]*D+j[None,:],dx,(r[:,None]<R)&(j[None,:]<D))
    contribution=dy*x*inv[:,None]
    partial=tl.sum(contribution,0)
    tl.store(PART+group*D+j,partial,j<D)

@tr.jit
def rms_finish(PART,DW,G,D:tl.constexpr,BG:tl.constexpr,BD:tl.constexpr):
    j=tl.program_id(0)*BD+tl.arange(0,BD)
    g=tl.arange(0,BG)
    x=tl.load(PART+g[:,None]*D+j[None,:],(g[:,None]<G)&(j[None,:]<D),0)
    tl.store(DW+j,tl.sum(x,0),j<D)

@tr.jit
def attention_dq_delta(Q,K,V,O,LSE,DO,DQ,Delta,T,HQ:tl.constexpr,HK:tl.constexpr,D:tl.constexpr,BD:tl.constexpr,BK:tl.constexpr,WINDOW:tl.constexpr):
    row=tl.program_id(0);head=row%HQ;position=(row//HQ)%T;batch=row//(HQ*T)
    kv_head=head//(HQ//HK);d=tl.arange(0,BD);j=tl.arange(0,BK)
    q=tl.load(Q+row*D+d,d<D,0).to(tl.float32)
    dy=tl.load(DO+row*D+d,d<D,0).to(tl.float32)
    output=tl.load(O+row*D+d,d<D,0).to(tl.float32)
    delta=tl.sum(dy*output,0)
    tl.store(Delta+row,delta)
    lse=tl.load(LSE+row);result=tl.full((BD,),0,tl.float32)
    first=tl.maximum(0,position+1-WINDOW) if WINDOW>0 else 0
    for start in range(first,position+1,BK):
        key_position=start+j;valid=key_position<=position
        indices=((batch*T+key_position[:,None])*HK+kv_head)*D+d[None,:]
        k=tl.load(K+indices,valid[:,None]&(d[None,:]<D),0).to(tl.float32)
        v=tl.load(V+indices,valid[:,None]&(d[None,:]<D),0).to(tl.float32)
        probability=tl.exp(tl.sum(k*q[None,:],1)*(D**-0.5)-lse)
        ds=tl.where(valid,probability*(tl.sum(v*dy[None,:],1)-delta)*(D**-0.5),0)
        result+=tl.sum(k*ds[:,None],0)
    tl.store(DQ+row*D+d,result,d<D)

@tr.jit
def attention_dkv_delta(Q,K,V,LSE,DO,Delta,DK,DV,T,HQ:tl.constexpr,HK:tl.constexpr,D:tl.constexpr,BD:tl.constexpr,BK:tl.constexpr,WINDOW:tl.constexpr):
    row=tl.program_id(0);kv_head=row%HK;position=(row//HK)%T;batch=row//(HK*T)
    d=tl.arange(0,BD);j=tl.arange(0,BK)
    k=tl.load(K+row*D+d,d<D,0).to(tl.float32)
    v=tl.load(V+row*D+d,d<D,0).to(tl.float32)
    dk=tl.full((BD,),0,tl.float32);dv=tl.full((BD,),0,tl.float32)
    end=tl.minimum(T,position+WINDOW) if WINDOW>0 else T
    for group in range(HQ//HK):
        head=kv_head*(HQ//HK)+group
        for start in range(position,end,BK):
            query_position=start+j;valid=query_position<end
            query_row=(batch*T+query_position)*HQ+head
            indices=query_row[:,None]*D+d[None,:]
            q=tl.load(Q+indices,valid[:,None]&(d[None,:]<D),0).to(tl.float32)
            dy=tl.load(DO+indices,valid[:,None]&(d[None,:]<D),0).to(tl.float32)
            lse=tl.load(LSE+query_row,valid,0)
            probability=tl.where(valid,tl.exp(tl.sum(q*k[None,:],1)*(D**-0.5)-lse),0)
            delta=tl.load(Delta+query_row,valid,0)
            ds=probability*(tl.sum(dy*v[None,:],1)-delta)*(D**-0.5)
            dk+=tl.sum(q*ds[:,None],0);dv+=tl.sum(dy*probability[:,None],0)
    tl.store(DK+row*D+d,dk,d<D);tl.store(DV+row*D+d,dv,d<D)

P=c.c_void_p;I=c.c_int;U=c.c_uint;Z=c.c_size_t
hip=c.CDLL(str(Path(os.environ['ROCM_PATH'])/'lib/libamdhip64.so.7'))
specs={'hipInit':([U],I),'hipSetDevice':([I],I),'hipGetErrorString':([I],c.c_char_p),
 'hipMalloc':([c.POINTER(P),Z],I),'hipFree':([P],I),'hipMemcpy':([P,P,Z,I],I),
 'hipStreamCreate':([c.POINTER(P)],I),'hipStreamSynchronize':([P],I),'hipStreamDestroy':([P],I),
 'hipEventCreate':([c.POINTER(P)],I),'hipEventRecord':([P,P],I),'hipEventElapsedTime':([c.POINTER(c.c_float),P,P],I),'hipEventDestroy':([P],I),
 'hipModuleLoadData':([c.POINTER(P),P],I),'hipModuleGetFunction':([c.POINTER(P),P,c.c_char_p],I),'hipModuleUnload':([P],I),
 'hipModuleLaunchKernel':([P,U,U,U,U,U,U,U,P,c.POINTER(P),c.POINTER(P)],I)}
for key,(args,result) in specs.items():f=getattr(hip,key);f.argtypes=args;f.restype=result
def call(name,*args):
    value=getattr(hip,name)(*args)
    if value:raise RuntimeError(f'{name}: {hip.hipGetErrorString(value).decode()}')
call('hipInit',0);call('hipSetDevice',0)
stream=P();call('hipStreamCreate',c.byref(stream))
events=[P(),P()]
for e in events:call('hipEventCreate',c.byref(e))
allocations=[];modules=[];compiled_records=[]
def alloc(size):
    p=P();call('hipMalloc',c.byref(p),int(size));allocations.append(p);return p
def upload(x):
    x=np.ascontiguousarray(x);p=alloc(x.nbytes);call('hipMemcpy',p,P(x.ctypes.data),x.nbytes,1);return p
def download(p,shape,dtype=np.float32):
    x=np.empty(shape,dtype=dtype);call('hipMemcpy',P(x.ctypes.data),p,x.nbytes,2);return x
def compile_kernel(fn,signature,constants,key,warps=4):
    obj=tr.compile(ASTSource(fn,signature=signature,constexprs=constants),target=GPUTarget('hip','gfx1032',32),options={'num_warps':warps,'enable_fp_fusion':True})
    binary=obj.asm['hsaco'];(ROOT/f'{key}.hsaco').write_bytes(binary)
    (ROOT/f'{key}.amdgcn').write_text(obj.asm['amdgcn'])
    m=P();f=P();blob=c.create_string_buffer(binary);call('hipModuleLoadData',c.byref(m),blob);call('hipModuleGetFunction',c.byref(f),m,obj.metadata.name.encode());modules.append(m)
    compiled_records.append({'key':key,'shared_bytes':obj.metadata.shared,'threads':obj.metadata.num_warps*obj.metadata.warp_size,'global_scratch':obj.metadata.global_scratch_size,'binary_sha256':hashlib.sha256(binary).hexdigest()})
    return (f,obj.metadata.shared,obj.metadata.num_warps*obj.metadata.warp_size)
def operation(kernel,args,grid):
    arguments=[x if isinstance(x,(P,I,c.c_float)) else I(x) for x in args]+[P(),P()]
    pointers=(P*len(arguments))(*[c.cast(c.byref(x),P) for x in arguments])
    return kernel,arguments,pointers,grid
def run(ops,count=1):
    launch=hip.hipModuleLaunchKernel
    for _ in range(count):
        for (f,shared,threads),arguments,pointers,grid in ops:
            code=launch(f,*grid,threads,1,1,shared,stream,pointers,None)
            if code:raise RuntimeError(hip.hipGetErrorString(code).decode())
def measure(variants,batch=128,rounds=4):
    names=list(variants);samples={n:[] for n in names}
    for n in names:run(variants[n],16)
    call('hipStreamSynchronize',stream)
    for _ in range(rounds):
        for n in names+list(reversed(names)):
            call('hipEventRecord',events[0],stream);start=time.perf_counter();run(variants[n],batch);enqueue=time.perf_counter()-start
            call('hipEventRecord',events[1],stream);call('hipStreamSynchronize',stream)
            ms=c.c_float();call('hipEventElapsedTime',c.byref(ms),*events)
            samples[n].append({'event_us':ms.value*1000/batch,'enqueue_us':enqueue*1e6/batch})
    return {n:{'median_us':statistics.median(x['event_us'] for x in xs),'minimum_us':min(x['event_us'] for x in xs),'enqueue_median_us':statistics.median(x['enqueue_us'] for x in xs),'samples':xs} for n,xs in samples.items()}
def error(x,y):
    x=x.astype(np.float64);y=y.astype(np.float64);d=x-y
    return {'max_abs':float(np.abs(d).max()),'relative_l2':float(np.linalg.norm(d)/max(np.linalg.norm(y),np.finfo(float).tiny))}

started=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
final_only = os.environ.get('SEJONG_PROBE_FINAL_ONLY') == '1'
rng=np.random.default_rng(20261101);results={'rmsnorm':[],'attention':[]}
fill_kernel=compile_kernel(fill,{'Y':'*fp32','n':'i32','value':'fp32'},{'BLOCK':256},'fill')
for R,D in ([(512,512)] if final_only else [(64,512),(512,128),(1024,512),(129,257)]):
    x=rng.normal(0,.5,(R,D)).astype(np.float16);w=rng.normal(1,.2,D).astype(np.float16);dy=rng.normal(0,.5,(R,D)).astype(np.float32)
    xp,wp,dyp=map(upload,[x,w,dy]);dx0=alloc(R*D*4);dw0=alloc(D*4)
    base=compile_kernel(rmsnorm_backward,{'X':'*fp16','W':'*fp16','DY':'*fp32','DX':'*fp32','DW':'*fp32'},{'D':D,'EPS':EPS,'BLOCK':tr.next_power_of_2(D)},f'rms-base-{D}')
    variants={'atomic':[operation(fill_kernel,[dw0,D,c.c_float(0)],((D+255)//256,1,1)),operation(base,[xp,wp,dyp,dx0,dw0],(R,1,1))]}
    outputs={'atomic':(dx0,dw0)};details={}
    for BR in [4,8,16]:
        G=(R+BR-1)//BR;partial=alloc(G*D*4);dx=alloc(R*D*4);dw=alloc(D*4)
        part=compile_kernel(rms_partial,{'X':'*fp16','W':'*fp16','DY':'*fp32','DX':'*fp32','PART':'*fp32','R':'i32'},{'D':D,'EPS':EPS,'BR':BR,'BD':tr.next_power_of_2(D)},f'rms-part-{D}-{BR}')
        final=compile_kernel(rms_finish,{'PART':'*fp32','DW':'*fp32','G':'i32'},{'D':D,'BG':tr.next_power_of_2(G),'BD':32},f'rms-final-{R}-{D}-{BR}')
        name=f'partial_rows_{BR}';variants[name]=[operation(part,[xp,wp,dyp,dx,partial,R],(G,1,1)),operation(final,[partial,dw,G],((D+31)//32,1,1))];outputs[name]=(dx,dw);details[name]={'partial_bytes':G*D*4,'groups':G}
    timing=measure(variants)
    xd=x.astype(np.float64);wd=w.astype(np.float64);dyd=dy.astype(np.float64)
    inv=1/np.sqrt((xd*xd).mean(1)+EPS);d=dyd*wd
    dxref=inv[:,None]*(d-xd*(d*xd).mean(1)[:,None]*inv[:,None]**2);dwref=(dyd*xd*inv[:,None]).sum(0)
    values={n:(download(p[0],(R,D)),download(p[1],(D,))) for n,p in outputs.items()}
    compare={n:{'dx_vs_fp64':error(v[0],dxref),'dw_vs_fp64':error(v[1],dwref),'dx_vs_atomic':error(v[0],values['atomic'][0]),'dw_vs_atomic':error(v[1],values['atomic'][1])} for n,v in values.items()}
    row={'rows':R,'width':D,'timing':timing,'scratch':details,'error':compare};results['rmsnorm'].append(row)
    print(json.dumps({'kind':'rmsnorm',**row}),flush=True)

for B,T,HQ,HK,D,WINDOW in ([] if final_only else [(1,512,2,2,64,128),(1,64,8,4,64,0),(2,33,4,2,32,17)]):
    shape=(B,T,HQ,D);kvshape=(B,T,HK,D)
    q=rng.normal(0,.5,shape).astype(np.float16);k=rng.normal(0,.5,kvshape).astype(np.float16);v=rng.normal(0,.5,kvshape).astype(np.float16);dy=rng.normal(0,.5,shape).astype(np.float32)
    qp,kp,vp,dyp=map(upload,[q,k,v,dy]);op=alloc(q.nbytes);lse=alloc(B*T*HQ*4);delta=alloc(B*T*HQ*4)
    constants={'HQ':HQ,'HK':HK,'D':D,'BD':tr.next_power_of_2(D),'BK':min(32,tr.next_power_of_2(T)),'WINDOW':WINDOW}
    fwd=compile_kernel(hierarchy_attention,{'Q':'*fp16','K':'*fp16','V':'*fp16','O':'*fp16','LSE':'*fp32','TQ':'i32','TK':'i32','OFFSET':'i32'},{**constants,'FP64':False},f'attention-fwd-{T}-{HQ}-{HK}-{D}-{WINDOW}')
    run([operation(fwd,[qp,kp,vp,op,lse,T,T,0],(B*T*HQ,1,1))]);call('hipStreamSynchronize',stream)
    dq0,dk0,dv0=[alloc(n*4) for n in [q.size,k.size,k.size]];dq1,dk1,dv1=[alloc(n*4) for n in [q.size,k.size,k.size]]
    common={'Q':'*fp16','K':'*fp16','V':'*fp16','O':'*fp16','LSE':'*fp32','DO':'*fp32'}
    aq=compile_kernel(hierarchy_attention_dq,{**common,'DQ':'*fp32','T':'i32'},constants,f'attention-dq-base-{T}-{HQ}-{HK}-{D}-{WINDOW}')
    akv=compile_kernel(hierarchy_attention_dkv,{**common,'DK':'*fp32','DV':'*fp32','T':'i32'},constants,f'attention-dkv-base-{T}-{HQ}-{HK}-{D}-{WINDOW}')
    bq=compile_kernel(attention_dq_delta,{**common,'DQ':'*fp32','Delta':'*fp32','T':'i32'},constants,f'attention-dq-delta-{T}-{HQ}-{HK}-{D}-{WINDOW}')
    bkv=compile_kernel(attention_dkv_delta,{'Q':'*fp16','K':'*fp16','V':'*fp16','LSE':'*fp32','DO':'*fp32','Delta':'*fp32','DK':'*fp32','DV':'*fp32','T':'i32'},constants,f'attention-dkv-delta-{T}-{HQ}-{HK}-{D}-{WINDOW}')
    variants={'recompute':[operation(aq,[qp,kp,vp,op,lse,dyp,dq0,T],(B*T*HQ,1,1)),operation(akv,[qp,kp,vp,op,lse,dyp,dk0,dv0,T],(B*T*HK,1,1))],
              'share_delta':[operation(bq,[qp,kp,vp,op,lse,dyp,dq1,delta,T],(B*T*HQ,1,1)),operation(bkv,[qp,kp,vp,lse,dyp,delta,dk1,dv1,T],(B*T*HK,1,1))]}
    timing=measure(variants,batch=64)
    o=download(op,shape,np.float16).astype(np.float64);l=download(lse,(B,T,HQ)).astype(np.float64)
    qd,kd,vd,dyd=[a.astype(np.float64) for a in [q,k,v,dy]]
    dqref=np.zeros(shape);dkref=np.zeros(kvshape);dvref=np.zeros(kvshape)
    mask=np.arange(T)[None,:]<=np.arange(T)[:,None]
    if WINDOW:mask &= np.arange(T)[None,:]>=(np.arange(T)[:,None]+1-WINDOW)
    for b in range(B):
        for h in range(HQ):
            kh=h//(HQ//HK);qq=qd[b,:,h];kk=kd[b,:,kh];vv=vd[b,:,kh];dd=dyd[b,:,h]
            scores=(qq@kk.T)*(D**-.5);p=np.where(mask,np.exp(scores-l[b,:,h,None]),0)
            de=(dd*o[b,:,h]).sum(1);ds=p*((dd@vv.T)-de[:,None])*(D**-.5)
            dqref[b,:,h]=ds@kk;dkref[b,:,kh]+=ds.T@qq;dvref[b,:,kh]+=p.T@dd
    baseline=[download(p,s) for p,s in [(dq0,shape),(dk0,kvshape),(dv0,kvshape)]];shared=[download(p,s) for p,s in [(dq1,shape),(dk1,kvshape),(dv1,kvshape)]]
    compare={name:{'share_vs_baseline':error(a,b),'baseline_vs_fp64':error(b,r),'share_vs_fp64':error(a,r)} for name,a,b,r in zip(['dq','dk','dv'],shared,baseline,[dqref,dkref,dvref])}
    row={'batch':B,'sequence':T,'query_heads':HQ,'kv_heads':HK,'head_width':D,'window':WINDOW,'delta_bytes':B*T*HQ*4,'timing':timing,'error':compare};results['attention'].append(row);print(json.dumps({'kind':'attention',**row}),flush=True)

for p in allocations:call('hipFree',p)
for m in modules:call('hipModuleUnload',m)
for e in events:call('hipEventDestroy',e)
call('hipStreamDestroy',stream)
results.update({'started_utc':started,'finished_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'target':'gfx1032','wave_size':32,'warps':4,'seed':20261101,'compiled':compiled_records,'protocol':'Dedicated GPU; HIP events bracket batches of kernel pairs; allocation/compilation/copies outside timing; each variant forward then reverse per round. Event times include idle host submission gaps. RMS baseline includes required DW fill; partial uses partial plus final. Attention compares existing dQ+dKV versus dQ stores Delta then dKV reads Delta.'})
results['probe_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(ROOT/('final-result.json' if final_only else 'result.json')).write_text(json.dumps(results,indent=2)+'\n')
