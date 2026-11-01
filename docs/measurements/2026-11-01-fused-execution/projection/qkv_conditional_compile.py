import json,re
from pathlib import Path
import triton as tr
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from baseline_kernels import matmul
@tr.jit
def qkv_conditional(X,WQ,WK,WV,Q,KOUT,VOUT,M,D,KWIDTH,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
 m=tl.program_id(0)*BM+tl.arange(0,BM)
 n=tl.program_id(1)*BN+tl.arange(0,BN)
 k=tl.arange(0,BK)
 aq=tl.full((BM,BN),0,tl.float32)
 if tl.program_id(1)*BN<KWIDTH:
  ak=tl.full((BM,BN),0,tl.float32)
  av=tl.full((BM,BN),0,tl.float32)
  for start in range(tl.cdiv(D,BK)):
   kk=start*BK+k
   x=tl.load(X+m[:,None]*D+kk[None,:],(m[:,None]<M)&(kk[None,:]<D),0).to(tl.float32)
   wq=tl.load(WQ+kk[:,None]*D+n[None,:],(kk[:,None]<D)&(n[None,:]<D),0).to(tl.float32)
   wk=tl.load(WK+kk[:,None]*KWIDTH+n[None,:],(kk[:,None]<D)&(n[None,:]<KWIDTH),0).to(tl.float32)
   wv=tl.load(WV+kk[:,None]*KWIDTH+n[None,:],(kk[:,None]<D)&(n[None,:]<KWIDTH),0).to(tl.float32)
   aq+=tl.dot(x,wq,input_precision='ieee')
   ak+=tl.dot(x,wk,input_precision='ieee')
   av+=tl.dot(x,wv,input_precision='ieee')
  tl.store(KOUT+m[:,None]*KWIDTH+n[None,:],ak,(m[:,None]<M)&(n[None,:]<KWIDTH))
  tl.store(VOUT+m[:,None]*KWIDTH+n[None,:],av,(m[:,None]<M)&(n[None,:]<KWIDTH))
 else:
  for start in range(tl.cdiv(D,BK)):
   kk=start*BK+k
   x=tl.load(X+m[:,None]*D+kk[None,:],(m[:,None]<M)&(kk[None,:]<D),0).to(tl.float32)
   wq=tl.load(WQ+kk[:,None]*D+n[None,:],(kk[:,None]<D)&(n[None,:]<D),0).to(tl.float32)
   aq+=tl.dot(x,wq,input_precision='ieee')
 tl.store(Q+m[:,None]*D+n[None,:],aq,(m[:,None]<M)&(n[None,:]<D))
ROOT=Path(__file__).resolve().parent
rows=[]
def compile_one(fn,sig,cs,key,bn):
 c=tr.compile(ASTSource(fn,signature=sig,constexprs=cs),target=GPUTarget('hip','gfx1032',32),options={'num_warps':4,'enable_fp_fusion':True})
 asm=c.asm['amdgcn'];meta=c.metadata
 for ext in ['amdgcn','ttgir','llir','hsaco']:
  val=c.asm[ext];p=ROOT/(key+'.'+ext);p.write_bytes(val) if isinstance(val,bytes) else p.write_text(val)
 def f(n):
  x=re.search(r'\.'+n+r':\s*(\d+)',asm);return int(x.group(1)) if x else None
 rows.append({'key':key,'symbol':meta.name,'BN':bn,'shared':meta.shared,'threads':meta.num_warps*meta.warp_size,'vgprs':f('vgpr_count'),'sgprs':f('sgpr_count'),'private':f('private_segment_fixed_size'),'global_scratch':meta.global_scratch_size})
compile_one(matmul,dict(A='*fp16',B='*fp16',C='*fp16',M='i32',N='i32',K='i32'),dict(TA=False,TB=False,BM=16,BN=32,BK=32),'matmul_qkv',32)
for bn in [32,16]:
 sig={**{x:'*fp16' for x in ['X','WQ','WK','WV','Q','KOUT','VOUT']},**{x:'i32' for x in ['M','D','KWIDTH']}}
 compile_one(qkv_conditional,sig,dict(BM=16,BN=bn,BK=32),'qkv_conditional_'+str(bn),bn)
(ROOT/'qkv-conditional-compile.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
