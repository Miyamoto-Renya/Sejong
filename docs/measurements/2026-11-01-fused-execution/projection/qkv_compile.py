import json,re
from pathlib import Path
import triton as tr
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
@tr.jit
def qkv_shared(X,WQ,WK,WV,Q,KOUT,VOUT,M,D,KWIDTH,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
 m=tl.program_id(0)*BM+tl.arange(0,BM)
 n=tl.program_id(1)*BN+tl.arange(0,BN)
 k=tl.arange(0,BK)
 aq=tl.full((BM,BN),0,tl.float32)
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
 tl.store(Q+m[:,None]*D+n[None,:],aq,(m[:,None]<M)&(n[None,:]<D))
 tl.store(KOUT+m[:,None]*KWIDTH+n[None,:],ak,(m[:,None]<M)&(n[None,:]<KWIDTH))
 tl.store(VOUT+m[:,None]*KWIDTH+n[None,:],av,(m[:,None]<M)&(n[None,:]<KWIDTH))
ROOT=Path(__file__).resolve().parent
sig={**{x:'*fp16' for x in ['X','WQ','WK','WV','Q','KOUT','VOUT']},**{x:'i32' for x in ['M','D','KWIDTH']}}
rows=[]
for bm,bn,bk,warps in [(16,32,32,4),(16,16,32,4),(8,32,32,4)]:
 c=tr.compile(ASTSource(qkv_shared,signature=sig,constexprs={'BM':bm,'BN':bn,'BK':bk}),target=GPUTarget('hip','gfx1032',32),options={'num_warps':warps,'enable_fp_fusion':True})
 asm=c.asm['amdgcn']
 def f(n):
  x=re.search(r'\.'+n+r':\s*(\d+)',asm);return int(x.group(1)) if x else None
 key=f'qkv-{bm}-{bn}-{bk}'
 (ROOT/(key+'.amdgcn')).write_text(asm)
 rows.append({'key':key,'tile_MNK':[bm,bn,bk],'warps':warps,'vgprs':f('vgpr_count'),'private':f('private_segment_fixed_size'),'shared':c.metadata.shared,'global_scratch':c.metadata.global_scratch_size})
(ROOT/'qkv-compile.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
