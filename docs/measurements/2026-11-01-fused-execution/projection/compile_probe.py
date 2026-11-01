import json,re
from pathlib import Path
from collections import Counter
import triton as tr
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from baseline_kernels import matmul,swiglu

@tr.jit
def gate_up_fused(X,WG,WU,G,U,H,M,N,K,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
 m=tl.program_id(0)*BM+tl.arange(0,BM)
 n=tl.program_id(1)*BN+tl.arange(0,BN)
 k=tl.arange(0,BK)
 ag=tl.full((BM,BN),0,tl.float32)
 au=tl.full((BM,BN),0,tl.float32)
 for start in range(tl.cdiv(K,BK)):
  kk=start*BK+k
  x=tl.load(X+m[:,None]*K+kk[None,:],(m[:,None]<M)&(kk[None,:]<K),0).to(tl.float32)
  wg=tl.load(WG+kk[:,None]*N+n[None,:],(kk[:,None]<K)&(n[None,:]<N),0).to(tl.float32)
  wu=tl.load(WU+kk[:,None]*N+n[None,:],(kk[:,None]<K)&(n[None,:]<N),0).to(tl.float32)
  ag+=tl.dot(x,wg,input_precision='ieee')
  au+=tl.dot(x,wu,input_precision='ieee')
 g=ag.to(G.dtype.element_ty).to(tl.float32)
 u=au.to(U.dtype.element_ty).to(tl.float32)
 idx=m[:,None]*N+n[None,:]
 mask=(m[:,None]<M)&(n[None,:]<N)
 tl.store(G+idx,g,mask)
 tl.store(U+idx,u,mask)
 tl.store(H+idx,g*tl.sigmoid(g)*u,mask)

ROOT=Path(__file__).resolve().parent
records=[]
def compile_one(fn,sig,constants,key):
 c=tr.compile(ASTSource(fn,signature=sig,constexprs=constants),target=GPUTarget('hip','gfx1032',32),options={'num_warps':4,'enable_fp_fusion':True})
 asm=c.asm['amdgcn'];meta=c.metadata
 for ext in ['amdgcn','ttgir','llir','hsaco']:
  val=c.asm[ext];p=ROOT/(key+'.'+ext)
  p.write_bytes(val) if isinstance(val,bytes) else p.write_text(val)
 def field(n):
  x=re.search(r'\.'+n+r':\s*(\d+)',asm);return int(x.group(1)) if x else None
 rec={'key':key,'symbol':meta.name,'shared':meta.shared,'threads':meta.num_warps*meta.warp_size,'vgprs':field('vgpr_count'),'sgprs':field('sgpr_count'),'private':field('private_segment_fixed_size'),'global_scratch':meta.global_scratch_size,'profile_scratch':meta.profile_scratch_size,'static_instructions':dict(Counter(re.findall(r'^\s*(v_\w+|ds_\w+|global_\w+|s_barrier)\b',asm,re.M)))}
 records.append(rec)
compile_one(matmul,dict(A='*fp16',B='*fp16',C='*fp16',M='i32',N='i32',K='i32'),dict(TA=False,TB=False,BM=16,BN=32,BK=32),'matmul')
compile_one(swiglu,dict(G='*fp16',U='*fp16',Y='*fp16',n='i32'),dict(BLOCK=256),'swiglu')
compile_one(gate_up_fused,dict(X='*fp16',WG='*fp16',WU='*fp16',G='*fp16',U='*fp16',H='*fp16',M='i32',N='i32',K='i32'),dict(BM=16,BN=32,BK=32),'gate_up_fused')
(ROOT/'compile.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps([{k:v for k,v in r.items() if k!='static_instructions'} for r in records],indent=2))
