import json,re
from pathlib import Path
import triton as tr
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from baseline_kernels import matmul,add
@tr.jit
def gate_up_dx(DG,DU,WG,WU,DX,M,N,K,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
 m=tl.program_id(0)*BM+tl.arange(0,BM)
 n=tl.program_id(1)*BN+tl.arange(0,BN)
 k=tl.arange(0,BK)
 ag=tl.full((BM,BN),0,tl.float32)
 au=tl.full((BM,BN),0,tl.float32)
 for start in range(tl.cdiv(K,BK)):
  kk=start*BK+k
  dg=tl.load(DG+m[:,None]*K+kk[None,:],(m[:,None]<M)&(kk[None,:]<K),0).to(tl.float32)
  du=tl.load(DU+m[:,None]*K+kk[None,:],(m[:,None]<M)&(kk[None,:]<K),0).to(tl.float32)
  wg=tl.load(WG+n[None,:]*K+kk[:,None],(kk[:,None]<K)&(n[None,:]<N),0).to(tl.float32)
  wu=tl.load(WU+n[None,:]*K+kk[:,None],(kk[:,None]<K)&(n[None,:]<N),0).to(tl.float32)
  ag+=tl.dot(dg,wg,input_precision='ieee')
  au+=tl.dot(du,wu,input_precision='ieee')
 tl.store(DX+m[:,None]*N+n[None,:],ag+au,(m[:,None]<M)&(n[None,:]<N))
ROOT=Path(__file__).resolve().parent
rows=[]
def compile_one(fn,sig,cs,key):
 c=tr.compile(ASTSource(fn,signature=sig,constexprs=cs),target=GPUTarget('hip','gfx1032',32),options={'num_warps':4,'enable_fp_fusion':True})
 asm=c.asm['amdgcn'];meta=c.metadata
 for ext in ['amdgcn','ttgir','llir','hsaco']:
  val=c.asm[ext];p=ROOT/(key+'.'+ext);p.write_bytes(val) if isinstance(val,bytes) else p.write_text(val)
 def f(n):
  x=re.search(r'\.'+n+r':\s*(\d+)',asm);return int(x.group(1)) if x else None
 rows.append({'key':key,'symbol':meta.name,'shared':meta.shared,'threads':meta.num_warps*meta.warp_size,'vgprs':f('vgpr_count'),'sgprs':f('sgpr_count'),'private':f('private_segment_fixed_size'),'global_scratch':meta.global_scratch_size})
compile_one(matmul,dict(A='*fp32',B='*fp16',C='*fp32',M='i32',N='i32',K='i32'),dict(TA=False,TB=True,BM=16,BN=32,BK=32),'matmul_dx')
compile_one(add,dict(X='*fp32',Z='*fp32',Y='*fp32',n='i32',alpha='fp32'),dict(BLOCK=256),'add_dx')
compile_one(gate_up_dx,dict(DG='*fp32',DU='*fp32',WG='*fp16',WU='*fp16',DX='*fp32',M='i32',N='i32',K='i32'),dict(BM=16,BN=32,BK=32),'gate_up_dx')
(ROOT/'dx-compile.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
