import json
import re
from pathlib import Path
from collections import Counter
import triton as tr
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from baseline_kernel import matmul

ROOT = Path(__file__).resolve().parent

@tr.jit
def native_half_matmul(A,B,C,M,N,K,TA:tl.constexpr,TB:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    m=tl.program_id(0)*BM+tl.arange(0,BM)
    n=tl.program_id(1)*BN+tl.arange(0,BN)
    k=tl.arange(0,BK)
    accum=tl.full((BM,BN),0,tl.float32)
    for start in range(tl.cdiv(K,BK)):
        kk=start*BK+k
        if TA: ap=kk[None,:]*M+m[:,None]
        else: ap=m[:,None]*K+kk[None,:]
        if TB: bp=n[None,:]*K+kk[:,None]
        else: bp=kk[:,None]*N+n[None,:]
        a=tl.load(A+ap,(m[:,None]<M)&(kk[None,:]<K),0)
        b=tl.load(B+bp,(kk[:,None]<K)&(n[None,:]<N),0)
        accum+=tl.dot(a,b,input_precision="ieee")
    tl.store(C+m[:,None]*N+n[None,:],accum,(m[:,None]<M)&(n[None,:]<N))

def inspect(fn,ta,tb):
    compiled=tr.compile(ASTSource(fn,signature={'A':'*fp16','B':'*fp16','C':'*fp16','M':'i32','N':'i32','K':'i32'},constexprs={'TA':ta,'TB':tb,'BM':16,'BN':32,'BK':32}), target=GPUTarget('hip','gfx1032',32), options={'num_warps':4,'enable_fp_fusion':True})
    asm=compiled.asm['amdgcn']
    metadata=compiled.metadata
    ops=Counter(re.findall(r'^\s*(v_\w+|ds_\w+|global_\w+|s_barrier)\b',asm,re.M))
    def field(name):
        match=re.search(r'\.'+name+r':\s*(\d+)',asm)
        return int(match.group(1)) if match else None
    key=f'{fn.__name__}-{int(ta)}-{int(tb)}'
    output=ROOT
    output.mkdir(parents=True,exist_ok=True)
    for ext in ['amdgcn','ttgir','llir','hsaco']:
        value=compiled.asm[ext]
        path=output/f'{key}.{ext}'
        path.write_bytes(value) if isinstance(value,bytes) else path.write_text(value)
    return {'kernel':key,'target':'gfx1032','wave_size':metadata.warp_size,'num_warps':metadata.num_warps,'tile':[16,32,32],'shared_bytes':metadata.shared,'global_scratch_bytes':metadata.global_scratch_size,'profile_scratch_bytes':metadata.profile_scratch_size,'vgprs':field('vgpr_count'),'sgprs':field('sgpr_count'),'private_bytes':field('private_segment_fixed_size'),'static_instructions':dict(ops)}

if __name__ == '__main__':
    results=[inspect(fn,ta,tb) for ta,tb in [(False,False),(False,True),(True,False)] for fn in [matmul,native_half_matmul]]
    path=ROOT/'dot2-compile.json'
    path.write_text(json.dumps(results,indent=2)+'\n')
    for row in results:
        row=dict(row)
        row['static_instructions']={k:v for k,v in row['static_instructions'].items() if 'fma' in k or 'dot' in k or 'cvt' in k or k.startswith('global_') or k=='s_barrier'}
        print(json.dumps(row))
