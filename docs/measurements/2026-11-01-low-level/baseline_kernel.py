import triton as tr
import triton.language as tl

@tr.jit
def matmul(A,B,C,M,N,K,TA:tl.constexpr,TB:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
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
        a=tl.load(A+ap,(m[:,None]<M)&(kk[None,:]<K),0).to(tl.float32)
        b=tl.load(B+bp,(kk[:,None]<K)&(n[None,:]<N),0).to(tl.float32)
        accum+=tl.dot(a,b,input_precision="ieee")
    tl.store(C+m[:,None]*N+n[None,:],accum,(m[:,None]<M)&(n[None,:]<N))

