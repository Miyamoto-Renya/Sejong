"""Sejong's tensor, Transformer, backward and optimizer GPU operations."""
import triton as tr
import triton.language as tl
from triton.language.extra import libdevice as math

@tr.jit
def fill(Y, n, value, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y+i, value, i<n)

@tr.jit
def copy(X, Y, n, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    tl.store(Y+i, tl.load(X+i,i<n,0), i<n)

@tr.jit
def add(X, Z, Y, n, alpha, BLOCK: tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    x=tl.load(X+i,i<n,0)
    z=tl.load(Z+i,i<n,0)
    tl.store(Y+i,x+alpha*z,i<n)

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

@tr.jit
def embedding(W,Ids,Y,N,D,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    token=tl.load(Ids+i//D,i<N*D,0)
    tl.store(Y+i,tl.load(W+token*D+i%D,i<N*D,0),i<N*D)

@tr.jit
def embedding_backward(DY,Ids,DW,N,D,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    token=tl.load(Ids+i//D,i<N*D,0)
    dy=tl.load(DY+i,i<N*D,0).to(tl.float32)
    tl.atomic_add(DW+token*D+i%D,dy,i<N*D,sem="relaxed")

@tr.jit
def rmsnorm(X,W,Y,D:tl.constexpr,EPS:tl.constexpr,BLOCK:tl.constexpr):
    r=tl.program_id(0); j=tl.arange(0,BLOCK)
    dtype:tl.constexpr=tl.float64 if X.dtype.element_ty==tl.float64 else tl.float32
    x=tl.load(X+r*D+j,j<D,0).to(dtype)
    w=tl.load(W+j,j<D,0).to(dtype)
    inv=1/math.sqrt(tl.sum(x*x,0)/D+EPS)
    tl.store(Y+r*D+j,x*inv*w,j<D)

@tr.jit
def rmsnorm_backward(X,W,DY,DX,DW,D:tl.constexpr,EPS:tl.constexpr,BLOCK:tl.constexpr):
    r=tl.program_id(0); j=tl.arange(0,BLOCK)
    x=tl.load(X+r*D+j,j<D,0).to(tl.float32)
    w=tl.load(W+j,j<D,0).to(tl.float32)
    dy=tl.load(DY+r*D+j,j<D,0).to(tl.float32)
    inv=tl.rsqrt(tl.sum(x*x,0)/D+EPS)
    d=dy*w
    dx=inv*(d-x*(tl.sum(d*x,0)/D)*inv*inv)
    tl.store(DX+r*D+j,dx,j<D)
    tl.atomic_add(DW+j,dy*x*inv,j<D,sem="relaxed")

@tr.jit
def swiglu(G,U,Y,n,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    g=tl.load(G+i,i<n,0).to(tl.float32); u=tl.load(U+i,i<n,0).to(tl.float32)
    tl.store(Y+i,g*tl.sigmoid(g)*u,i<n)

@tr.jit
def swiglu_backward(G,U,DY,DG,DU,n,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    g=tl.load(G+i,i<n,0).to(tl.float32); u=tl.load(U+i,i<n,0).to(tl.float32)
    dy=tl.load(DY+i,i<n,0).to(tl.float32); s=tl.sigmoid(g)
    tl.store(DG+i,dy*u*s*(1+g*(1-s)),i<n)
    tl.store(DU+i,dy*g*s,i<n)

@tr.jit
def rope(X,Y,N,T,OFFSET,D:tl.constexpr,H:tl.constexpr,THETA:tl.constexpr,INVERSE:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    pair=i%(D//2); headrow=i//(D//2)
    pos=(headrow//H)%T+OFFSET
    angle=pos*tl.exp((-2.0*pair/D)*tl.log(THETA))
    cos=tl.cos(angle); sin=tl.sin(angle)
    if INVERSE: sin=-sin
    base=headrow*D+2*pair
    a=tl.load(X+base,i<N//2,0).to(tl.float32); b=tl.load(X+base+1,i<N//2,0).to(tl.float32)
    tl.store(Y+base,a*cos-b*sin,i<N//2); tl.store(Y+base+1,a*sin+b*cos,i<N//2)

@tr.jit
def attention(Q,K,V,O,LSE,TQ,TK,OFFSET,H:tl.constexpr,D:tl.constexpr,BD:tl.constexpr,BK:tl.constexpr,CAUSAL:tl.constexpr,FP64:tl.constexpr):
    row=tl.program_id(0); h=row%H; qpos=(row//H)%TQ; batch=row//(H*TQ)
    dtype:tl.constexpr=tl.float64 if FP64 else tl.float32
    d=tl.arange(0,BD); kk=tl.arange(0,BK)
    q=tl.load(Q+row*D+d,d<D,0).to(dtype)
    m=tl.full((),-float('inf'),dtype); l=tl.full((),0,dtype)
    acc=tl.full((BD,),0,dtype)
    for start in range(tl.cdiv(qpos+OFFSET+1 if CAUSAL else TK,BK)):
        kpos=start*BK+kk
        valid=(kpos<TK)&((kpos<=qpos+OFFSET) | (not CAUSAL))
        kp=((batch*TK+kpos[:,None])*H+h)*D+d[None,:]
        k=tl.load(K+kp,valid[:,None]&(d[None,:]<D),0).to(dtype)
        v=tl.load(V+kp,valid[:,None]&(d[None,:]<D),0).to(dtype)
        score=tl.sum(k*q[None,:],1)*(D**-0.5)
        score=tl.where(valid,score,-float('inf'))
        new_m=tl.maximum(m,tl.max(score,0)); alpha=math.exp(m-new_m)
        p=math.exp(score-new_m)
        acc=acc*alpha+tl.sum(v*p[:,None],0)
        l=l*alpha+tl.sum(p,0); m=new_m
    tl.store(O+row*D+d,acc/l,d<D)
    tl.store(LSE+row,m+math.log(l))

@tr.jit
def attention_dq(Q,K,V,O,LSE,DO,DQ,T,H:tl.constexpr,D:tl.constexpr,BD:tl.constexpr,BK:tl.constexpr,CAUSAL:tl.constexpr):
    row=tl.program_id(0); h=row%H; pos=(row//H)%T; b=row//(H*T)
    d=tl.arange(0,BD); kk=tl.arange(0,BK)
    q=tl.load(Q+row*D+d,d<D,0).to(tl.float32)
    do=tl.load(DO+row*D+d,d<D,0).to(tl.float32)
    o=tl.load(O+row*D+d,d<D,0).to(tl.float32)
    delta=tl.sum(do*o,0); lse=tl.load(LSE+row)
    acc=tl.full((BD,),0,tl.float32)
    for start in range(tl.cdiv(pos+1 if CAUSAL else T,BK)):
        kpos=start*BK+kk; valid=(kpos<T)&((kpos<=pos) | (not CAUSAL))
        idx=((b*T+kpos[:,None])*H+h)*D+d[None,:]
        k=tl.load(K+idx,valid[:,None]&(d[None,:]<D),0).to(tl.float32)
        v=tl.load(V+idx,valid[:,None]&(d[None,:]<D),0).to(tl.float32)
        p=tl.exp(tl.sum(k*q[None,:],1)*(D**-0.5)-lse)
        ds=tl.where(valid,p*(tl.sum(v*do[None,:],1)-delta)*(D**-0.5),0)
        acc+=tl.sum(k*ds[:,None],0)
    tl.store(DQ+row*D+d,acc,d<D)

@tr.jit
def attention_dkv(Q,K,V,O,LSE,DO,DK,DV,T,H:tl.constexpr,D:tl.constexpr,BD:tl.constexpr,BK:tl.constexpr,CAUSAL:tl.constexpr):
    row=tl.program_id(0); h=row%H; pos=(row//H)%T; b=row//(H*T)
    d=tl.arange(0,BD); qq=tl.arange(0,BK)
    k=tl.load(K+row*D+d,d<D,0).to(tl.float32); v=tl.load(V+row*D+d,d<D,0).to(tl.float32)
    ak=tl.full((BD,),0,tl.float32); av=tl.full((BD,),0,tl.float32)
    for start in range(pos//BK if CAUSAL else 0,tl.cdiv(T,BK)):
        qpos=start*BK+qq; valid=(qpos<T)&((qpos>=pos) | (not CAUSAL))
        qr=(b*T+qpos)*H+h; idx=qr[:,None]*D+d[None,:]
        q=tl.load(Q+idx,valid[:,None]&(d[None,:]<D),0).to(tl.float32)
        do=tl.load(DO+idx,valid[:,None]&(d[None,:]<D),0).to(tl.float32)
        o=tl.load(O+idx,valid[:,None]&(d[None,:]<D),0).to(tl.float32)
        lse=tl.load(LSE+qr,valid,0)
        p=tl.where(valid,tl.exp(tl.sum(q*k[None,:],1)*(D**-0.5)-lse),0)
        delta=tl.sum(do*o,1)
        ds=p*(tl.sum(do*v[None,:],1)-delta)*(D**-0.5)
        ak+=tl.sum(q*ds[:,None],0); av+=tl.sum(do*p[:,None],0)
    tl.store(DK+row*D+d,ak,d<D); tl.store(DV+row*D+d,av,d<D)

@tr.jit
def cross_entropy(X,Targets,Loss,V:tl.constexpr,BLOCK:tl.constexpr):
    r=tl.program_id(0); j=tl.arange(0,BLOCK)
    x=tl.load(X+r*V+j,j<V,-float('inf')).to(tl.float32)
    target=tl.load(Targets+r); m=tl.max(x,0)
    logsum=m+tl.log(tl.sum(tl.exp(x-m),0))
    chosen=tl.load(X+r*V+target).to(tl.float32)
    tl.store(Loss+r,logsum-chosen)

@tr.jit
def cross_entropy_backward(X,Targets,DY,DX,V:tl.constexpr,BLOCK:tl.constexpr):
    r=tl.program_id(0); j=tl.arange(0,BLOCK)
    x=tl.load(X+r*V+j,j<V,-float('inf')).to(tl.float32)
    e=tl.exp(x-tl.max(x,0)); p=e/tl.sum(e,0)
    target=tl.load(Targets+r); dy=tl.load(DY+r)
    tl.store(DX+r*V+j,(p-(j==target).to(tl.float32))*dy,j<V)

@tr.jit
def adamw(W,G,M,V,N,LR,BETA1,BETA2,EPS,DECAY,INV_B1,INV_B2,SCALE,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK); mask=i<N
    w=tl.load(W+i,mask,0); g=tl.load(G+i,mask,0)*SCALE
    m=BETA1*tl.load(M+i,mask,0)+(1-BETA1)*g
    v=BETA2*tl.load(V+i,mask,0)+(1-BETA2)*g*g
    denominator=tl.sqrt(v*INV_B2)+EPS
    step=tl.where(denominator>0,(m*INV_B1)/denominator,0)
    value=w*(1-LR*DECAY)-LR*step
    tl.store(W+i,value,mask); tl.store(M+i,m,mask); tl.store(V+i,v,mask)

@tr.jit
def sum_squares(X,Y,N,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    x=tl.load(X+i,i<N,0).to(tl.float32)
    tl.store(Y+tl.program_id(0),tl.sum(x*x,0))

@tr.jit
def select_rows(A,B,Mask,Y,N,D,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    selected=tl.load(Mask+i//D,i<N,0)!=0
    a=tl.load(A+i,i<N,0).to(tl.float32); b=tl.load(B+i,i<N,0).to(tl.float32)
    tl.store(Y+i,tl.where(selected,a,b),i<N)

@tr.jit
def matmul64(A,B,C,M,N,K,TA:tl.constexpr,TB:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0); col=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    acc=tl.full((BLOCK,),0,tl.float64)
    for j in range(K):
        ai=j*M+row if TA else row*K+j
        bi=col*K+j if TB else j*N+col
        acc+=tl.load(A+ai).to(tl.float64)*tl.load(B+bi,col<N,0).to(tl.float64)
    tl.store(C+row*N+col,acc,col<N)

@tr.jit
def gelu(X,Y,N,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    dtype:tl.constexpr=tl.float64 if X.dtype.element_ty==tl.float64 else tl.float32
    x=tl.load(X+i,i<N,0).to(dtype)
    tl.store(Y+i,0.5*x*(1+math.erf(x*0.7071067811865476)),i<N)

@tr.jit
def gelu_backward(X,DY,DX,N,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    x=tl.load(X+i,i<N,0).to(tl.float32);dy=tl.load(DY+i,i<N,0)
    derivative=0.5*(1+math.erf(x*0.7071067811865476))+x*tl.exp(-x*x/2)*0.3989422804014327
    tl.store(DX+i,dy*derivative,i<N)

@tr.jit
def gather(X,Ids,Y,N,D,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row=tl.load(Ids+i//D,i<N*D,0)
    tl.store(Y+i,tl.load(X+row*D+i%D,i<N*D,0),i<N*D)

@tr.jit
def gather_backward(DY,Ids,DX,N,D,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row=tl.load(Ids+i//D,i<N*D,0)
    tl.atomic_add(DX+row*D+i%D,tl.load(DY+i,i<N*D,0),i<N*D,sem="relaxed")
