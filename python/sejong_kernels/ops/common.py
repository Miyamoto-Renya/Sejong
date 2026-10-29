"""Dense affine, normalization and layout operations for native model training."""
import triton as tr
import triton.language as tl
from triton.language.extra import libdevice as math

@tr.jit
def layernorm(X,W,B,Y,D:tl.constexpr,EPS:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);j=tl.arange(0,BLOCK)
    dtype:tl.constexpr=tl.float64 if X.dtype.element_ty==tl.float64 else tl.float32
    x=tl.load(X+row*D+j,j<D,0).to(dtype)
    mean=tl.sum(x,0)/D;center=tl.where(j<D,x-mean,0)
    inv=1/math.sqrt(tl.sum(center*center,0)/D+EPS)
    w=tl.load(W+j,j<D,0).to(dtype);b=tl.load(B+j,j<D,0).to(dtype)
    tl.store(Y+row*D+j,center*inv*w+b,j<D)

@tr.jit
def layernorm_backward(X,W,DY,DX,DW,DB,D:tl.constexpr,EPS:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);j=tl.arange(0,BLOCK)
    x=tl.load(X+row*D+j,j<D,0).to(tl.float32);dy=tl.load(DY+row*D+j,j<D,0).to(tl.float32)
    w=tl.load(W+j,j<D,0).to(tl.float32)
    mean=tl.sum(x,0)/D;c=tl.where(j<D,x-mean,0);inv=tl.rsqrt(tl.sum(c*c,0)/D+EPS);z=c*inv
    h=dy*w
    dx=(h-tl.sum(h,0)/D-z*tl.sum(h*z,0)/D)*inv
    tl.store(DX+row*D+j,dx,j<D)
    tl.atomic_add(DW+j,dy*z,j<D,sem='relaxed');tl.atomic_add(DB+j,dy,j<D,sem='relaxed')

@tr.jit
def bias_add(X,B,Y,N,D,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    tl.store(Y+i,tl.load(X+i,i<N,0)+tl.load(B+i%D,i<N,0),i<N)

@tr.jit
def bias_reduce(X,Y,R:tl.constexpr,D:tl.constexpr,BLOCK:tl.constexpr):
    j=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);acc=tl.full((BLOCK,),0,tl.float32)
    for r in range(R):acc+=tl.load(X+r*D+j,j<D,0).to(tl.float32)
    tl.store(Y+j,acc,j<D)

@tr.jit
def elementwise(X,Z,Y,N,ALPHA:tl.constexpr,MODE:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);x=tl.load(X+i,i<N,0)
    if MODE==0:y=x*ALPHA
    elif MODE==1:y=tl.maximum(x,0)
    elif MODE==2:y=tl.load(Z+i,i<N,0)*(x>0)
    else:y=x*tl.load(Z+i,i<N,0)
    tl.store(Y+i,y,i<N)

@tr.jit
def columns(X,Y,R,SRC_D:tl.constexpr,DST_D:tl.constexpr,START:tl.constexpr,WIDTH:tl.constexpr,INSERT:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);r=i//WIDTH;c=i%WIDTH
    if INSERT:tl.store(Y+r*DST_D+START+c,tl.load(X+r*SRC_D+c,r<R,0),r<R)
    else:tl.store(Y+r*DST_D+c,tl.load(X+r*SRC_D+START+c,r<R,0),r<R)
