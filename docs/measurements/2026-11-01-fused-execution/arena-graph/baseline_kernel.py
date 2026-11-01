import triton as tr
import triton.language as tl
from triton.language.extra import libdevice as math

@tr.jit
def add(X, Z, Y, n, alpha, BLOCK: tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    x=tl.load(X+i,i<n,0)
    z=tl.load(Z+i,i<n,0)
    tl.store(Y+i,x+alpha*z,i<n)

