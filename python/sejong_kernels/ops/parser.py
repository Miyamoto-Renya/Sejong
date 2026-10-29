"""Relative-position and same-entity attention for the query program encoder."""
import triton as tr
import triton.language as tl

@tr.jit
def parser_attention(Q,K,V,Bias,O,LSE,T,H:tl.constexpr,D:tl.constexpr,BD:tl.constexpr,BK:tl.constexpr):
    row=tl.program_id(0);h=row%H;pos=(row//H)%T;batch=row//(H*T)
    d=tl.arange(0,BD);kk=tl.arange(0,BK)
    q=tl.load(Q+row*D+d,d<D,0).to(tl.float32)
    maximum=tl.full((),-float('inf'),tl.float32);total=tl.full((),0,tl.float32);acc=tl.full((BD,),0,tl.float32)
    for start in range(tl.cdiv(T,BK)):
        key=start*BK+kk;valid=key<T
        index=((batch*T+key[:,None])*H+h)*D+d[None,:]
        k=tl.load(K+index,valid[:,None]&(d[None,:]<D),0).to(tl.float32)
        v=tl.load(V+index,valid[:,None]&(d[None,:]<D),0).to(tl.float32)
        bias=tl.load(Bias+((batch*T+pos)*T+key)*H+h,valid,0)
        scores=tl.where(valid,tl.sum(k*q[None,:],1)*(D**-0.5)+bias,-float('inf'))
        next_max=tl.maximum(maximum,tl.max(scores,0));adjust=tl.exp(maximum-next_max);prob=tl.exp(scores-next_max)
        acc=acc*adjust+tl.sum(v*prob[:,None],0);total=total*adjust+tl.sum(prob,0);maximum=next_max
    tl.store(O+row*D+d,acc/total,d<D);tl.store(LSE+row,maximum+tl.log(total))

@tr.jit
def parser_attention_dq(Q,K,V,Bias,O,LSE,DY,DQ,DB,T,H:tl.constexpr,D:tl.constexpr,BD:tl.constexpr,BK:tl.constexpr):
    row=tl.program_id(0);h=row%H;pos=(row//H)%T;batch=row//(H*T)
    d=tl.arange(0,BD);kk=tl.arange(0,BK)
    q=tl.load(Q+row*D+d,d<D,0).to(tl.float32);dy=tl.load(DY+row*D+d,d<D,0).to(tl.float32);o=tl.load(O+row*D+d,d<D,0).to(tl.float32)
    delta=tl.sum(dy*o,0);logsum=tl.load(LSE+row);acc=tl.full((BD,),0,tl.float32)
    for start in range(tl.cdiv(T,BK)):
        key=start*BK+kk;valid=key<T;index=((batch*T+key[:,None])*H+h)*D+d[None,:]
        k=tl.load(K+index,valid[:,None]&(d[None,:]<D),0).to(tl.float32);v=tl.load(V+index,valid[:,None]&(d[None,:]<D),0).to(tl.float32)
        bindex=((batch*T+pos)*T+key)*H+h;bias=tl.load(Bias+bindex,valid,0)
        prob=tl.where(valid,tl.exp(tl.sum(k*q[None,:],1)*(D**-0.5)+bias-logsum),0)
        ds=prob*(tl.sum(v*dy[None,:],1)-delta)
        acc+=tl.sum(k*ds[:,None],0)*(D**-0.5);tl.store(DB+bindex,ds,valid)
    tl.store(DQ+row*D+d,acc,d<D)

@tr.jit
def parser_attention_dkv(Q,K,V,Bias,O,LSE,DY,DK,DV,T,H:tl.constexpr,D:tl.constexpr,BD:tl.constexpr,BK:tl.constexpr):
    row=tl.program_id(0);h=row%H;pos=(row//H)%T;batch=row//(H*T)
    d=tl.arange(0,BD);qq=tl.arange(0,BK)
    k=tl.load(K+row*D+d,d<D,0).to(tl.float32);v=tl.load(V+row*D+d,d<D,0).to(tl.float32)
    ak=tl.full((BD,),0,tl.float32);av=tl.full((BD,),0,tl.float32)
    for start in range(tl.cdiv(T,BK)):
        query=start*BK+qq;valid=query<T;qr=(batch*T+query)*H+h;index=qr[:,None]*D+d[None,:]
        q=tl.load(Q+index,valid[:,None]&(d[None,:]<D),0).to(tl.float32);dy=tl.load(DY+index,valid[:,None]&(d[None,:]<D),0).to(tl.float32);o=tl.load(O+index,valid[:,None]&(d[None,:]<D),0).to(tl.float32)
        bias=tl.load(Bias+((batch*T+query)*T+pos)*H+h,valid,0);logsum=tl.load(LSE+qr,valid,0)
        prob=tl.where(valid,tl.exp(tl.sum(q*k[None,:],1)*(D**-0.5)+bias-logsum),0);delta=tl.sum(dy*o,1)
        ds=prob*(tl.sum(dy*v[None,:],1)-delta)*(D**-0.5)
        ak+=tl.sum(q*ds[:,None],0);av+=tl.sum(dy*prob[:,None],0)
    tl.store(DK+row*D+d,ak,d<D);tl.store(DV+row*D+d,av,d<D)
