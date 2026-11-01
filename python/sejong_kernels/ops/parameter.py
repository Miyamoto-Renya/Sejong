"""순전파 기저, 선택 잔차와 인수 gradient를 소비하는 SGD 커널."""
import triton as tr
import triton.language as tl


@tr.jit
def _random(state):
    state = state + tl.full((), 0x9e3779b97f4a7c15, tl.uint64)
    value = state
    value = (value ^ (value >> 30)) * tl.full((), 0xbf58476d1ce4e5b9, tl.uint64)
    value = (value ^ (value >> 27)) * tl.full((), 0x94d049bb133111eb, tl.uint64)
    return state, value ^ (value >> 31)


@tr.jit
def _draw(state, size):
    size = size.to(tl.uint64)
    threshold = (tl.full((), 0, tl.uint64) - size) % size
    state, value = _random(state)
    while value < threshold:
        state, value = _random(state)
    return state, (value % size).to(tl.int32)


@tr.jit
def parameter_indices(Scratch, Columns, Rows, Map, A, B, NB, S, R,
                      SEED_LO, SEED_HI, GROUP: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    for start in range(0, B, BLOCK):
        i = start + lane
        tl.store(Scratch + i, i, i < B)
    for start in range(0, NB, BLOCK):
        i = start + lane
        tl.store(Scratch + B + i, i, i < NB)
    for start in range(0, A, BLOCK):
        i = start + lane
        tl.store(Map + i, -1, i < A)
    tl.debug_barrier()
    state = SEED_LO.to(tl.uint32).to(tl.uint64) | (SEED_HI.to(tl.uint32).to(tl.uint64) << 32)
    for i in range(R):
        state, draw = _draw(state, B-i)
        j = i+draw
        old = tl.load(Scratch+i, volatile=True)
        value = tl.load(Scratch+j, volatile=True)
        tl.store(Scratch+i, value)
        tl.store(Scratch+j, old)
        tl.store(Columns+i, value)
        tl.debug_barrier()
    for i in range(S):
        state, draw = _draw(state, NB-i)
        j = i+draw
        old = tl.load(Scratch+B+i, volatile=True)
        value = tl.load(Scratch+B+j, volatile=True)
        tl.store(Scratch+B+i, value)
        tl.store(Scratch+B+j, old)
        for start in range(0, GROUP, BLOCK):
            offset = start+lane
            index = value*GROUP+offset
            position = i*GROUP+offset
            tl.store(Rows+position, tl.where(index<A,index,-1), offset<GROUP)
            tl.store(Map+index, position, (offset<GROUP)&(index<A))
        tl.debug_barrier()


@tr.jit
def parameter_basis_register(Y,Columns,Q,N,B,EPS,R:tl.constexpr,
                             NR:tl.constexpr,NC:tl.constexpr):
    row=tl.arange(0,NR)
    column=tl.arange(0,NC)
    selected=tl.load(Columns+column,column<R,0)
    basis=tl.load(Y+row[:,None]*B+selected[None,:],
                  (row[:,None]<N)&(column[None,:]<R),0).to(tl.float32)
    for current in range(R):
        vector=tl.sum(tl.where(column[None,:]==current,basis,0.),1)
        original=tl.sum(vector*vector,0)
        for repeat in tl.static_range(2):
            coefficient=tl.sum(tl.where(column[None,:]<current,basis*vector[:,None],0.),0)
            vector-=tl.sum(tl.where(column[None,:]<current,basis*coefficient[None,:],0.),1)
        magnitude=tl.sum(vector*vector,0)
        inverse=tl.where(magnitude>EPS*EPS*original,tl.rsqrt(magnitude),0.)
        basis=tl.where(column[None,:]==current,(vector*inverse)[:,None],basis)
    tl.store(Q+row[:,None]*R+column[None,:],basis,(row[:,None]<N)&(column[None,:]<R))


@tr.jit
def parameter_basis(Y, Columns, Q, N, B,EPS, R: tl.constexpr,
                    BR: tl.constexpr, BC: tl.constexpr):
    row = tl.arange(0, BR)
    col = tl.arange(0, BC)
    # Streaming CGS2는 N과 rank를 루프로 처리하며 register tile의 크기를 유지한다.
    for column in range(R):
        source = tl.load(Columns+column)
        original = tl.full((), 0, tl.float32)
        for start in range(0, N, BR):
            i = start+row
            x = tl.load(Y+i*B+source, i<N, 0).to(tl.float32)
            original += tl.sum(x*x,0)
            tl.store(Q+i*R+column, x, i<N)
        tl.debug_barrier()
        for repeat in tl.static_range(2):
            for first in range(0, column, BC):
                previous = first+col
                coefficients = tl.full((BC,),0,tl.float32)
                for start in range(0,N,BR):
                    i=start+row
                    x=tl.load(Q+i*R+column,i<N,0,volatile=True)
                    q=tl.load(Q+i[:,None]*R+previous[None,:],
                              (i[:,None]<N)&(previous[None,:]<column),0,volatile=True)
                    coefficients+=tl.sum(q*x[:,None],0)
                for start in range(0,N,BR):
                    i=start+row
                    x=tl.load(Q+i*R+column,i<N,0,volatile=True)
                    q=tl.load(Q+i[:,None]*R+previous[None,:],
                              (i[:,None]<N)&(previous[None,:]<column),0,volatile=True)
                    x-=tl.sum(q*coefficients[None,:],1)
                    tl.store(Q+i*R+column,x,i<N)
                tl.debug_barrier()
        magnitude=tl.full((),0,tl.float32)
        for start in range(0,N,BR):
            i=start+row
            x=tl.load(Q+i*R+column,i<N,0,volatile=True)
            magnitude+=tl.sum(x*x,0)
        inverse=tl.where(magnitude>EPS*EPS*original,tl.rsqrt(magnitude),0.)
        for start in range(0,N,BR):
            i=start+row
            x=tl.load(Q+i*R+column,i<N,0,volatile=True)
            tl.store(Q+i*R+column,x*inverse,i<N)
        tl.debug_barrier()


@tr.jit
def parameter_prepare(X,Q,A,Map,E,M,N,K,R,INVERSE,
                      BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    tile=tl.program_id(0)
    m=tile*BM+tl.arange(0,BM)
    rank=tl.arange(0,BN)
    reduction=tl.arange(0,BK)
    left=tl.full((BM,BN),0,tl.float32)
    for start in range(tl.cdiv(N,BK)):
        token=start*BK+reduction
        x=tl.load(X+token[None,:]*M+m[:,None],
                  (token[None,:]<N)&(m[:,None]<M),0).to(tl.float32)
        q=tl.load(Q+token[:,None]*R+rank[None,:],
                  (token[:,None]<N)&(rank[None,:]<R),0)
        left+=tl.dot(x,q,input_precision="ieee")
    tl.store(A+m[:,None]*R+rank[None,:],left,(m[:,None]<M)&(rank[None,:]<R))
    selected=tl.load(Map+m,m<M,-1)
    if tl.sum((selected>=0).to(tl.int32),0)>0:
        for start in range(tl.cdiv(N,BN)):
            token=start*BN+tl.arange(0,BN)
            q=tl.load(Q+token[None,:]*R+rank[:,None],
                      (token[None,:]<N)&(rank[:,None]<R),0)
            projected=tl.dot(left,q,input_precision="ieee")
            x=tl.load(X+token[None,:]*M+m[:,None],
                      (token[None,:]<N)&(m[:,None]<M),0).to(tl.float32)
            tl.store(E+(R+selected[:,None])*N+token[None,:],(x-projected)*INVERSE,
                     (selected[:,None]>=0)&(token[None,:]<N))
    if tile==0:
        for start in range(tl.cdiv(N,BN)):
            token=start*BN+tl.arange(0,BN)
            q=tl.load(Q+token[None,:]*R+rank[:,None],
                      (token[None,:]<N)&(rank[:,None]<R),0)
            tl.store(E+rank[:,None]*N+token[None,:],q,
                     (rank[:,None]<R)&(token[None,:]<N))


@tr.jit
def parameter_residual(X,Q,A,Rows,E,M,N,K,R,INVERSE,
                       BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    tile=tl.program_id(0)
    head_tiles=tl.cdiv(R,BM)
    m=(tile-head_tiles)*BM+tl.arange(0,BM)
    n=tl.program_id(1)*BN+tl.arange(0,BN)
    k=tl.arange(0,BK)
    if tile<head_tiles:
        rank=tile*BM+tl.arange(0,BM)
        q=tl.load(Q+n[None,:]*R+rank[:,None],(rank[:,None]<R)&(n[None,:]<N),0)
        tl.store(E+rank[:,None]*N+n[None,:],q,(rank[:,None]<R)&(n[None,:]<N))
    else:
        original=tl.load(Rows+m,m<K,-1)
        acc=tl.full((BM,BN),0,tl.float32)
        for start in range(tl.cdiv(R,BK)):
            rank=start*BK+k
            a=tl.load(A+original[:,None]*R+rank[None,:],
                      (m[:,None]<K)&(original[:,None]>=0)&(rank[None,:]<R),0)
            q=tl.load(Q+n[None,:]*R+rank[:,None],(n[None,:]<N)&(rank[:,None]<R),0)
            acc+=tl.dot(a,q,input_precision="ieee")
        x=tl.load(X+n[None,:]*M+original[:,None],
                  (m[:,None]<K)&(original[:,None]>=0)&(n[None,:]<N),0).to(tl.float32)
        tl.store(E+(R+m[:,None])*N+n[None,:],(x-acc)*INVERSE,(m[:,None]<K)&(n[None,:]<N))


@tr.jit
def _pointer(metadata, offset, INTEGER:tl.constexpr):
    lo=tl.load(metadata+offset).to(tl.uint32).to(tl.uint64)
    hi=tl.load(metadata+offset+1).to(tl.uint32).to(tl.uint64)
    address=lo|(hi<<32)
    if INTEGER:
        return address.to(tl.pointer_type(tl.int32))
    else:
        return address.to(tl.pointer_type(tl.float32))


@tr.jit
def _gradient(Meta,Parameter,M,N,mm,nn,RANK_OFFSET,TERM_OFFSET:tl.constexpr,
              BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    header=Parameter*6
    dense=_pointer(Meta,header,False)
    has_dense=(tl.load(Meta+header)!=0)|(tl.load(Meta+header+1)!=0)
    value=tl.load(dense+mm[:,None]*N+nn[None,:],
                  (mm[:,None]<M)&(nn[None,:]<N)&has_dense,0)
    begin=tl.load(Meta+header+2)
    count=tl.load(Meta+header+3)
    rank_begin=tl.load(Meta+header+4)
    total_rank=tl.load(Meta+header+5)
    k=tl.arange(0,BK)
    head=tl.full((BM,BN),0,tl.float32)
    for start in range(tl.cdiv(total_rank,BK)):
        kk=start*BK+k
        term=tl.load(Meta+RANK_OFFSET+rank_begin+kk,kk<total_rank,begin)
        offset=TERM_OFFSET+term*11
        left=_pointer(Meta,offset,False)
        right=_pointer(Meta,offset+2,False)
        rank=tl.load(Meta+offset+8)
        scale=tl.load(Meta+offset+9).to(tl.float32,bitcast=True)
        local_k=kk-tl.load(Meta+offset+10)
        a=tl.load(left[None,:]+mm[:,None]*rank[None,:]+local_k[None,:],
                  (mm[:,None]<M)&(kk[None,:]<total_rank),0)*scale[None,:]
        b=tl.load(right[:,None]+local_k[:,None]*N+nn[None,:],
                  (kk[:,None]<total_rank)&(nn[None,:]<N),0)
        head+=tl.dot(a,b,input_precision="ieee")
    value+=head
    for index in range(count):
        offset=TERM_OFFSET+(begin+index)*11
        tail=_pointer(Meta,offset+4,False)
        row_map=_pointer(Meta,offset+6,True)
        scale=tl.load(Meta+offset+9).to(tl.float32,bitcast=True)
        selected=tl.load(row_map+mm,mm<M,-1)
        correction=tl.load(tail+selected[:,None]*N+nn[None,:],
                           (mm[:,None]<M)&(selected[:,None]>=0)&(nn[None,:]<N),0)
        value+=correction*scale
    return value


@tr.jit
def _parameter_tile(Plan,Owners,Meta,RANK_OFFSET,TERM_OFFSET:tl.constexpr,
                    BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    tile=tl.program_id(0)
    parameter=tl.load(Owners+tile)
    offset=parameter*6
    weight=_pointer(Plan,offset,False)
    m=tl.load(Plan+offset+2);n=tl.load(Plan+offset+3)
    first=tl.load(Plan+offset+4)
    decay=tl.load(Plan+offset+5)
    column_tiles=tl.cdiv(n,BN)
    local=tile-first
    mm=(local//column_tiles)*BM+tl.arange(0,BM)
    nn=(local%column_tiles)*BN+tl.arange(0,BN)
    g=_gradient(Meta,parameter,m,n,mm,nn,RANK_OFFSET,TERM_OFFSET,BM,BN,BK)
    valid=(mm[:,None]<m)&(nn[None,:]<n)
    return weight,n,mm,nn,g,valid,decay


@tr.jit
def parameter_gradient_update(Plan,Owners,Meta,RANK_OFFSET,LR,DECAY,
                              TERM_OFFSET:tl.constexpr,
                              BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    weight,n,mm,nn,g,valid,decay=_parameter_tile(
        Plan,Owners,Meta,RANK_OFFSET,TERM_OFFSET,BM,BN,BK)
    w=tl.load(weight+mm[:,None]*n+nn[None,:],valid,0)
    tl.store(weight+mm[:,None]*n+nn[None,:],
             w*tl.where(decay!=0,1.-DECAY,1.)-LR*g,valid)


@tr.jit
def parameter_gradient_materialize(Plan,Owners,Meta,RANK_OFFSET,
                                    TERM_OFFSET:tl.constexpr,
                                    BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    weight,n,mm,nn,g,valid,decay=_parameter_tile(
        Plan,Owners,Meta,RANK_OFFSET,TERM_OFFSET,BM,BN,BK)
    tl.store(weight+mm[:,None]*n+nn[None,:],g,valid)


@tr.jit
def parameter_gradient_norm(Plan,Owners,Meta,RANK_OFFSET,Output,
                            TERM_OFFSET:tl.constexpr,
                            BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    weight,n,mm,nn,g,valid,decay=_parameter_tile(
        Plan,Owners,Meta,RANK_OFFSET,TERM_OFFSET,BM,BN,BK)
    g=tl.where(valid,g,0.).to(tl.float64)
    tl.store(Output+tl.program_id(0),tl.sum(tl.sum(g*g,1),0))
