import triton as tr
import triton.language as tl
from triton.language.extra import libdevice as math

@tr.jit
def fill(Y, n, value, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y+i, value, i<n)


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
def hierarchy_attention(Q, K, V, O, LSE, TQ, TK, OFFSET,
                        HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
                        BD: tl.constexpr, BK: tl.constexpr, WINDOW: tl.constexpr,
                        FP64: tl.constexpr):
    row = tl.program_id(0)
    head = row % HQ
    position = (row // HQ) % TQ + OFFSET
    batch = row // (HQ * TQ)
    kv_head = head // (HQ // HK)
    dtype: tl.constexpr = tl.float64 if FP64 else tl.float32
    d = tl.arange(0, BD)
    j = tl.arange(0, BK)
    q = tl.load(Q + row * D + d, d < D, 0).to(dtype)
    maximum = tl.full((), -float('inf'), dtype)
    denominator = tl.full((), 0, dtype)
    result = tl.full((BD,), 0, dtype)
    first = tl.maximum(0, position + 1 - WINDOW) if WINDOW > 0 else 0
    for start in range(first, position + 1, BK):
        key_position = start + j
        valid = (key_position <= position) & (key_position < TK)
        indices = ((batch * TK + key_position[:, None]) * HK + kv_head) * D + d[None, :]
        k = tl.load(K + indices, valid[:, None] & (d[None, :] < D), 0).to(dtype)
        v = tl.load(V + indices, valid[:, None] & (d[None, :] < D), 0).to(dtype)
        scores = tl.where(valid, tl.sum(k * q[None, :], 1) * (D ** -0.5), -float('inf'))
        next_maximum = tl.maximum(maximum, tl.max(scores, 0))
        rescale = math.exp(maximum - next_maximum)
        probability = math.exp(scores - next_maximum)
        result = result * rescale + tl.sum(v * probability[:, None], 0)
        denominator = denominator * rescale + tl.sum(probability, 0)
        maximum = next_maximum
    tl.store(O + row * D + d, result / denominator, d < D)
    tl.store(LSE + row, maximum + math.log(denominator))


@tr.jit
def hierarchy_attention_dq(Q, K, V, O, LSE, DO, DQ, T,
                           HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
                           BD: tl.constexpr, BK: tl.constexpr, WINDOW: tl.constexpr):
    row = tl.program_id(0)
    head = row % HQ
    position = (row // HQ) % T
    batch = row // (HQ * T)
    kv_head = head // (HQ // HK)
    d = tl.arange(0, BD)
    j = tl.arange(0, BK)
    q = tl.load(Q + row * D + d, d < D, 0).to(tl.float32)
    dy = tl.load(DO + row * D + d, d < D, 0).to(tl.float32)
    output = tl.load(O + row * D + d, d < D, 0).to(tl.float32)
    delta = tl.sum(dy * output, 0)
    lse = tl.load(LSE + row)
    result = tl.full((BD,), 0, tl.float32)
    first = tl.maximum(0, position + 1 - WINDOW) if WINDOW > 0 else 0
    for start in range(first, position + 1, BK):
        key_position = start + j
        valid = key_position <= position
        indices = ((batch * T + key_position[:, None]) * HK + kv_head) * D + d[None, :]
        k = tl.load(K + indices, valid[:, None] & (d[None, :] < D), 0).to(tl.float32)
        v = tl.load(V + indices, valid[:, None] & (d[None, :] < D), 0).to(tl.float32)
        probability = tl.exp(tl.sum(k * q[None, :], 1) * (D ** -0.5) - lse)
        ds = tl.where(valid, probability * (tl.sum(v * dy[None, :], 1) - delta) * (D ** -0.5), 0)
        result += tl.sum(k * ds[:, None], 0)
    tl.store(DQ + row * D + d, result, d < D)


@tr.jit
def hierarchy_attention_dkv(Q, K, V, O, LSE, DO, DK, DV, T,
                            HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
                            BD: tl.constexpr, BK: tl.constexpr, WINDOW: tl.constexpr):
    row = tl.program_id(0)
    kv_head = row % HK
    position = (row // HK) % T
    batch = row // (HK * T)
    d = tl.arange(0, BD)
    j = tl.arange(0, BK)
    k = tl.load(K + row * D + d, d < D, 0).to(tl.float32)
    v = tl.load(V + row * D + d, d < D, 0).to(tl.float32)
    dk = tl.full((BD,), 0, tl.float32)
    dv = tl.full((BD,), 0, tl.float32)
    end = tl.minimum(T, position + WINDOW) if WINDOW > 0 else T
    for group in range(HQ // HK):
        head = kv_head * (HQ // HK) + group
        for start in range(position, end, BK):
            query_position = start + j
            valid = query_position < end
            query_row = (batch * T + query_position) * HQ + head
            indices = query_row[:, None] * D + d[None, :]
            q = tl.load(Q + indices, valid[:, None] & (d[None, :] < D), 0).to(tl.float32)
            dy = tl.load(DO + indices, valid[:, None] & (d[None, :] < D), 0).to(tl.float32)
            output = tl.load(O + indices, valid[:, None] & (d[None, :] < D), 0).to(tl.float32)
            lse = tl.load(LSE + query_row, valid, 0)
            probability = tl.where(valid, tl.exp(tl.sum(q * k[None, :], 1) * (D ** -0.5) - lse), 0)
            delta = tl.sum(dy * output, 1)
            ds = probability * (tl.sum(dy * v[None, :], 1) - delta) * (D ** -0.5)
            dk += tl.sum(q * ds[:, None], 0)
            dv += tl.sum(dy * probability[:, None], 0)
    tl.store(DK + row * D + d, dk, d < D)
    tl.store(DV + row * D + d, dv, d < D)

