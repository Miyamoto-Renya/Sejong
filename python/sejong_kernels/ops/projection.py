"""공유 입력 tile로 여러 projection과 순전파 epilogue를 계산한다."""
import triton as tr
import triton.language as tl


@tr.jit
def projection_qkv(X, WQ, WK, WV, Q, KOUT, VOUT, M, N, K, KV,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    aq = tl.full((BM, BN), 0, tl.float32)
    qdtype: tl.constexpr = tl.float16 if X.dtype.element_ty == tl.float16 and WQ.dtype.element_ty == tl.float16 else tl.float32
    kdtype: tl.constexpr = tl.float16 if X.dtype.element_ty == tl.float16 and WK.dtype.element_ty == tl.float16 else tl.float32
    vdtype: tl.constexpr = tl.float16 if X.dtype.element_ty == tl.float16 and WV.dtype.element_ty == tl.float16 else tl.float32
    if tl.program_id(1) * BN < KV:
        ak = tl.full((BM, BN), 0, tl.float32)
        av = tl.full((BM, BN), 0, tl.float32)
        for start in range(tl.cdiv(K, BK)):
            kk = start * BK + k
            x = tl.load(X + m[:, None] * K + kk[None, :],
                        (m[:, None] < M) & (kk[None, :] < K), 0)
            wq = tl.load(WQ + kk[:, None] * N + n[None, :],
                         (kk[:, None] < K) & (n[None, :] < N), 0)
            wk = tl.load(WK + kk[:, None] * KV + n[None, :],
                         (kk[:, None] < K) & (n[None, :] < KV), 0)
            wv = tl.load(WV + kk[:, None] * KV + n[None, :],
                         (kk[:, None] < K) & (n[None, :] < KV), 0)
            aq += tl.dot(x.to(qdtype), wq.to(qdtype), input_precision="ieee")
            ak += tl.dot(x.to(kdtype), wk.to(kdtype), input_precision="ieee")
            av += tl.dot(x.to(vdtype), wv.to(vdtype), input_precision="ieee")
        mask = (m[:, None] < M) & (n[None, :] < KV)
        tl.store(KOUT + m[:, None] * KV + n[None, :], ak, mask)
        tl.store(VOUT + m[:, None] * KV + n[None, :], av, mask)
    else:
        for start in range(tl.cdiv(K, BK)):
            kk = start * BK + k
            x = tl.load(X + m[:, None] * K + kk[None, :],
                        (m[:, None] < M) & (kk[None, :] < K), 0)
            wq = tl.load(WQ + kk[:, None] * N + n[None, :],
                         (kk[:, None] < K) & (n[None, :] < N), 0)
            aq += tl.dot(x.to(qdtype), wq.to(qdtype), input_precision="ieee")
    tl.store(Q + m[:, None] * N + n[None, :], aq,
             (m[:, None] < M) & (n[None, :] < N))


@tr.jit
def projection_gate_up(X, WG, WU, G, U, H, M, N, K,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    ag = tl.full((BM, BN), 0, tl.float32)
    au = tl.full((BM, BN), 0, tl.float32)
    gdtype: tl.constexpr = tl.float16 if X.dtype.element_ty == tl.float16 and WG.dtype.element_ty == tl.float16 else tl.float32
    udtype: tl.constexpr = tl.float16 if X.dtype.element_ty == tl.float16 and WU.dtype.element_ty == tl.float16 else tl.float32
    for start in range(tl.cdiv(K, BK)):
        kk = start * BK + k
        x = tl.load(X + m[:, None] * K + kk[None, :],
                    (m[:, None] < M) & (kk[None, :] < K), 0)
        wg = tl.load(WG + kk[:, None] * N + n[None, :],
                     (kk[:, None] < K) & (n[None, :] < N), 0)
        wu = tl.load(WU + kk[:, None] * N + n[None, :],
                     (kk[:, None] < K) & (n[None, :] < N), 0)
        ag += tl.dot(x.to(gdtype), wg.to(gdtype), input_precision="ieee")
        au += tl.dot(x.to(udtype), wu.to(udtype), input_precision="ieee")
    g = ag.to(G.dtype.element_ty).to(tl.float32)
    u = au.to(U.dtype.element_ty).to(tl.float32)
    offset = m[:, None] * N + n[None, :]
    mask = (m[:, None] < M) & (n[None, :] < N)
    tl.store(G + offset, g, mask)
    tl.store(U + offset, u, mask)
    tl.store(H + offset, g * tl.sigmoid(g) * u, mask)
