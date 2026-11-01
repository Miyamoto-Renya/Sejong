"""계층 모델의 인과 attention, patch 변환과 expert 경로를 계산한다."""
import triton as tr
import triton.language as tl
from triton.language.extra import libdevice as math


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
def hierarchy_attention_dq(Q, K, V, O, LSE, DO, DQ, Delta, T,
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
    tl.store(Delta + row, delta)
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
def hierarchy_attention_dkv(Q, K, V, LSE, DO, Delta, DK, DV, T,
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
            lse = tl.load(LSE + query_row, valid, 0)
            probability = tl.where(valid, tl.exp(tl.sum(q * k[None, :], 1) * (D ** -0.5) - lse), 0)
            delta = tl.load(Delta + query_row, valid, 0)
            ds = probability * (tl.sum(dy * v[None, :], 1) - delta) * (D ** -0.5)
            dk += tl.sum(q * ds[:, None], 0)
            dv += tl.sum(dy * probability[:, None], 0)
    tl.store(DK + row * D + d, dk, d < D)
    tl.store(DV + row * D + d, dv, d < D)


@tr.jit
def hierarchy_patch_pool(X, Y, T, D, PATCH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    patches = tl.cdiv(T, PATCH)
    batch = row // patches
    start = (row % patches) * PATCH
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    dtype: tl.constexpr = tl.float64 if X.dtype.element_ty == tl.float64 else tl.float32
    total = tl.full((BLOCK,), 0, dtype)
    count = tl.minimum(PATCH, T - start)
    for position in range(count):
        total += tl.load(X + ((batch * T + start + position) * D + columns), columns < D, 0).to(dtype)
    tl.store(Y + row * D + columns, total / count, columns < D)


@tr.jit
def hierarchy_patch_pool_backward(DY, DX, N, T, D, PATCH: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    position = (index // D) % T
    batch = index // (T * D)
    patch = position // PATCH
    count = tl.minimum(PATCH, T - patch * PATCH)
    source = (batch * tl.cdiv(T, PATCH) + patch) * D + index % D
    value = tl.load(DY + source, index < N, 0).to(tl.float32) / count
    tl.store(DX + index, value, index < N)


@tr.jit
def hierarchy_patch_broadcast(X, BOS, Y, N, T, D, PATCH: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    position = (index // D) % T
    batch = index // (T * D)
    patch = position // PATCH - 1
    source = (batch * tl.cdiv(T, PATCH) + tl.maximum(patch, 0)) * D + index % D
    previous = tl.load(X + source, (index < N) & (patch >= 0), 0)
    bos = tl.load(BOS + index % D, index < N, 0)
    tl.store(Y + index, tl.where(patch >= 0, previous, bos), index < N)


@tr.jit
def hierarchy_patch_broadcast_backward(DY, DX, DBOS, N, T, D, PATCH: tl.constexpr, BLOCK: tl.constexpr):
    tile = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    batches = N // (T * D)
    patches = tl.cdiv(T, PATCH)
    elements = batches * patches * D
    body_tiles = tl.cdiv(elements, BLOCK)
    total = tl.full((BLOCK,), 0, tl.float32)
    if tile < body_tiles:
        index = tile * BLOCK + lane
        row = index // D
        batch = row // patches
        start = (row % patches + 1) * PATCH
        for offset in range(PATCH):
            position = start + offset
            value = tl.load(DY + (batch * T + position) * D + index % D,
                            (index < elements) & (position < T), 0).to(tl.float32)
            total += value
        tl.store(DX + index, total, index < elements)
    else:
        column = (tile - body_tiles) * BLOCK + lane
        for batch_index in range(batches):
            for bos_position in range(tl.minimum(PATCH, T)):
                total += tl.load(DY + (batch_index * T + bos_position) * D + column,
                                 column < D, 0).to(tl.float32)
        tl.store(DBOS + column, total, column < D)


@tr.jit
def hierarchy_route_topk(X, IDS, WEIGHTS, E: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    experts = tl.arange(0, BLOCK)
    scores = tl.load(X + row * E + experts, experts < E, -float('inf')).to(tl.float32)
    maximum = tl.max(scores, 0)
    denominator = tl.sum(tl.exp(scores - maximum), 0)
    for slot in range(K):
        best = tl.max(scores, 0)
        index = tl.min(tl.where((scores == best) & (experts < E), experts, E), 0)
        weight = tl.exp(best - maximum)
        tl.store(IDS + row * K + slot, index)
        tl.store(WEIGHTS + row * K + slot, weight)
        scores = tl.where(experts == index, -float('inf'), scores)
    for slot in range(K):
        weight = tl.load(WEIGHTS + row * K + slot)
        tl.store(WEIGHTS + row * K + slot, weight / denominator)


@tr.jit
def hierarchy_route_backward(X, IDS, WEIGHTS, DY, DX, E: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    experts = tl.arange(0, BLOCK)
    logits = tl.load(X + row * E + experts, experts < E, -float('inf')).to(tl.float32)
    unnormalized = tl.exp(logits - tl.max(logits, 0))
    probabilities = unnormalized / tl.sum(unnormalized, 0)
    selected_dy = tl.full((BLOCK,), 0, tl.float32)
    total = tl.full((), 0, tl.float32)
    for slot in range(K):
        expert = tl.load(IDS + row * K + slot)
        weight = tl.load(WEIGHTS + row * K + slot).to(tl.float32)
        dy = tl.load(DY + row * K + slot).to(tl.float32)
        selected_dy += tl.where(experts == expert, dy, 0)
        total += weight * dy
    tl.store(DX + row * E + experts, probabilities * (selected_dy - total), experts < E)


@tr.jit
def hierarchy_route_pack(IDS, TOKENS, SLOTS, COUNTS, R, K: tl.constexpr, BLOCK: tl.constexpr):
    expert = tl.program_id(0)
    offset = tl.arange(0, BLOCK)
    count = tl.full((), 0, tl.int32)
    for start in range(0, R, BLOCK):
        slot = start + offset
        ids = tl.load(IDS + slot, slot < R, -1)
        selected = (slot < R) & (ids == expert)
        rank = tl.cumsum(selected.to(tl.int32), 0) - 1 + count
        tl.store(TOKENS + expert * R + rank, slot // K, selected)
        tl.store(SLOTS + expert * R + rank, slot, selected)
        count += tl.sum(selected.to(tl.int32), 0)
    tl.store(COUNTS + expert, count)


@tr.jit
def hierarchy_expert_combine(X, SLOTS, WEIGHTS, Y, N, D, K: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(SLOTS + index // D, index < N * D, 0)
    weight = tl.load(WEIGHTS + slot, index < N * D, 0).to(tl.float32)
    value = tl.load(X + index, index < N * D, 0).to(tl.float32)
    tl.atomic_add(Y + (slot // K) * D + index % D, weight * value, index < N * D, sem="relaxed")


@tr.jit
def hierarchy_expert_backward(X, SLOTS, WEIGHTS, DY, DX, DW, D,
                              K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    slot = tl.load(SLOTS + row)
    weight = tl.load(WEIGHTS + slot).to(tl.float32)
    offset = tl.arange(0, BLOCK)
    total = tl.full((), 0, tl.float32)
    for start in range(0, D, BLOCK):
        column = start + offset
        value = tl.load(X + row * D + column, column < D, 0).to(tl.float32)
        dy = tl.load(DY + (slot // K) * D + column, column < D, 0).to(tl.float32)
        tl.store(DX + row * D + column, dy * weight, column < D)
        total += tl.sum(value * dy, 0)
    tl.store(DW + slot, total)


@tr.jit
def hierarchy_row_mean(X, Y, ROWS, D, BLOCK: tl.constexpr):
    column = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    dtype: tl.constexpr = tl.float64 if X.dtype.element_ty == tl.float64 else tl.float32
    total = tl.full((BLOCK,), 0, dtype)
    for row in range(ROWS):
        total += tl.load(X + row * D + column, column < D, 0).to(dtype)
    tl.store(Y + column, total / ROWS, column < D)


@tr.jit
def hierarchy_row_mean_backward(DY, DX, N, ROWS, D, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(DY + index % D, index < N, 0).to(tl.float32) / ROWS
    tl.store(DX + index, value, index < N)


@tr.jit
def hierarchy_normalize(X, Y, N, NORM, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + index, index < N, 0).to(tl.float32)
    tl.store(Y + index, value / NORM, index < N)


@tr.jit
def hierarchy_transpose(X, Y, ROWS, COLS, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + index, index < ROWS * COLS, 0)
    tl.store(Y + (index % COLS) * ROWS + index // COLS, value, index < ROWS * COLS)


@tr.jit
def hierarchy_kronecker(A, B, Y, AR, AC, BR, BC, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rows = AR * BR
    columns = AC * BC
    row = index // columns
    column = index % columns
    a = tl.load(A + (row // BR) * AC + column // BC, index < rows * columns, 0).to(tl.float32)
    b = tl.load(B + (row % BR) * BC + column % BC, index < rows * columns, 0).to(tl.float32)
    tl.store(Y + index, a * b, index < rows * columns)


@tr.jit
def hierarchy_coefficient_transport(H, TRANSFORM, Y, R,
                                    LEFT: tl.constexpr, BM: tl.constexpr,
                                    BN: tl.constexpr, BK: tl.constexpr):
    feature = tl.program_id(2)
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    column = tl.program_id(1) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    total = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(R, BK)):
        k = start * BK + inner
        if LEFT:
            a = tl.load(TRANSFORM + k[None, :] * R + row[:, None],
                        (row[:, None] < R) & (k[None, :] < R), 0).to(tl.float32)
            b = tl.load(H + feature * R * R + k[:, None] * R + column[None, :],
                        (k[:, None] < R) & (column[None, :] < R), 0).to(tl.float32)
        else:
            a = tl.load(H + feature * R * R + row[:, None] * R + k[None, :],
                        (row[:, None] < R) & (k[None, :] < R), 0).to(tl.float32)
            b = tl.load(TRANSFORM + k[:, None] * R + column[None, :],
                        (k[:, None] < R) & (column[None, :] < R), 0).to(tl.float32)
        total += tl.dot(a, b, input_precision="ieee")
    tl.store(Y + feature * R * R + row[:, None] * R + column[None, :], total,
             (row[:, None] < R) & (column[None, :] < R))


@tr.jit
def hierarchy_dot(A, B, Y, N, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + index, index < N, 0).to(tl.float32)
    b = tl.load(B + index, index < N, 0).to(tl.float32)
    tl.store(Y + tl.program_id(0), tl.sum(a * b, 0))


@tr.jit
def hierarchy_orthogonal_column(X, Q, ROWS, COLS, COLUMN: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.arange(0, BLOCK)
    for start in range(0, ROWS, BLOCK):
        row = start + offset
        value = tl.load(X + row * COLS + COLUMN, row < ROWS, 0).to(tl.float32)
        tl.store(Q + row * COLS + COLUMN, value, row < ROWS)
    # 두 차례 직교화로 앞선 열 방향의 반올림 잔차를 줄인다.
    for repeat in range(2):
        for previous in range(COLUMN):
            dot = tl.full((), 0, tl.float32)
            for start in range(0, ROWS, BLOCK):
                row = start + offset
                value = tl.load(Q + row * COLS + COLUMN, row < ROWS, 0)
                basis = tl.load(Q + row * COLS + previous, row < ROWS, 0)
                dot += tl.sum(value * basis, 0)
            for start in range(0, ROWS, BLOCK):
                row = start + offset
                value = tl.load(Q + row * COLS + COLUMN, row < ROWS, 0)
                basis = tl.load(Q + row * COLS + previous, row < ROWS, 0)
                tl.store(Q + row * COLS + COLUMN, value - dot * basis, row < ROWS)
    norm = tl.full((), 0, tl.float32)
    for start in range(0, ROWS, BLOCK):
        row = start + offset
        value = tl.load(Q + row * COLS + COLUMN, row < ROWS, 0)
        norm += tl.sum(value * value, 0)
    # 잔차가 작은 열은 기존 열의 점유율이 가장 작은 좌표에서 보완한다.
    if norm < 1.0e-20:
        best_score = tl.full((), -1.0, tl.float32)
        pivot = tl.full((), 0, tl.int32)
        for start in range(0, ROWS, BLOCK):
            row = start + offset
            occupancy = tl.full((BLOCK,), 0, tl.float32)
            for previous in range(COLUMN):
                basis = tl.load(Q + row * COLS + previous, row < ROWS, 0)
                occupancy += basis * basis
            score = tl.where(row < ROWS, 1.0 - occupancy, -1.0)
            local_score = tl.max(score, 0)
            local_pivot = tl.min(tl.where(score == local_score, row, ROWS), 0)
            pivot = tl.where(local_score > best_score, local_pivot, pivot)
            best_score = tl.maximum(best_score, local_score)
        for start in range(0, ROWS, BLOCK):
            row = start + offset
            tl.store(Q + row * COLS + COLUMN, (row == pivot).to(tl.float32), row < ROWS)
        for repeat in range(2):
            for previous in range(COLUMN):
                dot = tl.full((), 0, tl.float32)
                for start in range(0, ROWS, BLOCK):
                    row = start + offset
                    value = tl.load(Q + row * COLS + COLUMN, row < ROWS, 0)
                    basis = tl.load(Q + row * COLS + previous, row < ROWS, 0)
                    dot += tl.sum(value * basis, 0)
                for start in range(0, ROWS, BLOCK):
                    row = start + offset
                    value = tl.load(Q + row * COLS + COLUMN, row < ROWS, 0)
                    basis = tl.load(Q + row * COLS + previous, row < ROWS, 0)
                    tl.store(Q + row * COLS + COLUMN, value - dot * basis, row < ROWS)
        norm = tl.full((), 0, tl.float32)
        for start in range(0, ROWS, BLOCK):
            row = start + offset
            value = tl.load(Q + row * COLS + COLUMN, row < ROWS, 0)
            norm += tl.sum(value * value, 0)
    inverse = tl.rsqrt(norm)
    for start in range(0, ROWS, BLOCK):
        row = start + offset
        value = tl.load(Q + row * COLS + COLUMN, row < ROWS, 0)
        tl.store(Q + row * COLS + COLUMN, value * inverse, row < ROWS)
