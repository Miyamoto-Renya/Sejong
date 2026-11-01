#!/usr/bin/env python3
"""Sejong의 주요 부동소수 연산을 집계하고 계산 재배치의 항등식을 검증한다."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
from pathlib import Path
import platform
import tomllib

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Shape:
    width: int
    layers: int
    ffn: int
    heads: int
    vocabulary: int = 257

    def matrices(self, head: bool = True) -> list[tuple[int, int]]:
        d, f = self.width, self.ffn
        block = [(d, d)] * 4 + [(d, f)] * 2 + [(f, d)]
        return block * self.layers + ([(d, self.vocabulary)] if head else [])

    def parameters(self) -> int:
        return sum(a * b for a, b in self.matrices()) + self.vocabulary * self.width + (2 * self.layers + 1) * self.width


def mm(m: int, n: int, k: int, padded: bool = False) -> int:
    if padded:
        m, n, k = ((m + 15) // 16 * 16, (n + 31) // 32 * 32, (k + 31) // 32 * 32)
    return 2 * m * n * k


def attention(shape: Shape, length: int, padded: bool = False, window: int | None = None) -> tuple[int, int]:
    """현재 attention 커널의 내적·벡터 가중합을 FMA=2로 집계한다."""
    d = shape.width
    if window is not None:
        pairs = sum(min(t + 1, window) for t in range(length))
        return 4 * d * pairs * shape.layers, (16 * d * pairs + 2 * length * d) * shape.layers
    if padded:
        hdim = d // shape.heads
        d = shape.heads * (1 << (hdim - 1).bit_length())
        q_pairs = sum((t + 32) // 32 * 32 for t in range(length))
        kv_pairs = sum(((length + 31) // 32 - t // 32) * 32 for t in range(length))
        return 4 * d * q_pairs * shape.layers, (6 * d * q_pairs + 10 * d * kv_pairs + 2 * length * d) * shape.layers
    pairs = length * (length + 1) // 2
    return 4 * d * pairs * shape.layers, (16 * d * pairs + 2 * length * d) * shape.layers


def ledger(shape: Shape, length: int, checkpoint: bool, padded: bool = False,
           head: bool = True, window: int | None = None) -> dict:
    forward = sum(mm(length, b, a, padded) for a, b in shape.matrices(head))
    backward = sum(mm(length, a, b, padded) + mm(a, b, length, padded) for a, b in shape.matrices(head))
    recompute = sum(mm(length, b, a, padded) for a, b in shape.matrices(False)) if checkpoint else 0
    af, ab = attention(shape, length, padded, window)
    ar = af if checkpoint else 0
    total = forward + backward + recompute + af + ab + ar
    return {
        "shape": asdict(shape), "parameters": shape.parameters(), "positions": length,
        "checkpoint": checkpoint, "tile_padding_counted": padded,
        "matrix_forward": forward, "matrix_backward": backward, "matrix_recompute": recompute,
        "attention_forward_dots": af, "attention_backward_dots": ab,
        "attention_recompute_dots": ar, "counted_flops": total,
        "counted_flops_per_position": total / length,
    }


def factorized_local(shape: Shape, length: int, count: int, controls: int, rank: int) -> dict:
    """고정한 예측기 C=HD를 누적 배치 전체에서 인수 형태로 계산한다."""
    body = shape.matrices(False)
    forward = count * sum(mm(length, b, a) for a, b in body)
    exact_backward = controls * sum(mm(length, a, b) + mm(a, b, length) for a, b in body)
    feature = count * sum(mm(length, rank, a + b + 2 * shape.vocabulary) for a, b in body)
    contraction = count * sum(mm(a, rank, length) for a, _ in body)
    lift = sum(mm(a, b, rank) for a, b in body)
    predictor_training = controls * sum(
        3 * mm(length, b, rank) + mm(length, rank, a + b + 2 * shape.vocabulary)
        for a, b in body
    )
    head = count * 3 * mm(length, shape.vocabulary, shape.width)
    af, ab = attention(shape, length)
    parts = {
        "main_matrix_forward": forward, "main_exact_matrix_backward": exact_backward,
        "predictor_hidden_features": feature, "factor_contraction": contraction,
        "factor_lift_once_per_step": lift, "predictor_training": predictor_training,
        "exact_output_head": head, "attention_dots": count * af + controls * ab,
        "sgd_update": 2 * shape.parameters(),
    }
    reference = count * ledger(shape, length, False)["counted_flops"] + 2 * shape.parameters()
    total = sum(parts.values())
    return {
        "microbatches": count, "controls": controls, "rank": rank,
        "components": parts, "counted_flops": total,
        "reference_counted_flops": reference, "flop_ratio": total / reference,
        "reference_over_candidate": reference / total,
        "scope": "행렬곱·attention 내적·본체 SGD; 작은 norm/embedding 예측과 점별 연산은 별도 항",
    }


def proxy_cost(shape: Shape, length: int, count: int, controls: int, rank: int,
               refresh_every: int, checkpoint: bool = False) -> dict:
    proxy = Shape(512, 4, 1536, 8)
    feature_width = 64
    matrices = shape.matrices()
    p_linear = sum(a * b for a, b in matrices)
    coefficient_weights = len(matrices) * feature_width * rank**2
    feature_weights = 2 * proxy.width * feature_width
    basis_values = sum((a + b) * rank for a, b in matrices)
    main_one = ledger(shape, length, checkpoint)["counted_flops"]
    proxy_one = ledger(proxy, length, False)["counted_flops"]
    coefficient_forward = 2 * (coefficient_weights + feature_weights)
    projection = sum(mm(length, rank, a) + mm(length, rank, b) + mm(rank, rank, length) for a, b in matrices)
    lift = sum(2 * a * b * rank + 2 * min(a, b) * rank**2 for a, b in matrices)
    # 두 제어 기울기의 차이를 양쪽에서 스케치하고 QR·계수 좌표 이동을 수행한다.
    refresh = sum(8 * a * b * rank + 4 * (a + b) * rank**2 + 4 * feature_width * rank**3 for a, b in matrices)
    parts = {
        "main_full_forward_backward": controls * main_one,
        "proxy_forward_backward_all_microbatches": count * proxy_one,
        "coefficient_forward": count * coefficient_forward,
        "coefficient_backward_controls": controls * 2 * coefficient_forward,
        "true_gradient_projection_controls": controls * projection,
        "factor_lift_once_per_step": lift,
        "basis_refresh_amortized": refresh / refresh_every,
        "two_control_gradient_diagnostics": 3 * shape.parameters() + 8 * len(matrices) * rank**2,
        "main_sgd": 2 * shape.parameters(),
        "proxy_and_coefficient_sgd": 2 * (proxy.parameters() + coefficient_weights + feature_weights),
    }
    total = sum(parts.values())
    reference = count * main_one + 2 * shape.parameters()
    gamma = total / reference
    p = controls / count
    q_limit = (1 / gamma - 1) / (1 / p - 1)
    return {
        "microbatches": count, "controls": controls, "rank": rank,
        "refresh_every_steps": refresh_every, "main_checkpoint": checkpoint,
        "proxy_parameters": proxy.parameters(), "coefficient_parameters": coefficient_weights + feature_weights,
        "basis_entries": basis_values, "linear_parameters": p_linear,
        "components": parts, "one_basis_refresh_flops": refresh,
        "counted_flops": total, "reference_counted_flops": reference,
        "flop_ratio": gamma, "reference_over_candidate": 1 / gamma,
        "residual_variance_ratio_break_even": q_limit,
        "variance_cost_score": {
            str(q): gamma * (1 + (1 / p - 1) * q) for q in (0, 0.1, 0.25, 0.5, 0.9, 1)
        },
    }


def multirate(length: int, patch: int, checkpoint: bool) -> dict:
    global_model = Shape(2560, 32, 12800, 40)
    encoder = Shape(256, 2, 768, 4)
    decoder = Shape(512, 2, 1536, 8)
    chunks = (length + patch - 1) // patch
    global_count = ledger(global_model, chunks, checkpoint, head=False)["counted_flops"]
    encoder_count = ledger(encoder, length, False, head=False, window=128)["counted_flops"]
    decoder_count = ledger(decoder, length, False, window=128)["counted_flops"]
    bridge = 3 * (mm(chunks, global_model.width, encoder.width)
                  + mm(chunks, decoder.width, global_model.width)
                  + mm(length, decoder.width, encoder.width))
    # pooling·broadcast의 순전파와 역전파는 위치 수×폭에 비례하는 항으로 더한다.
    pooling_broadcast = 2 * length * (encoder.width + decoder.width)
    p_global = sum(a * b for a, b in global_model.matrices(False)) + (2 * global_model.layers + 2) * global_model.width
    p_encoder = encoder.parameters() - encoder.width * encoder.vocabulary
    p_decoder = decoder.parameters() - decoder.width * decoder.vocabulary
    p_bridge = encoder.width * global_model.width + global_model.width * decoder.width + encoder.width * decoder.width
    total = global_count + encoder_count + decoder_count + bridge + pooling_broadcast
    return {
        "bytes": length, "patch_bytes": patch, "global_positions": chunks,
        "global_checkpoint": checkpoint,
        "parameters": p_global + p_encoder + p_decoder + p_bridge,
        "components": {"global": global_count, "encoder": encoder_count,
                       "decoder": decoder_count, "bridges": bridge, "pooling_and_broadcast": pooling_broadcast},
        "counted_flops": total, "counted_flops_per_byte": total / length,
    }


def checked(name: str, actual, expected) -> float:
    a, e = np.asarray(actual), np.asarray(expected)
    error = float(np.max(np.abs(a - e)))
    if not np.allclose(a, e, atol=1e-10, rtol=1e-10):
        raise ArithmeticError(f"{name}: {error}")
    return error


def algebra() -> dict:
    rng = np.random.default_rng(20261101)
    count, controls, length, ni, no, rank = 8, 2, 17, 13, 11, 4
    xs = rng.normal(size=(count, length, ni))
    hs = rng.normal(size=(count, length, rank))
    right = rng.normal(size=(rank, no))
    deltas = rng.normal(size=(count, length, no))
    truth = np.stack([x.T @ delta for x, delta in zip(xs, deltas)])
    predictions = np.stack([x.T @ (h @ right) for x, h in zip(xs, hs)])
    errors, estimators = [], []
    for control in itertools.combinations(range(count), controls):
        weights = np.full(count, 1 / count)
        weights[list(control)] -= 1 / controls
        contracted = sum(w * (x.T @ h) for w, x, h in zip(weights, xs, hs))
        factored = contracted @ right + truth[list(control)].mean(axis=0)
        direct = predictions.mean(axis=0) + (truth - predictions)[list(control)].mean(axis=0)
        errors.append(checked("누적 저랭크 결합", factored, direct))
        estimators.append(factored)
    left, _ = np.linalg.qr(rng.normal(size=(ni, rank)))
    right_basis, _ = np.linalg.qr(rng.normal(size=(no, rank)))
    projected = np.stack([left.T @ g @ right_basis for g in truth])
    projection_error = checked("기울기 계수 직접 계산", projected[0], (xs[0] @ left).T @ (deltas[0] @ right_basis))
    coeff_prediction = projected + rng.normal(scale=0.3, size=projected.shape)
    proxy_predictions = np.stack([left @ c @ right_basis.T for c in coeff_prediction])
    estimates = np.stack([
        proxy_predictions.mean(axis=0) + (truth - proxy_predictions)[list(c)].mean(axis=0)
        for c in itertools.combinations(range(count), controls)
    ])
    center = lambda x: x - x.mean(axis=0)
    orthogonal = truth - np.stack([left @ c @ right_basis.T for c in projected])
    residual = truth - proxy_predictions
    decomposition_error = checked(
        "중심화 잔차의 직교 분해", np.square(center(residual)).sum(),
        np.square(center(orthogonal)).sum() + np.square(center(projected - coeff_prediction)).sum(),
    )
    causal_cases = 0
    for patch in (1, 4, 8, 16):
        for target in range(67):
            latest_global_source = (target // patch) * patch - 1
            if max(latest_global_source, target - 1) >= target:
                raise ArithmeticError("patch 시프트의 인과 순서 오류")
            causal_cases += 1
    return {
        "scope": "고정 행렬 항등식·제어 부분집합 전수 열거·patch 입력 의존 순서",
        "control_subsets": len(estimators), "factorization_max_abs_error": max(errors),
        "factorized_estimator_mean_error": checked("결합 기울기 평균", np.mean(estimators, axis=0), truth.mean(axis=0)),
        "projected_target_max_abs_error": projection_error,
        "proxy_estimator_mean_error": checked("보조 모델 기울기 평균", estimates.mean(axis=0), truth.mean(axis=0)),
        "centered_variance_decomposition_abs_error": decomposition_error,
        "causal_dependency_cases": causal_cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = tomllib.loads((ROOT / "configs/sejong.toml").read_text())
    c = config["model"]
    current = Shape(c["width"], c["layers"], c["ffn_width"], c["heads"])
    large = Shape(2560, 32, 12800, 40)
    bpe = Shape(2560, 32, 12160, 40, 32768)
    length = config["training"]["sequence"]
    bpe_bytes_per_token = 4
    baseline = ledger(large, length, False)
    bpe_reference = ledger(bpe, length // bpe_bytes_per_token, False)
    report = {
        "research_date": "2026-11-01", "python": platform.python_version(), "numpy": np.__version__,
        "convention": {
            "fma_flops": 2,
            "scope": "행렬곱·attention 내적 및 명시한 SGD·변환 비용; exp/sigmoid/norm/형변환·통신은 별도 항",
            "scalar_dimension_sums": "내적·가중합은 2d 관례",
            "gpu_timing": "후속 커널 실행에서 측정",
        },
        "source_sha256": {
            str(p): hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
            for p in ("scripts/research-flops.py", "configs/sejong.toml", "python/sejong_kernels/ops/kernels.py")
        },
        "current_default": ledger(current, length, True),
        "current_default_tiled": ledger(current, length, True, True),
        "dense_4b": baseline,
        "dense_4b_checkpointed": ledger(large, length, True),
        "dense_4b_tiled": ledger(large, length, False, True),
        "dense_4b_context2048": ledger(large, 2048, False),
        "factorized_local": [factorized_local(large, length, 8, m, 32) for m in (2, 4)],
        "proxy_correction": [proxy_cost(large, length, n, m, r, 32)
                             for n, m, r in ((8, 2, 16), (8, 2, 32), (8, 2, 64), (16, 2, 32), (8, 4, 32))],
        "proxy_correction_checkpointed": proxy_cost(large, length, 8, 2, 32, 32, True),
        "multirate": [multirate(length, p, False) for p in (4, 8, 16)],
        "bpe_comparison_assumption": {
            "average_bytes_per_token": bpe_bytes_per_token,
            "parameters": bpe.parameters(),
            "counted_flops_per_byte": bpe_reference["counted_flops"] / length,
        },
        "algebra": algebra(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"FLOPs 집계와 항등식 검증 완료: {args.out}")


if __name__ == "__main__":
    main()
