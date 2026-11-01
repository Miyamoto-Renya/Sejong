#!/usr/bin/env python3
"""기울기 추정식의 유한 열거 검증과 Sejong 형상·전송량 계산."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import tomllib

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3
MIB = 1024**2


def require_close(name: str, actual, expected) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    error = float(np.max(np.abs(actual - expected)))
    if not np.allclose(actual, expected, rtol=1e-11, atol=1e-11):
        raise ArithmeticError(f"{name}: 오차 {error:.6g}")
    return error


def finite_population(gradients: np.ndarray, predictions: np.ndarray, m: int) -> dict:
    """고정한 관측·예측에서 모든 제어 부분집합을 같은 확률로 열거한다."""
    count = gradients.shape[0]
    residual = gradients - predictions
    truth = gradients.mean(axis=0)
    estimate = np.stack([
        predictions.mean(axis=0) + residual[list(indices)].mean(axis=0)
        for indices in itertools.combinations(range(count), m)
    ])
    centered = residual - residual.mean(axis=0)
    residual_variance = float(np.square(centered).sum() / (count - 1))
    predicted_variance = (1 - m / count) * residual_variance / m
    observed_variance = float(np.square(estimate - truth).sum(axis=1).mean())
    return {
        "microbatches": count,
        "controls": m,
        "subsets": len(estimate),
        "mean_max_abs_error": require_close("제어 표본 평균", estimate.mean(axis=0), truth),
        "variance_abs_error": require_close("제어 표본 분산", observed_variance, predicted_variance),
        "observed_variance": observed_variance,
        "predicted_variance": predicted_variance,
    }


def identities() -> dict:
    rng = np.random.default_rng(6600)
    dimension = 8
    gradient = rng.normal(size=dimension)
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=dimension)))
    forward_estimates = (signs @ gradient)[:, None] * signs
    expected_variance = (dimension - 1) * float(gradient @ gradient)
    forward = {
        "dimension": dimension,
        "directions": len(signs),
        "mean_max_abs_error": require_close("방향미분 평균", forward_estimates.mean(axis=0), gradient),
        "variance_abs_error": require_close(
            "방향미분 분산",
            np.square(forward_estimates - gradient).sum(axis=1).mean(),
            expected_variance,
        ),
        "relative_mse_one_direction": dimension - 1,
    }

    gradients = rng.normal(size=(8, 24))
    predictions = 0.8 * gradients + rng.normal(scale=0.2, size=gradients.shape)
    checks = {
        f"dense_m{m}": finite_population(gradients, predictions, m)
        for m in (1, 2, 4, 8)
    }
    bias = rng.normal(size=(1, 24))
    checks["constant_prediction_bias"] = finite_population(gradients, gradients + bias, 2)

    sparse = np.zeros((8, 64))
    for i in range(8):
        sparse[i, i * 8:(i + 1) * 8] = rng.normal(size=8)
    checks["disjoint_page_support"] = finite_population(sparse, 0.75 * sparse, 2)

    x = rng.normal(size=(7, 5))
    delta = rng.normal(size=(7, 3))
    predicted_delta = rng.normal(size=(7, 3))
    microbatches, controls = 8, 2
    mixed = delta / controls + (1 / microbatches - 1 / controls) * predicted_delta
    separate = (x.T @ predicted_delta) / microbatches + (x.T @ (delta - predicted_delta)) / controls
    mixed_error = require_close("선형 연산의 두 수정 신호 결합", x.T @ mixed, separate)

    # 제어 표본을 보고 같은 배치의 예측을 바꿀 때 생기는 편향을 계산한다.
    selected_fit = np.stack([
        gradients[list(indices)].sum(axis=0) / len(gradients)
        for indices in itertools.combinations(range(len(gradients)), 2)
    ])
    bias_example_error = require_close(
        "표본 선택에 종속된 예측의 평균",
        selected_fit.mean(axis=0),
        2 / len(gradients) * gradients.mean(axis=0),
    )
    return {
        "scope": "고정 벡터·행렬의 추정량 항등식, 모든 부호·제어 부분집합 열거",
        "forward_gradient": forward,
        "control_subset": checks,
        "mixed_adjoint_max_abs_error": mixed_error,
        "selection_dependent_predictor": {
            "mean_factor_relative_to_true_gradient": 0.25,
            "identity_max_abs_error": bias_example_error,
        },
    }


def dense_parameters(d: int, layers: int, f: int, vocabulary: int = 257) -> int:
    return layers * (4 * d * d + 3 * d * f + 2 * d) + 2 * vocabulary * d + d


def page_traffic(sequence: int, span: int, banks: int, pages: int, topk: int,
                 page_bytes: int, accumulation: int) -> dict:
    decisions = math.ceil(sequence / span)
    # 한 결정은 서로 다른 topk개 페이지를 균등 선택한다. 결정끼리는 독립이다.
    distinct = banks * pages * (-math.expm1(decisions * math.log1p(-topk / pages)))
    max_pages = banks * min(pages, decisions * topk)
    accumulated_max = banks * min(pages, decisions * topk * accumulation)
    return {
        "span_bytes": span,
        "sequence_bytes": sequence,
        "routing_decisions_per_bank": decisions,
        "uniform_expected_distinct_pages": distinct,
        "cold_weight_h2d_mib": distinct * page_bytes / MIB,
        "cold_weight_h2d_d2h_mib": 2 * distinct * page_bytes / MIB,
        "weight_cache_hit_assumption": 0,
        "max_gradient_pages_per_microbatch": max_pages,
        "max_gradient_mib_per_microbatch": max_pages * page_bytes / MIB,
        "max_gradient_mib_per_accumulation": accumulated_max * page_bytes / MIB,
    }


def costs() -> dict:
    config = tomllib.loads((ROOT / "configs/sejong.toml").read_text())
    current = config["model"]
    count = dense_parameters(current["width"], current["layers"], current["ffn_width"])
    d, layers, f, vocabulary = 1024, 12, 2048, 257
    banks, key_side, page_width, topk, span, rank = 4, 86, 64, 2, 32, 32
    pages = key_side**2
    page_parameters = 2 * d * page_width
    core = dense_parameters(d, layers, f, vocabulary)
    memory = banks * pages * page_parameters
    router = banks * key_side * d
    total = core + memory + router
    factored_state = banks * pages * 2 * (d + page_width) * 4

    affine_predictors = layers * rank * (16 * d + 5 * f + 14 * vocabulary + 7)
    norm_and_bank_predictors = (2 * layers + 1 + banks) * rank * (3 * d + 2 * vocabulary + 1)
    embedding_predictor = rank * (2 * d + 2 * vocabulary + 1)
    predictors = affine_predictors + norm_and_bank_predictors + embedding_predictor
    core_linear_parameters = layers * (4 * d * d + 3 * d * f) + vocabulary * d
    predictor_blocks = 7 * layers + 2 * layers + 1 + banks + 1
    predictor_linear_parameters = predictors - predictor_blocks * rank
    predictor_output_parameters = rank * (layers * (5 * d + 2 * f) + (2 * layers + 2 + banks) * d)
    predictor_forward_ratio = predictor_linear_parameters / core_linear_parameters
    # 예측기 입력을 고정하면 학습은 각 가중치 미분과 마지막 투영의 입력 미분을 계산한다.
    predictor_backward_ratio = (predictor_linear_parameters + predictor_output_parameters) / core_linear_parameters
    base_state_gib = 18 * (core + router) / GIB
    predictor_state_gib = 16 * predictors / GIB
    cache_gib, activations_gib, scratch_gib = 2.0, 1.0, 0.75
    page_workspace_gib = 64 / 1024
    gpu_budget_gib = base_state_gib + predictor_state_gib + cache_gib + activations_gib + scratch_gib + page_workspace_gib

    traffic = [page_traffic(512, s, banks, pages, topk, page_parameters * 4, 8)
               for s in (1, 8, 32)]
    # CPU가 page gradient를 모아 optimizer를 적용하는 설계: 가중치 H2D + 기울기 D2H.
    cold_span_transfer = 2 * traffic[-1]["cold_weight_h2d_mib"] * MIB
    sensitivities = []
    for m in (1, 2, 4, 8):
        p = m / 8
        overhead = predictor_forward_ratio + p * predictor_backward_ratio
        relative_cost = (2 + p + overhead) / 3
        q_limit = (1 / relative_cost - 1) / (1 / p - 1) if m < 8 else None
        sensitivities.append({
            "controls_of_eight": m,
            "predictor_matmul_overhead_forward_units": overhead,
            "matmul_cost_ratio_to_exact_no_recompute": relative_cost,
            "matmul_step_speed_ratio": 1 / relative_cost,
            "centered_residual_variance_ratio_break_even": q_limit,
        })
    return {
        "current_default": {
            "parameters": count,
            "fp32_weight_gradient_adam_floor_gib": count * 16 / GIB,
            "weight_storage_dtype": "fp32",
            "token_unit": "UTF-8 byte + EOS",
        },
        "dense_4b": {
            "parameters": 4_000_000_000,
            "fp32_weight_gradient_adam_floor_gib": 4_000_000_000 * 16 / GIB,
            "fp16_weights_gib": 4_000_000_000 * 2 / GIB,
            "forward_gradient_relative_mse_32_rademacher_directions": (4_000_000_000 - 1) / 32,
        },
        "candidate_shape": {
            "width": d, "layers": layers, "ffn_width": f, "vocabulary": vocabulary,
            "banks": banks, "key_side": key_side, "pages_per_bank": pages,
            "page_hidden_width": page_width, "selected_pages_per_bank": topk,
            "routing_span_bytes": span, "predictor_rank": rank,
            "core_parameters": core, "memory_parameters": memory,
            "router_parameters": router, "inference_parameters": total,
            "training_predictor_parameters": predictors,
            "page_parameters": page_parameters,
            "page_fp32_mib": page_parameters * 4 / MIB,
            "active_memory_parameters_per_byte": banks * topk * page_parameters,
            "resident_core_and_router_plus_active_memory_parameters": core + router + banks * topk * page_parameters,
            "host_memory_weights_gib": memory * 4 / GIB,
            "host_factored_second_moments_gib": factored_state / GIB,
            "host_pending_page_gradients_max_gib": traffic[-1]["max_gradient_mib_per_accumulation"] / 1024,
        },
        "gpu_budget_assumptions_gib": {
            "core_router_fp32_weights_gradients_adam_fp16_copy": base_state_gib,
            "predictors_fp32_weights_gradients_adam": predictor_state_gib,
            "fp32_page_cache": cache_gib,
            "activations_reservation": activations_gib,
            "scratch_reservation": scratch_gib,
            "page_gradient_staging": page_workspace_gib,
            "sum": gpu_budget_gib,
        },
        "routing_traffic": traffic,
        "span32_cold_transfer_time_s_at_assumed_effective_gbytes_s": {
            str(bw): cold_span_transfer / (bw * 1e9) for bw in (4, 8, 12)
        },
        "gradient_arithmetic_sensitivity": sensitivities,
        "predictor_arithmetic": {
            "reference_core_linear_parameters": core_linear_parameters,
            "predictor_forward_matmul_ratio": predictor_forward_ratio,
            "predictor_backward_matmul_ratio_per_control": predictor_backward_ratio,
            "accounting_scope": "선형 행렬곱; attention·점별 연산·optimizer·전송은 별도 항",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="계산 결과 JSON 경로")
    args = parser.parse_args()
    report = {
        "date": "2026-11-01",
        "execution": {
            "kind": "cpu_algebra_and_shape_accounting",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "gpu_device_paths": {p: Path(p).exists() for p in ("/dev/kfd", "/dev/dri")},
        },
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "configs/sejong.toml",
                "crates/sejong-model/src/lib.rs",
                "crates/sejong-train/src/lib.rs",
                "crates/sejong-gpu/src/graph.rs",
                "scripts/research-gradient-cost.py",
            )
        },
        "identities": identities(),
        "costs": costs(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"수식 검증과 형상 계산 완료: {args.out}")


if __name__ == "__main__":
    main()
