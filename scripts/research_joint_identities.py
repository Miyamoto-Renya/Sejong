#!/usr/bin/env python3
"""고정 행렬의 표집 평균·분산과 저랭크 SGD 항등식을 계산한다."""

from __future__ import annotations

import itertools
import json
import math

import numpy as np


def _squared_norm(values: np.ndarray) -> float:
    return float(np.sum(values * values))


def _sample_trace_variance(values: np.ndarray) -> float:
    centered = values - values.mean(axis=0)
    return _squared_norm(centered) / (len(values) - 1)


def _weighted_moments(
    values: np.ndarray, probabilities: np.ndarray
) -> tuple[np.ndarray, float]:
    mean = np.tensordot(probabilities, values, axes=(0, 0))
    squared_distances = np.sum(
        (values - mean).reshape(len(values), -1) ** 2, axis=1
    )
    return mean, float(probabilities @ squared_distances)


def _stratified_uniform() -> dict:
    rng = np.random.default_rng(701)
    prediction = rng.normal(size=(6, 2, 3))
    residual = rng.normal(size=prediction.shape)
    residual[:3] -= 4.0
    residual[3:] += 4.0
    truth = prediction + residual
    strata = (tuple(range(3)), tuple(range(3, 6)))
    counts = (1, 2)
    selections = [tuple(itertools.combinations(s, m)) for s, m in zip(strata, counts)]
    estimates = []
    for selected in itertools.product(*selections):
        estimate = prediction.mean(axis=0).copy()
        for stratum, indices in zip(strata, selected):
            estimate += len(stratum) / len(truth) * residual[list(indices)].mean(axis=0)
        estimates.append(estimate)
    estimates = np.asarray(estimates)
    mean, variance = _weighted_moments(
        estimates, np.full(len(estimates), 1.0 / len(estimates))
    )
    formula = sum(
        (len(s) / len(truth)) ** 2
        * (1.0 - m / len(s))
        * _sample_trace_variance(residual[list(s)]) / m
        for s, m in zip(strata, counts)
    )
    return {
        "population": len(truth),
        "stratum_sizes": [len(s) for s in strata],
        "controls_per_stratum": list(counts),
        "enumerated_outcomes": len(estimates),
        "conditional_mean_max_abs_error": float(np.max(np.abs(mean - truth.mean(axis=0)))),
        "enumerated_trace_variance": variance,
        "formula_trace_variance": formula,
        "variance_abs_error": abs(variance - formula),
    }


def _nested_token_sampling() -> dict:
    rng = np.random.default_rng(702)
    population, controls, tokens = 4, 2, 3
    x = rng.normal(size=(population, tokens, 2))
    delta = rng.normal(size=(population, tokens, 3))
    products = np.einsum("mta,mtb->mtab", x, delta)
    truth = products.sum(axis=1)
    prediction = 0.6 * truth + rng.normal(size=truth.shape)
    residual = truth - prediction
    token_probabilities = np.linalg.norm(x, axis=2) * np.linalg.norm(delta, axis=2)
    token_probabilities /= token_probabilities.sum(axis=1, keepdims=True)
    sampled_gradients = products / token_probabilities[:, :, None, None]
    inner_means = np.einsum("mt,mtab->mab", token_probabilities, sampled_gradients)
    inner_variances = np.sum(
        token_probabilities
        * np.sum((sampled_gradients - truth[:, None]) ** 2, axis=(2, 3)),
        axis=1,
    )
    estimates, probabilities = [], []
    outer_probability = 1.0 / math.comb(population, controls)
    for selected in itertools.combinations(range(population), controls):
        for token_indices in itertools.product(range(tokens), repeat=controls):
            estimate = prediction.mean(axis=0).copy()
            probability = outer_probability
            for i, token in zip(selected, token_indices):
                estimate += (sampled_gradients[i, token] - prediction[i]) / controls
                probability *= token_probabilities[i, token]
            estimates.append(estimate)
            probabilities.append(probability)
    mean, variance = _weighted_moments(np.asarray(estimates), np.asarray(probabilities))
    outer_variance = (
        (1.0 - controls / population)
        * _sample_trace_variance(residual) / controls
    )
    inner_addition = float(inner_variances.mean() / controls)
    return {
        "population": population,
        "outer_controls": controls,
        "tokens_per_control": tokens,
        "sampled_tokens_per_control": 1,
        "enumerated_outcomes": len(estimates),
        "probability_sum_abs_error": abs(sum(probabilities) - 1.0),
        "inner_conditional_mean_max_abs_error": float(np.max(np.abs(inner_means - truth))),
        "conditional_mean_max_abs_error": float(np.max(np.abs(mean - truth.mean(axis=0)))),
        "enumerated_trace_variance": variance,
        "outer_trace_variance": outer_variance,
        "inner_variance_addition": inner_addition,
        "variance_decomposition_abs_error": abs(variance - outer_variance - inner_addition),
    }


def _lazy_low_rank_sgd() -> dict:
    rng = np.random.default_rng(703)
    input_width, output_width, rank = 7, 9, 3
    left, _ = np.linalg.qr(rng.normal(size=(input_width, rank)))
    right, _ = np.linalg.qr(rng.normal(size=(output_width, rank)))
    dense = rng.normal(size=(input_width, output_width))
    base = dense.copy()
    accumulated = np.zeros((rank, rank))
    errors = {"effective_weight": 0.0, "forward": 0.0, "input_gradient": 0.0, "flush": 0.0}
    flushes = 0
    for step in range(9):
        # 같은 step의 가중치로 순전파와 입력 기울기를 계산한다.
        x = rng.normal(size=(5, input_width))
        delta = rng.normal(size=(5, output_width))
        effective = base + left @ accumulated @ right.T
        factored_forward = x @ base + (x @ left) @ accumulated @ right.T
        factored_dx = delta @ base.T + (delta @ right) @ accumulated.T @ left.T
        errors["effective_weight"] = max(errors["effective_weight"], float(np.max(np.abs(dense - effective))))
        errors["forward"] = max(errors["forward"], float(np.max(np.abs(x @ dense - factored_forward))))
        errors["input_gradient"] = max(errors["input_gradient"], float(np.max(np.abs(delta @ dense.T - factored_dx))))

        control_gradient = rng.normal(size=dense.shape)
        coefficient_correction = rng.normal(size=accumulated.shape)
        learning_rate = 0.01 / (step + 1)
        dense -= learning_rate * (control_gradient + left @ coefficient_correction @ right.T)
        base -= learning_rate * control_gradient
        accumulated -= learning_rate * coefficient_correction
        errors["effective_weight"] = max(
            errors["effective_weight"],
            float(np.max(np.abs(dense - (base + left @ accumulated @ right.T)))),
        )
        if (step + 1) % 3 == 0:
            before_flush = base + left @ accumulated @ right.T
            base += left @ accumulated @ right.T
            accumulated.fill(0.0)
            errors["flush"] = max(errors["flush"], float(np.max(np.abs(base - before_flush))))
            flushes += 1
    return {
        "steps": 9,
        "flushes": flushes,
        "matrix_shape": [input_width, output_width],
        "rank": rank,
        **{name + "_max_abs_error": value for name, value in errors.items()},
    }


def _fixed_past_calibration() -> dict:
    rng = np.random.default_rng(704)
    population, controls, rank = 6, 2, 2
    left, _ = np.linalg.qr(rng.normal(size=(5, rank)))
    right, _ = np.linalg.qr(rng.normal(size=(4, rank)))
    past_b = rng.normal(size=(8, rank, rank))
    past_k = 0.65 * past_b + 0.08 * rng.normal(size=past_b.shape)
    centered_b = past_b - past_b.mean(axis=0)
    centered_k = past_k - past_k.mean(axis=0)
    # 과거 계수로 구한 alpha를 현재 표본 선택 전체에 고정한다.
    alpha = float(np.sum(centered_k * centered_b) / np.sum(centered_b * centered_b))
    coefficients = rng.normal(size=(population, rank, rank))
    target_coefficients = 0.65 * coefficients + 0.08 * rng.normal(size=coefficients.shape)
    prediction = np.asarray([left @ b @ right.T for b in coefficients])
    ambient = rng.normal(size=prediction.shape)
    orthogonal = np.asarray([g - left @ (left.T @ g @ right) @ right.T for g in ambient])
    truth = np.asarray([left @ k @ right.T for k in target_coefficients]) + 0.1 * orthogonal
    projected = np.asarray([left.T @ g @ right for g in truth])
    centered_g = truth - truth.mean(axis=0)
    centered_c = prediction - prediction.mean(axis=0)
    centered_current_b = coefficients - coefficients.mean(axis=0)
    centered_current_k = projected - projected.mean(axis=0)
    covariance = float(np.sum(centered_g * centered_c) / (population - 1))
    projected_covariance = float(
        np.sum(centered_current_k * centered_current_b) / (population - 1)
    )
    residual = truth - alpha * prediction
    estimates = np.asarray([
        alpha * prediction.mean(axis=0) + residual[list(selected)].mean(axis=0)
        for selected in itertools.combinations(range(population), controls)
    ])
    mean, variance = _weighted_moments(
        estimates, np.full(len(estimates), 1.0 / len(estimates))
    )
    residual_variance = _sample_trace_variance(residual)
    quadratic = (
        _sample_trace_variance(truth) - 2.0 * alpha * covariance
        + alpha * alpha * _sample_trace_variance(prediction)
    )
    formula = (1.0 - controls / population) * residual_variance / controls
    return {
        "past_calibration_samples": len(past_b),
        "population": population,
        "controls": controls,
        "fixed_alpha_from_past": alpha,
        "enumerated_outcomes": len(estimates),
        "conditional_mean_max_abs_error": float(np.max(np.abs(mean - truth.mean(axis=0)))),
        "enumerated_trace_variance": variance,
        "formula_trace_variance": formula,
        "variance_abs_error": abs(variance - formula),
        "residual_variance_quadratic_abs_error": abs(residual_variance - quadratic),
        "projected_covariance_abs_error": abs(covariance - projected_covariance),
        "residual_trace_variance_alpha_one": _sample_trace_variance(truth - prediction),
        "residual_trace_variance_calibrated": residual_variance,
    }


def run_checks() -> dict:
    """네 행렬 항등식을 검산하고 JSON으로 저장할 수 있는 수치를 반환한다."""
    checks = {
        "stratified_uniform": _stratified_uniform(),
        "nested_token_sampling": _nested_token_sampling(),
        "lazy_low_rank_sgd": _lazy_low_rank_sgd(),
        "fixed_past_calibration": _fixed_past_calibration(),
    }
    tolerance = 1e-10
    errors = [
        value for result in checks.values() for key, value in result.items()
        if key.endswith("_error")
    ]
    for error in errors:
        if not math.isfinite(error) or error > tolerance:
            raise AssertionError(f"항등식 오차 {error}가 허용치 {tolerance}를 넘었다.")
    return {
        "scope": "고정된 float64 행렬의 표집 항등식과 9회 SGD 갱신 항등식",
        "tolerance": tolerance,
        "max_abs_error": max(errors),
        "checks": checks,
    }


if __name__ == "__main__":
    print(json.dumps(run_checks(), ensure_ascii=False, indent=2, allow_nan=False))
