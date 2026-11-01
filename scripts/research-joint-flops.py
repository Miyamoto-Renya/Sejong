#!/usr/bin/env python3
"""계층 모델·제어 표집·저랭크 누적의 주요 연산과 타일 비용을 계산한다."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np

from research_joint_identities import run_checks


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("sejong_flop_algebra", ROOT / "scripts/research-flops.py")
base = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = base
SPEC.loader.exec_module(base)
Shape, mm = base.Shape, base.mm
GLOBAL = Shape(2560, 32, 12800, 40)
ENCODER = Shape(256, 2, 768, 4)
DECODER = Shape(512, 2, 1536, 8)
PROXY = Shape(512, 4, 1536, 8)
EXPERT_WIDTHS = [896] * 21 + [864] * 11
EXPERTS, TOP_K, KV_WIDTH = 16, 2, 512
FEATURE, RANK, REFRESH = 64, 32, 32


def roundup(n: int, block: int) -> int:
    return (n + block - 1) // block * block


def expected_expert_cost(positions: int, padded: bool) -> dict:
    """각 위치에서 균등·독립 top-2를 선택한 분포의 주변 기대값을 합한다."""
    probability = TOP_K / EXPERTS
    probabilities = [math.comb(positions, n) * probability**n * (1 - probability)**(positions - n)
                     for n in range(positions + 1)]
    rows16 = sum(p * roundup(n, 16) for n, p in enumerate(probabilities))
    rows32 = sum(p * roundup(n, 32) for n, p in enumerate(probabilities))
    average = positions * probability
    ffn_scale = GLOBAL.width * sum(EXPERT_WIDTHS) * EXPERTS
    forward = 6 * ffn_scale * (rows16 if padded else average)
    backward = 6 * ffn_scale * ((rows16 + rows32) if padded else 2 * average)
    return {"forward": forward, "backward": backward,
            "mean_tokens_per_expert": average, "mean_rounded_rows_16": rows16,
            "mean_rounded_rows_32": rows32,
            "mean_touched_experts_per_layer": EXPERTS * (1 - (1 - probability)**positions),
            "ffn_padding_ratio": (12 * rows16 + 6 * rows32) / (18 * average)}


def matrix_cost(matrices: list[tuple[int, int, int]], padded: bool) -> tuple[int, int]:
    forward = sum(mm(t, b, a, padded) for a, b, t in matrices)
    backward = sum(mm(t, a, b, padded) + mm(a, b, t, padded) for a, b, t in matrices)
    return forward, backward


def local_cost(shape: Shape, length: int, head: bool, padded: bool) -> tuple[int, int]:
    forward, backward = matrix_cost([(a, b, length) for a, b in shape.matrices(head)], padded)
    # 제안한 local kernel은 각 query의 유효 window 시작점부터 32개씩 읽는다.
    pairs = sum(roundup(min(t + 1, 128), 32) if padded else min(t + 1, 128)
                for t in range(length))
    return (forward + 4 * shape.width * pairs * shape.layers,
            backward + (16 * shape.width * pairs + 2 * length * shape.width) * shape.layers)


def global_matrices(length: int, sparse: bool) -> list[tuple[int, int, int]]:
    d = GLOBAL.width
    if sparse:
        return [(a, b, length) for a, b in [(d, d), (d, KV_WIDTH), (d, KV_WIDTH), (d, d), (d, EXPERTS)] * GLOBAL.layers]
    return [(a, b, length) for a, b in GLOBAL.matrices(False)]


def global_cost(length: int, sparse: bool, padded: bool) -> tuple[float, float]:
    forward, backward = matrix_cost(global_matrices(length, sparse), padded)
    af, ab = base.attention(GLOBAL, length, padded)
    if sparse:
        expert = expected_expert_cost(length, padded)
        forward += expert["forward"]
        backward += expert["backward"]
        # query별 dK/dV를 KV head로 합치는 덧셈을 집계한다.
        backward += 2 * length * (GLOBAL.width - KV_WIDTH) * GLOBAL.layers
    return forward + af, backward + ab


def bridge_matrices(length: int, latent: int) -> list[tuple[int, int, int]]:
    return [(256, 2560, latent), (2560, 512, latent), (256, 512, length)]


def architecture(length: int, patch: int, sparse: bool, padded: bool, checkpoint: bool = False) -> dict:
    latent = math.ceil(length / patch)
    ef, eb = local_cost(ENCODER, length, False, padded)
    df, db = local_cost(DECODER, length, True, padded)
    gf, gb = global_cost(latent, sparse, padded)
    bf, bb = matrix_cost(bridge_matrices(length, latent), padded)
    parts = {"encoder_forward": ef, "encoder_backward": eb,
             "global_forward": gf, "global_backward": gb,
             "global_recompute": gf if checkpoint else 0,
             "decoder_forward": df, "decoder_backward": db,
             "bridges": bf + bb, "pool_and_broadcast": 2 * length * (256 + 512)}
    return {"components": parts, "total": sum(parts.values()),
            "parameters": base.multirate(length, patch, False)["parameters"]}


def predictor_cost(matrices: list[tuple[int, int, int]], count: int, controls: int,
                   padded: bool, deferred: bool, feature_input: int, checkpoint_matrices: int) -> dict:
    projection = lift = refresh_gemm = refresh_qr = lazy_extra = checkpoint_extra = 0
    head_forward = head_backward = coefficient_parameters = basis_values = coefficients = 0
    for index, (a, b, t) in enumerate(matrices):
        r = min(RANK, a, b)
        coefficients += r * r
        coefficient_parameters += FEATURE * r * r
        basis_values += (a + b) * r
        head_forward += mm(count, r * r, FEATURE, padded)
        head_backward += mm(controls, FEATURE, r * r, padded) + mm(FEATURE, r * r, controls, padded)
        projection += controls * (mm(t, r, a, padded) + mm(t, r, b, padded) + mm(r, r, t, padded))
        lift += min(mm(a, r, r, padded) + mm(a, b, r, padded),
                    mm(r, b, r, padded) + mm(a, b, r, padded))
        refresh_gemm += 2 * mm(a, r, b, padded) + 2 * mm(b, r, a, padded)
        refresh_gemm += 2 * mm(r, r * FEATURE, r, padded)
        refresh_qr += 4 * (a + b) * r * r
        # XU와 delta V는 projection 항에서 한 번씩 계산한다.
        lazy_extra += controls * (2 * mm(t, r, r, padded) + mm(t, b, r, padded) + mm(t, a, r, padded))
        if index < checkpoint_matrices:
            checkpoint_extra += controls * (mm(t, r, a, padded) + mm(t, r, r, padded) + mm(t, b, r, padded))
    coefficient_parameters += feature_input * FEATURE
    head_forward += mm(count, FEATURE, feature_input, padded)
    head_backward += mm(controls, feature_input, FEATURE, padded) + mm(feature_input, FEATURE, controls, padded)
    parts = {"coefficient_forward": head_forward, "coefficient_backward": head_backward,
             "gradient_target_projection": projection,
             "lift_or_amortized_flush": lift / REFRESH if deferred else lift,
             "deferred_forward_and_dx": lazy_extra if deferred else 0,
             "deferred_recompute": checkpoint_extra if deferred else 0,
             "basis_refresh_gemm_amortized": refresh_gemm / REFRESH,
             "basis_refresh_qr_budget_amortized": refresh_qr / REFRESH,
             "coefficient_loss_accumulation_calibration": (10 * count + 16 * controls) * coefficients,
             "coefficient_optimizer": 2 * coefficient_parameters + 2 * coefficients}
    return {"components": parts, "coefficient_parameters": coefficient_parameters,
            "basis_entries": basis_values, "coefficient_entries": coefficients,
            "eager_lift_flops": lift, "initial_basis_qr_budget": refresh_qr,
            "predicted_matrix_parameters": sum(a * b for a, b, _ in matrices)}


def combined(patch: int, sparse: bool = False, shared: bool = True, deferred: bool = True,
             padded: bool = False, checkpoint: bool = False) -> dict:
    length, count, controls = 512, 8, 2
    latent = length // patch
    arch = architecture(length, patch, sparse, padded, checkpoint)
    components = {key: controls * value for key, value in arch["components"].items()}
    matrices = global_matrices(latent, sparse)
    matrices += [(a, b, length) for a, b in DECODER.matrices()]
    matrices += bridge_matrices(length, latent)
    if shared:
        components["encoder_forward"] = count * arch["components"]["encoder_forward"]
        feature_width = ENCODER.width
    else:
        components["independent_proxy_training"] = count * base.ledger(PROXY, length, False, padded)["counted_flops"]
        components["independent_proxy_optimizer"] = 2 * PROXY.parameters()
        matrices += [(a, b, length) for a, b in ENCODER.matrices(False)]
        feature_width = PROXY.width
    predictor = predictor_cost(matrices, count, controls, padded, deferred, 2 * feature_width,
                               len(global_matrices(latent, sparse)) if checkpoint else 0)
    components.update(predictor["components"])
    p = arch["parameters"]
    components.update({"feature_pool_products": 4 * count * length * feature_width,
                       "gradient_diagnostics": 3 * p + 8 * predictor["coefficient_entries"],
                       "control_gradient_accumulation_and_scale": 2 * controls * p,
                       "main_sgd": 2 * p})
    total = sum(components.values())
    dense_reference = count * base.ledger(GLOBAL, length, False, padded)["counted_flops"] + (2 * count + 2) * GLOBAL.parameters()
    own_reference = count * arch["total"] + (2 * count + 2) * p
    gamma = total / own_reference
    return {"patch_bytes": patch, "sparse_gqa_experts": sparse, "shared_encoder": shared,
            "deferred_lift": deferred, "tile_padding_counted": padded,
            "global_checkpoint": checkpoint, "microbatches": count, "controls": controls,
            "parameters": p, "coefficient_parameters": predictor["coefficient_parameters"],
            "basis_entries": predictor["basis_entries"],
            "components": components, "counted_flops": total,
            "counted_gflops_per_raw_byte": total / (count * length * 1e9),
            "dense_byte_reference_counted_flops": dense_reference,
            "own_architecture_reference_counted_flops": own_reference,
            "dense_byte_reference_over_candidate": dense_reference / total,
            "own_architecture_flop_ratio": gamma,
            "own_architecture_variance_cost_scores": {str(q): gamma * (1 + 3 * q) for q in (0, 0.1, 0.25, 0.5, 1)},
            "own_architecture_q_break_even": (1 / gamma - 1) / 3,
            "encoder_replay_if_cache_released": controls * arch["components"]["encoder_forward"] if shared else 0,
            "initial_basis_qr_budget": predictor["initial_basis_qr_budget"],
            "eager_lift_flops": predictor["eager_lift_flops"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    candidates = {}
    for padded in (False, True):
        suffix = "tiled" if padded else "logical"
        for patch in (4, 8, 16):
            for sparse in (False, True):
                name = f"p{patch}_{'sparse' if sparse else 'dense'}_{suffix}"
                candidates[name] = combined(patch, sparse=sparse, padded=padded)
        for name, shared, deferred in (("independent_eager", False, False), ("shared_eager", True, False)):
            candidates[f"p8_dense_{name}_{suffix}"] = combined(8, shared=shared, deferred=deferred, padded=padded)
        candidates[f"p8_dense_checkpoint_{suffix}"] = combined(8, padded=padded, checkpoint=True)
    body_parameters = 4 * GLOBAL.layers * GLOBAL.width**2 + 3 * GLOBAL.layers * GLOBAL.width * GLOBAL.ffn
    sparse_body = sum(a * b for a, b, _ in global_matrices(64, True)) + 3 * EXPERTS * GLOBAL.width * sum(EXPERT_WIDTHS)
    assert body_parameters == sparse_body
    for patch in (4, 8, 16):
        calculated = architecture(512, patch, False, False)["total"]
        assert calculated == base.multirate(512, patch, False)["counted_flops"]
    sources = [Path(__file__), ROOT / "scripts/research-flops.py", ROOT / "scripts/research_joint_identities.py"]
    result = {
        "scope": "형상 기반 주요 GEMM·attention dot·SGD·기울기 통계; FMA=2; FLOPs 산술과 FP64 항등식",
        "separate_operation_classes": ["exp/sigmoid/norm", "routing top-k comparisons and gather/scatter", "dtype conversion", "memory transfer"],
        "assumptions": {"bytes_per_microbatch": 512, "microbatches": 8, "controls": 2,
                        "rank": RANK, "rank_clamped_to_matrix_dimensions": True, "basis_refresh_steps": REFRESH,
                        "control_execution": "각 control의 GEMM과 dW를 별도 실행; 두 gradient의 차이로 진단·기저 갱신",
                        "expert_routing_scenario": "각 위치에서 균등·독립 top-2, 빈 expert 생략, token 보존",
                        "shared_encoder_cache": "전체 M개의 작은 encoder 활성값을 control 선택까지 보존",
                        "unpredicted_gradients": "공유 encoder, embedding/norm/BOS, sparse expert의 c=0"},
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "parameters": {"dense_byte": GLOBAL.parameters(), "sparse_byte": sparse_body + GLOBAL.parameters() - body_parameters,
                       "hierarchy": base.multirate(512, 8, False)["parameters"],
                       "sparse_active_per_position": sum(a * b for a, b, _ in global_matrices(64, True))
                       + 3 * TOP_K * GLOBAL.width * sum(EXPERT_WIDTHS) + GLOBAL.parameters() - body_parameters},
        "references": {"dense_byte_gflops_per_byte": base.ledger(GLOBAL, 512, False)["counted_flops"] / 512e9,
                       "bpe_assumed_four_bytes_per_token_gflops_per_byte": base.ledger(Shape(2560, 32, 12160, 40, 32768), 128, False)["counted_flops"] / 512e9},
        "candidates": candidates,
        "expert_packing_scenarios": {f"p{patch}_{'packed' if packed else 'separate'}":
            expected_expert_cost((512 // patch) * (2 if packed else 1), True)
            for patch in (4, 8, 16) for packed in (False, True)},
        "identity_checks": run_checks(),
    }
    output = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.out:
        args.out.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
