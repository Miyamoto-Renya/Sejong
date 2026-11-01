#!/usr/bin/env python3
"""실제 Sejong 역전파 trace의 spectrum·VJP 오차·블록 표집 비용을 분석한다."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np


def norm2(x):
    return float(np.sum(np.square(x)))


def ratio(a, b):
    return a / b if b else 0.0


def gemm(m, n, k, tile):
    bm, bn, bk = tile
    return 2 * math.ceil(m / bm) * bm * math.ceil(n / bn) * bn * math.ceil(k / bk) * bk


class Trace:
    def __init__(self, path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "trace.json").read_text())
        self.linear = self.meta["linear_operations"]
        self.operations = {r["parameter"]: r for r in self.meta["linear_operations"]}
        digest = hashlib.sha256()
        for tensor in sorted(self.path.glob("*.f32")):
            digest.update(tensor.name.encode())
            digest.update(tensor.read_bytes())
        self.tensor_sha256 = digest.hexdigest()

    def read(self, record):
        return np.fromfile(self.path / record["file"], dtype="<f4").reshape(record["shape"]).astype(np.float64)

    def value(self, ident):
        return self.read(self.meta["tensors"][str(ident)])


def stage(name):
    if name.startswith("layers."):
        return "global_expert" if ".experts." in name else "global_shared"
    if name.startswith("encoder."):
        return "encoder"
    if name.startswith("decoder."):
        return "decoder"
    return "bridge_and_output"


def spectrum(energy, levels):
    total = float(np.sum(energy))
    cumsum = np.cumsum(energy) / total if total else np.zeros_like(energy)
    return {str(level): int(np.searchsorted(cumsum, level) + 1) if total else 0 for level in levels}


def analyse_linear(trace, ranks, levels, tile):
    records = []
    for op in trace.linear:
        name = op["parameter"]
        x, w, dy = trace.value(op["input"]), trace.value(op["weight"]), trace.read(op["delta"])
        n, a = x.shape
        b = w.shape[1]
        u, s, vt = np.linalg.svd(dy, full_matrices=False)
        dx, dw = dy @ w.T, x.T @ dy
        baseline_logical = 4 * n * a * b
        baseline_issued = gemm(n, a, b, tile) + gemm(a, b, n, tile)
        forward_issued = gemm(n, b, a, tile)
        checkpoint_issued = forward_issued if name.startswith("layers.") else 0
        rank_records = []
        for requested in ranks:
            r = min(requested, min(n, b))
            approx = (u[:, :r] * s[:r]) @ vt[:r]
            e = dy - approx
            # 짧은 축의 기저가 주어진 projection과 두 VJP의 GEMM을 합산한다.
            projection = gemm(r,b,n,tile) if n <= b else gemm(n,r,b,tile)
            cost = (projection + gemm(a, r, b, tile)
                    + gemm(n, a, r, tile) + gemm(a, r, n, tile) + gemm(a, b, r, tile))
            logical = 2 * n * b * r + 4 * a * r * (n + b)
            if r == min(n, b):
                cost = baseline_issued
                logical = baseline_logical
            rank_records.append({"requested_rank": requested, "rank": r,
                "delta_tail_fraction": ratio(norm2(e), norm2(dy)),
                "dx_relative_squared_error": ratio(norm2(e @ w.T), norm2(dx)),
                "dw_relative_squared_error": ratio(norm2(x.T @ e), norm2(dw)),
                "dw_error_squared": norm2(x.T @ e),
                "logical_backward_flops_known_basis": logical,
                "issued_backward_flops_known_basis": cost,
                "issued_backward_ratio_known_basis": ratio(cost, baseline_issued),
                "issued_total_ratio_known_basis": ratio(forward_issued + checkpoint_issued + cost,
                    forward_issued + checkpoint_issued + baseline_issued),
                "orthogonal_residual_probe_relative_mse": {
                    str(probes): (min(n,b) - r - probes) / probes * ratio(norm2(e), norm2(dy))
                    for probes in ranks if 0 < probes <= min(n,b) - r}})
        records.append({"parameter": name, "output_variable":op["output"], "stage": stage(name), "shape": [n, a, b],
            "delta_energy":norm2(dy),"dx_energy":norm2(dx),"dw_energy":norm2(dw),
            "delta_energy_ranks": spectrum(s*s,levels),
            "baseline_logical_backward_flops":baseline_logical,
            "baseline_issued_backward_flops":baseline_issued,
            "forward_issued_flops":forward_issued,"checkpoint_issued_flops":checkpoint_issued,
            "rank_candidates":rank_records})
    groups = {}
    for group in sorted({r["stage"] for r in records} | {"all"}):
        selected = [r for r in records if group == "all" or r["stage"] == group]
        den = sum(r["baseline_issued_backward_flops"] for r in selected)
        forward = sum(r["forward_issued_flops"] + r["checkpoint_issued_flops"] for r in selected)
        groups[group] = {"operations":len(selected),"baseline_issued_backward_flops":den,
            "forward_and_checkpoint_issued_flops":forward,
            "median_energy_ranks":{str(level):float(np.median([r["delta_energy_ranks"][str(level)] for r in selected])) for level in levels},
            "rank_candidates":[]}
        for index, requested in enumerate(ranks):
            candidates = [r["rank_candidates"][index] for r in selected]
            # 형상 비교에서는 기존 VJP와 압축 VJP 중 issued FLOPs가 작은 경로를 고른다.
            selected_cost = sum(min(c["issued_backward_flops_known_basis"],r["baseline_issued_backward_flops"])
                                for c,r in zip(candidates,selected))
            groups[group]["rank_candidates"].append({"requested_rank":requested,
                "issued_backward_ratio_known_basis_with_dense_fallback":ratio(selected_cost,den),
                "issued_total_ratio_known_basis_with_dense_fallback":ratio(forward+selected_cost,forward+den),
                "median_delta_tail_fraction":float(np.median([c["delta_tail_fraction"] for c in candidates])),
                "sum_operator_dw_relative_squared_error":ratio(sum(c["dw_error_squared"] for c,r in zip(candidates,selected)
                    if c["issued_backward_flops_known_basis"] < r["baseline_issued_backward_flops"]),sum(r["dw_energy"] for r in selected))})
    return records, groups


def importance_probabilities(energies, budget):
    weights = np.sqrt(np.asarray(energies))
    if not np.any(weights):
        return np.full(weights.shape, budget / len(weights))
    low, high = 0.0, len(weights) / min(weights[weights > 0])
    # 기계 정밀도까지 budget 방정식을 이분한다.
    while np.nextafter(low, high) < high:
        mid = (low + high) / 2
        if mid == low or mid == high:
            break
        if float(np.sum(np.minimum(1, mid * weights))) < budget:
            low = mid
        else:
            high = mid
    return np.minimum(1, high * weights)


def analyse_ffn(trace, block, fractions, levels, tile):
    records = []
    for name, down in trace.operations.items():
        if not name.endswith(".down"):
            continue
        prefix = name.removesuffix(".down")
        gate, up = trace.operations[prefix+".gate"], trace.operations[prefix+".up"]
        x, wg, wu, wd = trace.value(gate["input"]), trace.value(gate["weight"]), trace.value(up["weight"]), trace.value(down["weight"])
        g, u, h, delta = trace.value(gate["output"]), trace.value(up["output"]), trace.value(down["input"]), trace.read(down["delta"])
        n, d = x.shape
        f = wg.shape[1]
        if f % block:
            raise ValueError(f"{name} hidden width {f} must be divisible by research block {block}")
        sigmoid = 1/(1+np.exp(-g))
        jac_g = u*sigmoid*(1+g*(1-sigmoid))
        jac_u = g*sigmoid
        dh = delta @ wd.T
        dg, du = dh*jac_g, dh*jac_u
        dx = dg@wg.T + du@wu.T
        dwg, dwu, dwd = x.T@dg, x.T@du, h.T@delta
        param_energy = norm2(dwg)+norm2(dwu)+norm2(dwd)
        contributions, block_param, block_dx = [], [], []
        for start in range(0,f,block):
            sl = slice(start,start+block)
            c = dg[:,sl]@wg[:,sl].T + du[:,sl]@wu[:,sl].T
            contributions.append(c)
            block_dx.append(norm2(c))
            block_param.append(norm2(dwg[:,sl])+norm2(dwu[:,sl])+norm2(dwd[sl]))
        blocks = len(block_param)
        dx_sum_error = norm2(np.sum(contributions,axis=0)-dx)
        actual_g = trace.read(gate["delta"])
        actual_u = trace.read(up["delta"])
        dg_reference_error = ratio(norm2(dg-actual_g),norm2(actual_g))
        du_reference_error = ratio(norm2(du-actual_u),norm2(actual_u))
        jg_sv = np.linalg.svd(jac_g,compute_uv=False)
        ju_sv = np.linalg.svd(jac_u,compute_uv=False)
        baseline = (gemm(n,f,d,tile) + 2*gemm(n,d,f,tile)
                    + 2*gemm(d,f,n,tile)+gemm(f,d,n,tile))
        forward = 2*gemm(n,f,d,tile)+gemm(n,d,f,tile)
        checkpoint = forward if prefix.startswith("layers.") else 0
        candidates = []
        for fraction in fractions:
            count = min(blocks,max(1,math.ceil(blocks*fraction)))
            p = count/blocks
            k = count*block
            issued = (gemm(n,k,d,tile)+2*gemm(n,d,k,tile)
                      +2*gemm(d,k,n,tile)+gemm(k,d,n,tile))
            dx_noise = ((blocks-count)/(count*(blocks-1)) * (blocks*sum(block_dx)-norm2(dx))) if blocks > 1 else 0.0
            probs = importance_probabilities(block_param,count)
            importance_noise = sum((1/prob-1)*energy for prob,energy in zip(probs,block_param) if prob>0)
            candidates.append({"requested_fraction":fraction,"selected_blocks":count,"blocks":blocks,"channels":k,"probability":p,
                "issued_backward_ratio":ratio(issued,baseline),
                "issued_total_ratio_including_checkpoint":ratio(forward+checkpoint+issued,forward+checkpoint+baseline),
                "uniform_weight_gradient_relative_mse":1/p-1,
                "uniform_dx_relative_mse":ratio(max(0,dx_noise),norm2(dx)),
                "oracle_importance_weight_gradient_relative_mse":ratio(importance_noise,param_energy),
                "oracle_importance_probabilities":probs.tolist()})
        records.append({"prefix":prefix,"stage":stage(prefix),"shape":[n,d,f],
            "parameter_gradient_energy":param_energy,"dx_energy":norm2(dx),"block_parameter_gradient_energy":block_param,
            "block_dx_energy":block_dx,"sum_block_dx_error_squared":dx_sum_error,
            "swiglu_dg_reference_relative_squared_error":dg_reference_error,
            "swiglu_du_reference_relative_squared_error":du_reference_error,
            "jacobian_gate_energy_ranks":spectrum(jg_sv*jg_sv,levels),
            "jacobian_up_energy_ranks":spectrum(ju_sv*ju_sv,levels),
            "baseline_issued_backward_flops":baseline,"forward_and_checkpoint_issued_flops":forward+checkpoint,
            "candidates":candidates})
    return records


def identities(trace):
    record = next(r for r in trace.operations.values() if trace.read(r["delta"]).shape[0]>=8 and trace.read(r["delta"]).shape[1]>=8)
    delta = trace.read(record["delta"])[:8,:8]
    # 실제 cotangent의 좌표에서 가능한 residual subspace 선택을 전부 열거한다.
    u,s,vt = np.linalg.svd(delta,full_matrices=False)
    head, probes, dimension = 2, 2, 8
    primary = (u[:,:head]*s[:head])@vt[:head]
    residual = delta-primary
    sketches = []
    for choice in itertools.combinations(range(head,dimension),probes):
        sketches.append(primary+(dimension-head)/probes*(u[:,choice]@(u[:,choice].T@delta)))
    mean = np.mean(sketches,axis=0)
    mse = float(np.mean([norm2(s-delta) for s in sketches]))
    expected = ((dimension-head)/probes-1)*norm2(residual)
    # 같은 projector를 연속 두 지점에서 쓰면 P²의 평균이 I에서 벗어난다.
    vec = delta[0,:4]
    projectors = [np.diag(np.eye(4)[i]*4) for i in range(4)]
    reused = np.mean([p@p@vec for p in projectors],axis=0)
    fresh = np.mean([p@q@vec for p in projectors for q in projectors],axis=0)
    # 시간마다 바뀌는 Jacobian과 adjoint error feedback의 이산 부분 적분을 검산한다.
    weight = trace.value(record["weight"])
    j0,j1 = weight[:4,:4], weight[4:8,:4]
    d0,d1 = delta[0,:4],delta[1,:4]
    mask = np.array([1.,1.,0.,0.])
    e0=np.zeros(4); y0=(d0+e0)*mask; e1=d0+e0-y0
    y1=(d1+e1)*mask; e2=d1+e1-y1
    actual = j0@y0+j1@y1-(j0@d0+j1@d1)
    derived = (j1-j0)@e1-j1@e2
    q=vt[:head].T
    h=delta@q
    tail=delta-h@q.T
    hybrid=[]
    for chosen in itertools.combinations(range(4),2):
        index=np.concatenate([np.arange(i*2,i*2+2) for i in chosen])
        estimate=h@q.T
        estimate[:,index]+=2*tail[:,index]
        hybrid.append(estimate)
    hybrid_mean=np.mean(hybrid,axis=0)
    hybrid_mse=float(np.mean([norm2(v-delta) for v in hybrid]))
    local_x=trace.value(record["input"])[:8,:8]
    local_w=weight[:8,:8]
    index=np.array([0,1,4,5])
    z_head=local_x.T@h
    z_tail=local_x.T@(2*tail[:,index])
    gradient=z_head@q.T
    gradient[:,index]+=z_tail
    cache_updated=local_w@q-z_head-z_tail@q[index]
    cache_exact=(local_w-gradient)@q
    qt=np.linalg.qr(trace.value(record['output'])[:8,:2],mode='reduced')[0]
    a=local_x.T@qt;b=qt.T@delta
    ex=local_x-qt@a.T
    parameter_samples=[]
    for chosen in itertools.combinations(range(4),2):
        selected=np.concatenate([np.arange(i*2,i*2+2) for i in chosen])
        sample=a@b
        sample[selected]+=2*ex[:,selected].T@delta
        parameter_samples.append(sample)
    gw=local_x.T@delta
    parameter_mse=float(np.mean([norm2(v-gw) for v in parameter_samples]))
    expected_parameter_mse=norm2(ex.T@delta)
    results={"protected_residual_enumerated_mean_relative_squared_error":ratio(norm2(mean-delta),norm2(delta)),
        "protected_residual_enumerated_mse":mse,"protected_residual_derived_mse":expected,
        "protected_residual_mse_relative_error":ratio(abs(mse-expected),expected),
        "fresh_projectors_mean_relative_squared_error":ratio(norm2(fresh-vec),norm2(vec)),
        "reused_projector_mean_relative_squared_bias":ratio(norm2(reused-vec),norm2(vec)),
        "adjoint_error_feedback_jacobian_drift_identity_max_abs_error":float(np.max(np.abs(actual-derived))),
        "hybrid_coordinate_tail_mean_relative_squared_error":ratio(norm2(hybrid_mean-delta),norm2(delta)),
        "hybrid_coordinate_tail_mse_relative_error":ratio(abs(hybrid_mse-norm2(tail)),norm2(tail)),
        "cached_weight_projection_update_max_abs_error":float(np.max(np.abs(cache_updated-cache_exact))),
        "forward_basis_parameter_vjp_mean_relative_squared_error":ratio(norm2(np.mean(parameter_samples,axis=0)-gw),norm2(gw)),
        "forward_basis_parameter_vjp_mse_relative_error":ratio(abs(parameter_mse-expected_parameter_mse),expected_parameter_mse),
        "rank_one_before_hadamard":int(np.linalg.matrix_rank(np.ones((8,8)))),
        "rank_after_swiglu_separable_derivative":int(np.linalg.matrix_rank(np.ones((8,8))*np.eye(8)))}
    operation_budget=(len(sketches)+len(parameter_samples)+1)*delta.size*dimension**2
    roundoff=operation_budget*np.finfo(np.float64).eps/(1-operation_budget*np.finfo(np.float64).eps)
    results['identity_roundoff_budget']=roundoff
    assert results["protected_residual_mse_relative_error"] <= roundoff
    assert results["protected_residual_enumerated_mean_relative_squared_error"] <= roundoff**2
    assert results["fresh_projectors_mean_relative_squared_error"] <= roundoff**2
    assert results["forward_basis_parameter_vjp_mean_relative_squared_error"] <= roundoff**2
    assert results["forward_basis_parameter_vjp_mse_relative_error"] <= roundoff
    return results


def analyse_hybrid(reference, target, ranks, fractions, block, tile, reuse, fresh_rank=0, power_steps=0):
    """다른 관측에서 고정한 feature 기저와 현재 정확한 residual column의 혼합 VJP."""
    reference_deltas = {}
    for op in reference.linear:
        reference_deltas.setdefault(op["parameter"],[]).append(reference.read(op["delta"]))
    bases = {name:np.linalg.svd(np.concatenate(values,axis=0),full_matrices=False)[2].T
             for name,values in reference_deltas.items()}
    records=[]
    reference_weights={op["parameter"]:reference.value(op["weight"]) for op in reference.linear}
    rng=np.random.default_rng(reference.meta['model']['seed'])
    for op in target.linear:
        name=op["parameter"]
        x,w,delta=target.value(op["input"]),target.value(op["weight"]),target.read(op["delta"])
        if name in reference_weights and not np.array_equal(reference_weights[name],w):
            raise ValueError(f"hybrid cross-input comparison requires same weight tensor: {name}")
        n,a=x.shape;b=w.shape[1]
        dx,dw=delta@w.T,x.T@delta
        baseline=gemm(n,a,b,tile)+gemm(a,b,n,tile)
        forward=gemm(n,b,a,tile)*(2 if name.startswith("layers.") else 1)
        count=math.ceil(b/block)
        for requested in ranks:
            base_rank=min(requested,bases[name].shape[1]) if name in bases else 0
            q_base=bases[name][:,:base_rank] if name in bases else np.zeros((b,0))
            fresh=min(fresh_rank,b-base_rank) if name in bases else 0
            basis_gemm=0
            qr_logical=0
            if fresh:
                z=delta.T@rng.standard_normal((n,fresh))
                basis_gemm=gemm(b,fresh,n,tile)
                for iteration in range(power_steps+1):
                    for _ in range(2):
                        z-=q_base@(q_base.T@z)
                    z=np.linalg.qr(z,mode='reduced')[0]
                    basis_gemm+=2*(gemm(base_rank,fresh,b,tile)+gemm(b,fresh,base_rank,tile))
                    qr_logical+=4*b*fresh*fresh
                    if iteration<power_steps:
                        z=delta.T@(delta@z)
                        basis_gemm+=gemm(n,fresh,b,tile)+gemm(b,fresh,n,tile)
                q=np.linalg.qr(np.concatenate([q_base,z],axis=1),mode='reduced')[0]
                qr_logical+=4*b*(base_rank+fresh)**2
            else:
                q=q_base
            r=q.shape[1]
            coefficient=delta@q
            residual=delta-coefficient@q.T
            residual_dw=x.T@residual
            c=[residual[:,i:i+block]@w[:,i:i+block].T for i in range(0,b,block)]
            for fraction in fractions:
                selected=min(count,max(1,math.ceil(count*fraction)))
                p=selected/count
                k=min(b,selected*block)
                cost=(gemm(n,r,b,tile)+gemm(n,k,r,tile)+gemm(a,base_rank,b,tile)/reuse
                      +gemm(a,fresh,b,tile)+basis_gemm+qr_logical
                      +gemm(n,a,r,tile)+gemm(n,a,k,tile)+gemm(a,r,n,tile)
                      +gemm(a,b,r,tile)+gemm(a,k,n,tile))
                vdx=(count-selected)/(selected*(count-1))*(count*sum(norm2(v) for v in c)-norm2(sum(c))) if count>1 else 0.0
                active=name in bases and cost<baseline and r<b and selected<count
                records.append({"parameter":name,"output_variable":op["output"],"stage":stage(name),"shape":[n,a,b],
                    "requested_rank":requested,"rank":r,"requested_fraction":fraction,"selected_blocks":selected,
                    "base_rank":base_rank,"fresh_rank":fresh,"basis_gemm_issued_flops":basis_gemm,
                    "basis_qr_logical_budget":qr_logical,
                    "reference_basis_available":name in bases,
                    "blocks":count,"probability":p,"maximum_selected_columns":k,
                    "residual_fraction":ratio(norm2(residual),norm2(delta)),
                    "weight_gradient_tail_fraction":ratio(norm2(residual_dw),norm2(dw)),
                    "weight_gradient_relative_mse":(1/p-1)*ratio(norm2(residual_dw),norm2(dw)),
                    "dx_relative_mse":ratio(max(0,vdx),norm2(dx)),
                    "weight_gradient_energy":norm2(dw),"input_gradient_energy":norm2(dx),
                    "expected_weight_gradient_error_squared":(1/p-1)*norm2(residual_dw),
                    "baseline_issued_backward_flops":baseline,"forward_and_checkpoint_issued_flops":forward,
                    "modelled_backward_cost_including_basis_budget":cost,"shape_selects_hybrid":active})
    groups={}
    for group in sorted({r["stage"] for r in records}|{"all"}):
        entries=[r for r in records if group=="all" or r["stage"]==group]
        groups[group]=[]
        for rank in ranks:
            for fraction in fractions:
                chosen=[r for r in entries if r["requested_rank"]==rank and r["requested_fraction"]==fraction]
                den=sum(r["baseline_issued_backward_flops"] for r in chosen)
                fwd=sum(r["forward_and_checkpoint_issued_flops"] for r in chosen)
                actual=sum(r["modelled_backward_cost_including_basis_budget"] if r["shape_selects_hybrid"] else r["baseline_issued_backward_flops"] for r in chosen)
                groups[group].append({"requested_rank":rank,"requested_fraction":fraction,"operations":len(chosen),
                    "shape_selects_hybrid":sum(r["shape_selects_hybrid"] for r in chosen),
                    "modelled_backward_cost_ratio_with_dense_fallback":ratio(actual,den),
                    "modelled_linear_total_cost_ratio_including_checkpoint":ratio(actual+fwd,den+fwd),
                    "median_residual_fraction":float(np.median([r["residual_fraction"] for r in chosen])),
                    "sum_operator_dw_relative_mse_with_dense_fallback":ratio(sum(r["expected_weight_gradient_error_squared"] for r in chosen if r["shape_selects_hybrid"]),sum(r["weight_gradient_energy"] for r in chosen))})
    return {"reference_trace":str(reference.path),"target_trace":str(target.path),"reference_tensor_sha256":reference.tensor_sha256,
        "target_tensor_sha256":target.tensor_sha256,"weight_projection_reuses":reuse,"groups":groups,"records":records,
        "matched_weight_tensors_equal":True,
        "fresh_rank_budget":fresh_rank,"power_steps":power_steps,
        "scope":"feature bases from a different input at the same weights with optional current-input residual range finding; local VJP variance; fresh-basis GEMMs plus QR arithmetic budget counted, scatter/nonlinear wall time requires measured kernels"}


def validate_replay(path, trace):
    path=Path(path)
    report=json.loads((path/'replay.json').read_text())
    gate,up,down=[trace.operations[report['prefix']+'.'+suffix] for suffix in ['gate','up','down']]
    x,wg,wu,wd=trace.value(gate['input']),trace.value(gate['weight']),trace.value(up['weight']),trace.value(down['weight'])
    g,u,h,dy=trace.value(gate['output']),trace.value(up['output']),trace.value(down['input']),trace.read(down['delta'])
    sigmoid=1/(1+np.exp(-g))
    cases=[]
    for mode in ['exact','sampled_dense_with_packing']:
        selected=np.arange(wg.shape[1]) if mode=='exact' else np.array(report['selected_channels'])
        p=1.0 if mode=='exact' else report['sampling_probability']
        dh=(dy@wd[selected].T)/p
        dg=dh*u[:,selected]*sigmoid[:,selected]*(1+g[:,selected]*(1-sigmoid[:,selected]))
        du=dh*g[:,selected]*sigmoid[:,selected]
        dx=dg@wg[:,selected].T+du@wu[:,selected].T
        dwg=np.zeros_like(wg);dwu=np.zeros_like(wu);dwd=np.zeros_like(wd)
        dwg[:,selected]=x.T@dg;dwu[:,selected]=x.T@du;dwd[selected]=h[:,selected].T@dy/p
        errors={}
        for name,expected in zip(['dx','dwg','dwu','dwd'],[dx,dwg,dwu,dwd]):
            measured=np.fromfile(path/f'{mode}-{name}.f32',dtype='<f4').reshape(expected.shape)
            if not np.all(np.isfinite(measured)):
                raise ValueError(f'nonfinite GPU replay {path} {name}')
            errors[name]={'max_abs_error':float(np.max(np.abs(measured-expected))),
                'relative_squared_error':ratio(norm2(measured-expected),norm2(expected))}
        cases.append({'mode':mode,'errors':errors})
    return {'path':str(path),'input_trace':str(trace.path),'input_tensor_sha256':trace.tensor_sha256,
        'measurement':report,'float64_reference_comparisons':cases}


def analyse_forward_basis(trace, ranks, fractions, block, tile):
    """순전파 Y의 일부 열로 현재 X의 token 기저를 만들고 parameter VJP 잔차만 표집한다."""
    inputs={}
    for op in trace.linear:
        inputs.setdefault(op['input'],[]).append(op)
    rng=np.random.default_rng(trace.meta['model']['seed'])
    records=[]
    for operations in inputs.values():
        donor=max(operations,key=lambda op:trace.meta['tensors'][op['output']]['shape'][1])
        x,y=trace.value(donor['input']),trace.value(donor['output'])
        n,a=x.shape;uses=len(operations)
        for requested in ranks:
            r=min(requested,*y.shape)
            columns=rng.choice(y.shape[1],r,replace=False)
            q=np.linalg.qr(y[:,columns],mode='reduced')[0]
            xtq=x.T@q
            residual=x-q@xtq.T
            for op in operations:
                delta=trace.read(op['delta']);b=delta.shape[1];name=op['parameter']
                dw=x.T@delta
                remainder=residual.T@delta
                baseline_dw=gemm(a,b,n,tile)
                baseline_dx=gemm(n,a,b,tile)
                forward=gemm(n,b,a,tile)*(2 if name.startswith('layers.') else 1)
                count=math.ceil(a/block)
                for fraction in fractions:
                    selected=min(count,max(1,math.ceil(count*fraction)));p=selected/count;k=min(a,selected*block)
                    parameter_cost=(gemm(a,r,n,tile)/uses+gemm(r,b,n,tile)+gemm(a,b,r,tile)
                                    +gemm(n,k,r,tile)/uses+gemm(k,b,n,tile)+4*n*r*r/uses)
                    active=parameter_cost<baseline_dw and selected<count
                    variance=(1/p-1)*norm2(remainder)
                    records.append({'parameter':name,'output_variable':op['output'],'stage':stage(name),
                        'shape':[n,a,b],'requested_rank':requested,'rank':r,'fraction':fraction,
                        'basis_donor':donor['parameter'],'basis_donor_output':donor['output'],
                        'donor_columns':columns.tolist(),'shared_input_uses':uses,
                        'input_tail_fraction':ratio(norm2(residual),norm2(x)),
                        'parameter_gradient_tail_fraction':ratio(norm2(remainder),norm2(dw)),
                        'expected_parameter_gradient_relative_mse':ratio(variance,norm2(dw)),
                        'expected_parameter_gradient_error_squared':variance,'parameter_gradient_energy':norm2(dw),
                        'baseline_issued_parameter_flops':baseline_dw,'exact_issued_input_flops':baseline_dx,
                        'parameter_gemm_and_qr_cost':parameter_cost,'forward_and_checkpoint_issued_flops':forward,
                        'shape_selects_compression':active})
    groups={}
    for group in sorted({r['stage'] for r in records}|{'all'}):
        groups[group]=[]
        for requested in ranks:
            for fraction in fractions:
                chosen=[r for r in records if (group=='all' or r['stage']==group) and r['requested_rank']==requested and r['fraction']==fraction]
                original=sum(r['baseline_issued_parameter_flops'] for r in chosen)
                exact_dx=sum(r['exact_issued_input_flops'] for r in chosen)
                actual=sum(r['parameter_gemm_and_qr_cost'] if r['shape_selects_compression'] else r['baseline_issued_parameter_flops'] for r in chosen)
                forward=sum(r['forward_and_checkpoint_issued_flops'] for r in chosen)
                groups[group].append({'requested_rank':requested,'fraction':fraction,'operations':len(chosen),
                    'shape_selects_compression':sum(r['shape_selects_compression'] for r in chosen),
                    'parameter_vjp_cost_ratio':ratio(actual,original),
                    'backward_cost_ratio_with_exact_dx':ratio(actual+exact_dx,original+exact_dx),
                    'linear_total_cost_ratio_including_checkpoint':ratio(actual+exact_dx+forward,original+exact_dx+forward),
                    'median_parameter_gradient_tail_fraction':float(np.median([r['parameter_gradient_tail_fraction'] for r in chosen])),
                    'sum_operator_parameter_gradient_relative_mse':ratio(sum(r['expected_parameter_gradient_error_squared'] for r in chosen if r['shape_selects_compression']),sum(r['parameter_gradient_energy'] for r in chosen))})
    return {'trace_path':str(trace.path),'tensor_sha256':trace.tensor_sha256,'groups':groups,'records':records,
        'scope':'current exact forward output supplies token basis; shared-input projections amortized; exact dX; local dW variance; GEMM tile and two-pass QR arithmetic budget, packing and scatter require measured kernels'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace",action="append",required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--ranks",required=True,help="explicit diagnostic rank budgets")
    parser.add_argument("--energies",required=True,help="explicit retained spectral-energy targets")
    parser.add_argument("--fractions",required=True,help="explicit backward block fractions")
    parser.add_argument("--block",type=int,required=True,help="hidden-channel block matching GEMM reduction/output tiles")
    parser.add_argument("--tile",required=True,help="BM,BN,BK from Engine::matmul")
    parser.add_argument("--hybrid-reference",help="frozen-basis trace for cross-input hybrid VJP")
    parser.add_argument("--hybrid-target",help="different input trace at same weights")
    parser.add_argument("--projection-reuses",type=int,required=True,help="microbatches sharing fixed weights and basis")
    parser.add_argument("--replay",action='append',nargs=2,default=[],metavar=('REPLAY','TRACE'),help="GPU replay directory and its input trace")
    parser.add_argument("--fresh-rank",type=int,required=True,help="explicit current-input residual range budget")
    parser.add_argument("--power-steps",type=int,required=True,help="explicit residual subspace iteration budget")
    parser.add_argument("--detail-rank",type=int,required=True,help="rank budget for per-operator hybrid records")
    parser.add_argument("--detail-fraction",type=float,required=True,help="fraction for per-operator hybrid records")
    args=parser.parse_args()
    ranks=[int(x) for x in args.ranks.split(',')]
    levels=[float(x) for x in args.energies.split(',')]
    fractions=[float(x) for x in args.fractions.split(',')]
    tile=tuple(int(x) for x in args.tile.split(','))
    if len(tile)!=3 or min(tile)<=0 or not ranks or min(ranks)<=0 or args.block<=0 or any(not 0<x<=1 for x in levels+fractions):
        parser.error("positive dimensions and energy/fraction values in (0,1] required")
    if args.projection_reuses<=0 or min(args.fresh_rank,args.power_steps)<0 or bool(args.hybrid_reference)!=bool(args.hybrid_target):
        parser.error("positive projection reuses and both hybrid trace paths required")
    if args.detail_rank not in ranks or args.detail_fraction not in fractions:
        parser.error('detail rank and fraction must appear in diagnostic budgets')
    results=[];traces=[]
    for path in args.trace:
        trace=Trace(path)
        traces.append(trace)
        linear,groups=analyse_linear(trace,ranks,levels,tile)
        ffn=analyse_ffn(trace,args.block,fractions,levels,tile)
        results.append({"trace_path":path,"trace_metadata":{k:v for k,v in trace.meta.items() if k not in ["tensors","linear_operations"]},
            "trace_sha256":hashlib.sha256((trace.path/'trace.json').read_bytes()).hexdigest(),"tensor_sha256":trace.tensor_sha256,
            "linear":linear,"groups":groups,"ffn":ffn,"identities":identities(trace)})
        print(json.dumps({"trace":path,"groups":groups,"ffn_operations":len(ffn),"identities":results[-1]["identities"]}))
    output={"format":"sejong-adjoint-research-v1","numpy":np.__version__,
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "arguments":vars(args)|{"out":str(args.out)},"runs":results,
        "cost_scope":"known-basis GEMM arithmetic and tile padding; basis construction, nonlinear and scheduling costs require measured kernels",
        "spectrum_scope":"current exact cotangents, independently truncated per operator; full-network compressed-gradient errors require a composed backward run"}
    if args.hybrid_reference:
        reference,target=Trace(args.hybrid_reference),Trace(args.hybrid_target)
        output["hybrid"]=analyse_hybrid(reference,target,ranks,fractions,args.block,tile,args.projection_reuses)
        if args.fresh_rank:
            output['hybrid_adaptive']=analyse_hybrid(reference,target,ranks,fractions,args.block,tile,args.projection_reuses,args.fresh_rank,args.power_steps)
        output['forward_basis_parameter_vjp']=analyse_forward_basis(target,ranks,fractions,args.block,tile)
    for key in ['hybrid','hybrid_adaptive','forward_basis_parameter_vjp']:
        if key in output:
            output[key]['records']=[v for v in output[key]['records'] if v['requested_rank']==args.detail_rank
                and v.get('requested_fraction',v.get('fraction'))==args.detail_fraction]
            output[key]['record_detail_budget']={'rank':args.detail_rank,'fraction':args.detail_fraction}
    by_path={trace.path.resolve():trace for trace in traces}
    output['gpu_replays']=[validate_replay(path,by_path[Path(trace).resolve()]) for path,trace in args.replay]
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(output,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


if __name__ == "__main__":
    main()
