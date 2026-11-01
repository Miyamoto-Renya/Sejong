"""구조별 주요 행렬 연산량과 조건부 시간을 계산한다. Python 표준 라이브러리 사용."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
run = json.loads((ROOT / 'docs/measurements/2026-11-01-architecture-fit/applied.json').read_text())['runs'][0]
rate = 74727358464 / run['training_step_seconds']['mean']
probe = json.loads((HERE / 'kernel-probe.json').read_text())

# 계산 예시: global 13층, GQA 16/8, E8/top2, patch8, local 2+2층.
d, l, layers, experts, topk, f, patch, seq, values = 1024, 256, 13, 8, 2, 768, 8, 1024, 4096
local_matrix = 4 * (4*l*l + 3*l*768)
global_shared = layers * (3*d*d + d*experts)
global_experts = layers * 3*d*f*experts
params = (local_matrix + 4*2*l + global_shared + layers*2*d + global_experts
          + 4*d*l + l*l + 3*d + 2*l + 259*d + (d+1)*values)
global_active = global_shared + layers*3*d*f*topk
linear_per_byte = local_matrix + 2*d*l + l*l + (global_active+2*d*l)/patch
def pairs(n, window):
    w = min(n, window)
    return w*(w+1)//2 + (n-w)*w
local_pairs = pairs(seq, 128)
global_pairs = pairs(seq//patch, seq//patch)
attention = 18*(4*l*local_pairs + layers*d*global_pairs)/seq
# NativeValue Graph::new(training=true): global checkpoint가 순전파를 재계산한다.
# 이 계산 예시의 local checkpoint는 둘 다 false.
recompute = 2*global_active/patch + 4*layers*d*global_pairs/seq
head_and_bos = 6*(d*values+d*l)/seq
per_byte = 6*linear_per_byte + attention + recompute + head_and_bos
scenarios = []
for ratio in (2,4,6):
    flops = 10*8e9*ratio*per_byte
    days = flops/rate/86400
    scenarios.append(dict(bytes_per_token=ratio, total_flops=flops,
                          days_per_original=days/10, days_total=days,
                          energy_kwh_at_200W=days*24*.2,
                          electricity_KRW_at_200_per_kwh=days*24*.2*200))

# Memory: body와 K/V 원본 파라미터. readout 및 affine port는 별도 집계.
md, ml, mf, mh, mt = 1024, 6, 4096, 109088, 128
body = ml*(4*md*md+2*md*mf+8*md+mf)+md*md
kv = 2*md*mh
forward = 2*mt*ml*(4*md*md+2*md*mf)+4*ml*mt*mt*md+2*md*md
fact = forward+5*kv
query = 9*forward  # 세 query view 각각 forward+backward 주요 행렬 연산 근사.
port = (md+1)*md
bw = probe['pageable_H2D']['steady_GB_s']*1e9
direct_ms = sum(k['median_ms'] for k in probe['kernels'] if k['label'].startswith('direct-'))
workspace = (2*(2+12*layers)*seq*d*4+6*layers*seq*3072*4
             +2*layers*seq*16*4+4*d*4+4*values*4+6*4+seq*4)
result = dict(
    assumptions=dict(originals=10, ordinary_tokens_per_original=8e9,
                     sequence_bytes_including_special_positions=seq, values=values,
                     throughput_reference='26.49M byte-SGD benchmark; conditional extrapolation',
                     native_gradient='dense leading matrix FLOPs; global recomputation included',
                     excluded_work='pointwise, optimizer, CPU preparation, serialization, port fitting; throughput reference includes its own runtime overhead'),
    measured_reference_flops_per_second=rate,
    native_value=dict(parameters=params, total_10_parameters=10*params,
                      linear_parameter_uses_per_byte=linear_per_byte,
                      training_flops_per_byte=per_byte, scenarios=scenarios,
                      planner_fixed_approx_bytes=26*params,
                      planner_workspace_per_example_bytes=workspace,
                      r4_days_at_TF={str(r):10*8e9*4*per_byte/(r*1e12)/86400 for r in (.4, .67, 1.)}),
    memory=dict(body_parameters=body, kv_parameters=kv, original_parameters=body+kv,
                port_parameters=port, readout_parameters='(dimension+1)*value_count',
                context_positions=mt, forward_flops_per_request=forward,
                fact_flops_per_visit=fact, query_flops_per_base_visit_three_views=query,
                fact_hours_per_million_visits=fact*1e6/rate/3600,
                query_hours_per_million_visits=query*1e6/rate/3600,
                direct_KV_kernel_ms=direct_ms,
                all_ten_original_fp64_bytes=10*(body+kv)*8,
                planned_ten_KV_port_transfer_seconds=10*(kv+port)*8/bw))
if __name__ == '__main__':
    print(json.dumps(result, indent=2, ensure_ascii=False))
