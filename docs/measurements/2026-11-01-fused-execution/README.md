# Attention·RMSNorm·projection·HIP graph 실험

2026-11-01. RX 6600 gfx1032, wave32, Triton 3.8.0+git669b31ac.rocm10.1.0, 직접 HIP 실행. [연구와 실행 계획](../../research/attention-projection-execution-plan.md)은 기존 계층 본체·역전파·parameter VJP에 연결할 설계를 담는다. 이 측정은 독립 연산과 dispatch chain의 범위다. [source SHA256](source-sha256.json), 기존 커널 snapshot과 재현 코드를 함께 보관했다.

GPU 실험은 순차로 실행했다. 고정 seed로 생성한 operand와 실제 본체의 형상을 사용했다. HIP event는 커널 묶음 앞뒤에서 기록했고, host 제출 사이의 idle을 포함한다. Compile·allocation·입출력 전송은 시간 구간 밖에서 수행했다. 전체 결과는 후보별 표본·최솟값·중앙값·출력 대조·컴파일 자원·binary fingerprint를 담는다.

| 연산·형상 | 기존→후보 중앙값 µs | 변화·출력 |
| --- | ---: | --- |
| RMSNorm global R64,D512 | 16.207→6.932 | 두 launch 유지, dx bitwise 일치 |
| RMSNorm local R512,D128 | 167.478→7.682 | 두 launch 유지, dx bitwise 일치 |
| RMSNorm final R512,D512 | 144.743→11.300 | 두 launch 유지, dx bitwise 일치 |
| Attention local B1,T512,H2/2,D64,W128 | 233.066→223.449 | 두 launch 유지, dQ/dK/dV bitwise 일치 |
| Attention global B1,T64,H8/4,D64 | 129.016→123.966 | 두 launch 유지, dQ/dK/dV bitwise 일치 |
| Gate/up+SwiGLU M512,D128,F384 | 85.894→65.805 | 3→1 launch, G/U/H bitwise 일치 |
| Gate/up+SwiGLU M32,D512,F384 | 64.992→35.658 | 3→1 launch, G/U/H bitwise 일치 |
| Gate/up+SwiGLU M47,D512,F384 | 66.847→41.550 | 3→1 launch, G/U/H bitwise 일치 |
| QKV M512,D128,KV128 | 157.881→73.026 | 3→1 launch, Q/K/V bitwise 일치 |
| QKV M64,D512,KV256 | 124.748→58.017 | 3→1 launch, Q/K/V bitwise 일치 |
| Gate/up dX M512,D128,F384 | 237.537→231.997 | 3→1 launch, dX bitwise 일치 |
| Gate/up dX M32,D512,F384 | 115.095→113.674 | 3→1 launch, dX bitwise 일치 |

RMSNorm의 실제 baseline ISA는 [CAS retry loop](attention-rms/rms-base-128.amdgcn#L170)를 사용한다. Partial과 final은 atomic 없이 실행한다. BR4·8·16을 모두 비교했고 선택된 BR은 global4·local8·final8이었다. 이 값은 실험의 선택 결과다. 큰 R·비정렬 R129,D257과 attention의 B2,T33,GQA4/2,D32,W17도 실행했다. RMS dw는 FP64 수식 대비 상대 L2 약8–9×10⁻⁸, 비정렬 attention dK의 기존 대비 최대 절대차는 2.98×10⁻⁸이었다.

QKV local의 기존 최소/최대는 123.761/160.525µs, 결합 BN32는 71.283/73.771µs였다. GQA의 기존 최소/최대는 120.867/132.076µs, 결합 BN32는 57.305/58.690µs였다. QKV 최종 측정은 256회 예열, batch256, ABCCBA8회이며 BN16도 함께 비교했다. Gate/up forward·dX는 24회 예열, batch128, ABBA6회다. 기존 FP32 IEEE dot를 유지하고 순전파는 FP16 중간 반올림·출력, dX는 FP32 출력으로 fusion 효과를 측정했다. dX는 VGPR127→249, LDS6→8KiB가 늘며 작은 시간 이득을 기록했다.

| Add chain의 원소·nodes | CPU enqueue direct→replay ms | 동기 완료 direct→replay ms | capture+instantiate ms |
| --- | ---: | ---: | ---: |
| 256 × 128 | 0.241→0.00425 | 0.411→0.333 | 0.677 |
| 256 × 1,024 | 1.889→0.01214 | 3.042→2.499 | 3.649 |
| 256 × 4,096 | 7.510→0.03783 | 12.274→10.017 | 15.541 |
| 256 × 14,260 | 27.944→0.11632 | 43.194→36.120 | 82.063 |
| 65,536 × 4,096 | 7.457→0.02861 | 14.185→11.508 | 13.765 |

HIP graph는 실제 Sejong FP32 add가 같은 buffer에 ones를 누적하는 chain이다. 결과는 모든 원소에서 node 수와 같았다. 128-node graph의 node 하나를 끄면127, 다시 켜고 alpha를2로 바꾸면129였다. 직접 실행과 replay를 예열하고 ABBA2회 측정했다. 14,260 nodes는 현재 학습의 대략적인 launch 수에 맞춘 dispatch 실험이다.

| 원본 | 내용 |
| --- | --- |
| [요약](summary.json) | 실제 norm 호출 수와 독립 연산 시간 |
| [attention·RMS](attention-rms/result.json), [final norm](attention-rms/final-result.json) | 모든 후보·표본·FP64 대조·컴파일 자원 |
| [작업량·ISA](attention-rms/derived.json), [provenance](attention-rms/provenance.json) | atomic·Delta 연산·논리적 traffic·source SHA |
| [gate/up](projection/microbench.json), [dX](projection/dx-microbench.json) | 전체 출력·CPU 표본·시간·HSACO SHA |
| [QKV](projection/qkv-microbench.json), [초기 관측](projection/qkv-microbench-initial.json) | BN32/16과 GQA·균등 폭의 시간·출력 |
| [projection 작업량](projection/derived-counts.json), [snapshot 정보](projection/artifact-manifest.json) | family 호출·launch·유효 byte 계산 |
| [HIP graph](arena-graph/result.json) | capture·instantiate·replay·node 갱신·runtime |

아래 명령은 각 probe를 개별 실행한다. Compile 명령은 같은 폴더에 HSACO·IR·ISA를 생성하고 뒤의 측정 명령이 이를 읽는다. Baseline은 이 폴더에 보관한 기존 커널 snapshot이다.

```bash
source scripts/env.sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python docs/measurements/2026-11-01-fused-execution/attention-rms/probe.py
SEJONG_PROBE_FINAL_ONLY=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python docs/measurements/2026-11-01-fused-execution/attention-rms/probe.py
python docs/measurements/2026-11-01-fused-execution/projection/compile_probe.py
python docs/measurements/2026-11-01-fused-execution/projection/hip_microbench.py
python docs/measurements/2026-11-01-fused-execution/projection/dx_compile.py
python docs/measurements/2026-11-01-fused-execution/projection/dx_microbench.py
python docs/measurements/2026-11-01-fused-execution/projection/qkv_conditional_compile.py
python docs/measurements/2026-11-01-fused-execution/projection/qkv_microbench.py
python docs/measurements/2026-11-01-fused-execution/arena-graph/graph_probe.py
```
