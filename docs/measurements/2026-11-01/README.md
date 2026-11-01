# RX 6600의 Sejong 학습·추론 측정

2026-11-01. Rust 학습 루프와 Triton 커널을 AMD Radeon RX 6600에서 실행했다. 기본 계층·GQA·전문가 구성은 같은 4,096-byte 정확 미분 step에서 dense 비교군보다 주요 FLOPs가 **9.16배**, step 시간이 **4.66배** 줄었다. 기본 설정은 이 구조와 정확 SGD를 사용한다. 공유 encoder 교정 SGD도 학습·저장·재개·추론까지 구현했으며 `configs/sejong-control-variate.toml`에서 선택한다.

실행 환경은 Python 3.14.4, Triton `3.8.0+git669b31ac.rocm10.1.0`, ROCm SDK 10.1.0, Rust release 빌드다. Python 환경의 `torch_installed` 값은 false다. Rust가 HIP tensor 할당·커널 launch·수동 역전파·optimizer를 담당한다.

원본 결과는 [구조·크기 비교 JSON](rx6600-compute.json), [66-step 학습 비교 JSON](rx6600-learning-window.json), [교정 학습 계약 결과](training-contract.json)에 있다. JSON은 설정 전체, 입력·소스·실행 파일 BLAKE3, 측정 step의 시간·FLOPs·학습 손실과 초기·최종 검증 손실을 보존한다. 측정 당시 CV 경로는 `configs/sejong.toml`이었다. 측정 후 기본값을 정확 SGD로 바꾸고 같은 CV 값을 `configs/sejong-control-variate.toml`로 옮겼다. 결과의 내장 `config`가 당시 실행 설정이다.

학습은 저장한 [train.txt](train.txt) 16,939 bytes, 검증은 별도 문서에서 저장한 [validation.txt](validation.txt) 7,633 bytes를 사용했다. 학습 snapshot은 README·native-data·gpu-boundary, 검증 snapshot은 architecture 문서다. 모든 구성의 모델 seed는 42, 자료 seed는 123이다. 학습 sequence는 512, microbatch는 1이다. 평가 값은 byte+문서 EOS 어휘의 평균 교차 엔트로피를 ln(2)로 나눈 bits-per-byte-token(BPB)이다.

FLOPs는 실행한 GEMM·attention 내적·기저 좌표 이동의 실제 shape를 FMA=2로 집계한 값이다. 아래 표는 타일 패딩을 포함한 issued FLOPs를 사용한다. JSON의 logical FLOPs는 유효 shape 기준이다. GPU 동기화로 잰 wall time에는 정규화·routing·packing·host count 회수·전송·기울기 norm 등도 포함된다. 메모리는 Sejong allocator의 학습 peak MiB다.

먼저 6회 실제 업데이트 중 초기 2회를 시간 집계에서 제외하고 4회를 측정했다. 측정 4회에서 신규 Triton 컴파일 수는 모두 0이었다. 이때 `--warmup-steps 2`는 시간 집계 경계다. Optimizer는 원래 설정인 100-step 학습률 warmup을 따르며, 이 구간에서 학습률은 step에 따라 증가한다.

| 구성 | 본체 파라미터 | 관측 / 정확 미분 bytes·step | GFLOPs·step | 평균 초·step | peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense + SGD | 27,534,848 | 4,096 / 4,096 | 1,006.532 | 1.238329 | 482 |
| P8 계층 + SGD | 28,568,960 | 4,096 / 4,096 | 146.004 | 0.255837 | 493 |
| P8 + GQA + 전문가 + SGD, 현재 기본값 | 26,488,192 | 4,096 / 4,096 | 109.870 | 0.265828 | 465 |
| 같은 본체 + 교정 SGD | 26,488,192 | 4,096 / 1,024 | 34.191 | 0.102312 | 516 |
| 폭 2배 Dense + SGD | 109,595,648 | 4,096 / 4,096 | 3,799.771 | 4.535824 | 1,879 |
| 폭 2배 계층·GQA·전문가 + 교정 SGD | 105,569,024 | 4,096 / 1,024 | 134.970 | 0.243641 | 2,228 |

P8 계층은 작은 encoder·decoder가 byte열을 처리하고 global 본체를 8-byte 구간마다 실행한다. GQA는 query 8개에 KV 4개를 공유한다. 전문가 구성은 기존 FFN 내부 폭 1,536을 4 × 384로 나누고 위치당 2개를 선택한다. 전문가를 더한 구성은 계층 dense 대비 FLOPs가 24.75% 줄었고 step 시간은 3.90% 늘었다. 작은 전문가 GEMM의 타일·packing·launch 비용이 실제 시간에 드러난다. 기본값 선택은 사용자가 요청한 FLOPs 절감과 활성 용량 분리에 맞췄다.

교정 SGD의 0.102312초는 전체 8개 관측 중 2개의 정확 미분을 실행한 비용이다. Dense 대비 12.10배의 step 시간 감소에는 구조 개선과 정확 미분 표본 수 감소가 함께 들어간다. 이 둘의 학습 효과는 다음 같은 구조 비교로 살폈다.

초기화·6회 학습·최종 모델 저장을 포함한 전체 시간은 위 표 순서로 7.928, 2.099, 2.147, 1.247, 30.141, 7.586초였다. 이 전체 실행의 issued TFLOPs는 6.0392, 0.8760, 0.6590, 0.2054, 22.7986, 0.8090이었다. 평가와 추론은 별도로 실행했다. 확대 모델에서 CV 기저 초기화·저장 비용을 포함한 총시간과 정상 step 시간을 함께 비교할 수 있다.

교정 기저를 두 번 갱신하는 66-step 실행에서는 처음 2회를 제외한 64회의 시간을 집계했다. 같은 초기 모델과 검증 8개 sequence에서 초기 BPB는 8.596439였다. 모든 66회 업데이트가 optimizer의 100-step warmup 구간에 있다.

| 같은 26.49M 본체의 optimizer | 관측 / 정확 미분 bytes·step | GFLOPs·step | 평균 초·step | 66-step BPB | 초기화·학습·저장 초 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 기본 SGD, accumulation 8 | 4,096 / 4,096 | 109.713 | 0.264337 | 7.935042 | 18.021808 |
| SGD, accumulation 2 | 1,024 / 1,024 | 27.469 | 0.071580 | 7.948070 | 5.275493 |
| 교정 SGD, 관측 8 / control 2 | 4,096 / 1,024 | 34.333 | 0.105199 | 7.941578 | 7.616680 |

교정 SGD는 같은 정확 미분 예산의 accumulation-2 SGD보다 평균 step 시간이 1.47배였고 최종 BPB가 0.00649 낮았다. 입력 entropy 층화 선택과 predictor가 함께 작동하는 비교다. Step 32와 64의 관측된 control residual dispersion / gradient dispersion은 각각 0.99999974, 0.99969854였다. 이 값은 선택된 control 집합의 가중 분산 진단이며, 층화 추정량의 분산 계산에는 각 strata 내부의 잔차 분산과 유한 모집단 보정이 필요하다. 현재 진단에서 predictor의 잔차 감소는 작고 부가 계산 비용이 뚜렷하다. 이 측정에 따라 정확 SGD를 기본값으로 정했다. 교정 경로는 같은 실행기로 더 긴 학습 구간과 다양한 자료를 비교할 수 있다.

Step 32와 64의 교정 학습 시간은 각각 0.180753초, 0.180015초였으며, 이 중 gradient 진단은 0.034363초와 0.034189초였다. 표의 평균은 두 번의 기저 갱신과 진단을 포함한다. 66회 전체 학습·초기화·저장의 issued TFLOPs는 SGD accumulation-8이 7.2411, SGD accumulation-2가 1.8128, 교정 SGD가 2.2660이었다. 이 실험은 단일 seed와 저장한 소규모 자료에서 초기 학습 구간의 비용·손실을 기록한다. 목표 언어 품질까지의 비용은 목표 BPB에 도달하는 누적 학습 곡선으로 평가한다.

추론은 저장한 모델을 다시 읽고 동일 검증 입력의 496-token prefill과 16-token teacher-forced decode를 실행했다. Cache 경로를 한 번 준비한 뒤 두 번째 실행을 동기화해 측정했다.

| 구성 | 496-token prefill 초 | decode byte tokens/s | cached/full 마지막 logit 최대 오차 |
| --- | ---: | ---: | ---: |
| Dense + SGD | 0.025534 | 407.20 | 0 |
| P8 계층 + SGD | 0.005224 | 1,356.97 | 0 |
| 현재 기본 계층·GQA·전문가 + SGD | 0.006490 | 1,225.34 | 0 |
| 같은 본체 + 교정 SGD | 0.006303 | 1,215.43 | 0 |
| 폭 2배 Dense + SGD | 0.092241 | 204.79 | 0 |
| 폭 2배 계층·GQA·전문가 + 교정 SGD | 0.015245 | 793.72 | 0 |

초기 prefill은 encoder·global·decoder를 전체 배치로 실행하며 KV를 채운다. 이후 cache는 남은 patch 경계까지 처리하고 완성된 patch마다 global을 갱신한다. 첫 구현의 patch별 순차 prefill을 이 경로로 바꾼 뒤 교정 구성의 496-token prefill은 0.194초에서 0.0063초로 줄었다.

GPU 계약 검증에서 GQA/window attention의 forward·backward 최대 절대 오차는 Rust 참조식 대비 1.04×10⁻⁷, expert/router는 2.38×10⁻⁷였다. 계층 모델의 F32 cache 오차는 1.43×10⁻⁶, directional gradient 오차는 약 1.68×10⁻⁷였다. F16 계층 모델의 67개 gradient tensor는 모두 유한했고 cache 오차는 0이었다. Partial patch prefill·여러 배치·인과성·NativeValue 2-step 학습 후 봉인·저장·재읽기를 실행했다. 교정 학습의 4-step 계약에서는 기저 refresh·checkpoint·재개를 거쳤으며, 재개된 가중치·optimizer 오차와 모든 관측을 control로 둔 정확 SGD 대비 오차는 모두 0이었다.

재현 명령은 다음과 같다. Rust release 빌드 후 프로젝트 루트에서 실행한다. 측정 report는 측정에 사용한 실행 파일과 소스 fingerprint를 보존한다. 측정 후 추가한 역할 parser 옵션 검증은 parser 초기화·로드에 적용된다.

```bash
source scripts/env.sh
cargo build --release -p sejong-cli
target/release/sejong benchmark \
  --config configs/sejong-dense.toml \
  --config configs/sejong-hierarchy.toml \
  --config configs/sejong-sparse.toml \
  --config configs/sejong-control-variate.toml \
  --config configs/sejong-dense-scaled.toml \
  --config configs/sejong-scaled.toml \
  --data docs/measurements/2026-11-01/train.txt \
  --validation docs/measurements/2026-11-01/validation.txt \
  --out runs/compute-comparison \
  --warmup-steps 2 --measure-steps 4 --evaluation-batches 2 --decode-bytes 16

target/release/sejong benchmark \
  --config configs/sejong-sparse.toml \
  --config configs/sejong-sparse-small-batch.toml \
  --config configs/sejong-control-variate.toml \
  --data docs/measurements/2026-11-01/train.txt \
  --validation docs/measurements/2026-11-01/validation.txt \
  --out runs/compute-learning-window \
  --warmup-steps 2 --measure-steps 64 --evaluation-batches 8 --decode-bytes 16
```
