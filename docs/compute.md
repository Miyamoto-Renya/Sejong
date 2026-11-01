# Sejong의 계층 본체와 교정 학습 실행

Sejong의 byte 본체와 NativeValue 본체는 설정에 따라 byte encoder, 구간 단위 global 본체, byte decoder를 연결한다. Rust가 tensor와 수동 역전파·optimizer·체크포인트를 관리하고 Triton이 HIP 메모리에 커널을 실행한다. Native 원본의 값 규격과 로컬 학습·봉인·확률 합성·memory full/direct 읽기는 기존 연결 계약을 사용한다.

`configs/sejong.toml`의 실행 경로는 계층 본체, GQA, 선택적 전문가와 [순전파 기저 parameter VJP](parameter-vjp.md)를 사용하는 SGD다. `sejong-dense.toml`, `sejong-hierarchy.toml`, `sejong-sparse.toml`은 정확 SGD로 구조별 비용을 비교하는 설정이다. `sejong-control-variate.toml`은 같은 본체에 공유 encoder의 교정 SGD를 적용한다. 기본 global 폭·층 수는 기존 설정의 512·8을 유지한다. 모델 크기는 각 stage와 전문가 수로 정하며, 실제 파라미터 수를 실행 결과에 기록한다.

[구조 변경의 RX 6600 측정](measurements/2026-11-01/README.md)에서 계층·GQA·전문가 구조는 같은 4,096-byte 정확 미분 step의 주요 FLOPs를 1,006.53에서 109.87 GFLOPs로 줄였고, 시간은 1.2383초에서 0.2658초로 줄었다. 교정 SGD의 66-step 비교에서는 같은 정확 미분 예산의 작은 배치 SGD가 더 빨랐으며, control 집합의 잔차 분산 진단비는 0.99970이었다. 현재 기본 설정은 이 본체와 SGD 갱신에 순전파 기저·잔차 표집을 결합한다.

| 설정 | 의미 |
| --- | --- |
| `model.hierarchy.patch_size` | global 본체를 실행할 byte 구간 길이 |
| `encoder_*`, `decoder_*` | 각 local stage의 폭·층·head·FFN 크기 |
| `local_window` | local attention이 읽는 causal byte 범위 |
| `checkpoint_encoder`, `checkpoint_decoder` | local stage의 활성값 재계산 선택 |
| `model.kv_heads` | query heads가 공유할 KV head 수; query head 수의 약수 |
| `model.experts.count`, `top_k`, `ffn_width` | 전문가 수, 위치당 선택 수, 전문가 내부 폭 |
| `training.optimizer` | `adamw`, `sgd`, `control_variate_sgd` |
| `training.control_variate` | 제어 관측 수·rank·표집·기저 갱신 주기·예측기 학습률의 명시적 예산 |
| `training.gradient_diagnostics_every` | 기울기 진단 주기; clipping 또는 진단 시점에 실제 norm을 계산 |

Local 폭은 기본 비교 프로파일에서 global의 1/4이고 head 폭은 같은 64다. 전문가 프로파일은 기존 FFN 내부 폭 1536을 전문가 4개 × 384로 나누고 2개를 선택한다. P8과 rank 16은 비교 프로파일에 명시한 실험 예산이다. 실행기는 모델 크기와 shape에 따라 allocation, cache, kernel grid와 rank 연산을 계산한다. CV를 선택하면 예산의 모든 필드를 지정해야 한다. 초기화는 각 tensor의 인덱스 범위·checked 크기·실제 GPU 할당 단위와 가용 메모리를 검사한 뒤 가중치를 만든다.

Encoder는 입력 byte를 causal하게 처리한다. Global 본체는 완료된 구간의 평균 표현을 처리하고, 그 결과가 다음 구간의 decoder에 전달된다. 첫 구간은 학습 가능한 BOS 표현을 사용한다. 마지막 partial patch의 출력은 다음 구간이 있을 때 사용한다. Global에서 decoder로 가는 큰 projection은 구간마다 한 번 실행하고 byte 위치에 broadcast한다. Decoder 출력은 public hidden 폭으로 투영하므로 NativeValue의 외부 embedding과 SEP readout이 같은 API를 사용한다.

추론 cache는 encoder·global·decoder의 KV와 진행 중인 구간의 합·길이·global 조건을 저장한다. 초기 prefill은 encoder·global·decoder를 전체 배치로 실행하면서 KV를 채운다. 이후 입력은 남은 구간 경계까지 여러 byte를 함께 처리하고 구간이 완성될 때 global을 실행한다. Partial prefill의 global slot은 해당 구간이 완성될 때 갱신한다. 한 byte 생성과 여러 byte prefill이 같은 causal 상태를 따른다. `Cache.reset()`은 모든 구간 상태를 초기화한다.

전문가 router는 전체 전문가 logit의 softmax 확률에서 top-k를 선택한다. 선택된 확률을 gate로 사용하므로 top-1에서도 router 미분이 생긴다. Triton이 선택 slot과 expert별 token 목록을 만들고 Rust가 실제 행 수로 GEMM을 실행한다. 빈 전문가는 건너뛰고 각 expert의 실제 행 범위를 사용한다. Count의 host 회수와 packing 비용은 실제 wall time에 포함한다.

교정 학습은 step 동안 본체·예측기·기저를 고정한다. 모든 M개 관측에 encoder를 실행하고, 선택된 m개 관측에는 global·decoder의 정확한 언어 미분을 수행한다. 제어 관측의 encoder 활성값은 같은 Graph에서 재사용한다. Feature는 encoder 평균, 목표 byte embedding 평균, 두 값의 원소곱 평균과 affine 좌표로 구성하며 폭은 `3 × encoder_width + 1`이다. 이 feature의 gradient는 predictor에 적용한다. 본체 encoder는 제어 관측의 언어 loss로 학습한다.

각 예측 행렬의 gradient를 `cᵢ=U Bᵢ Vᵀ`로 표현한다. 균등 표집의 추정량은 `mean(cᵢ) + mean_control(gᵢ−cᵢ)`다. 층화 표집은 입력 byte entropy로 자료를 정렬해 그룹을 고정한 뒤 그룹 안에서 균등 선택한다. 각 제어 gradient의 가중치는 `N_h/(M m_h)`다. Encoder·norm·embedding·BOS·router·routed expert는 정확한 제어 gradient로 갱신한다.

가중치는 `W_eff=W_base+U A Vᵀ`로 실행한다. W_base는 정확한 제어 gradient로 갱신하고 A는 예측 보정 계수로 갱신한다. Forward와 dX는 이 누적 성분을 즉시 반영한다. `XU`와 `δV`는 coefficient target에도 재사용한다. 모든 관측을 control로 지정하면 correction 계수가 상쇄되어 같은 구성의 정확 SGD가 된다.

기저 갱신은 제어 gradient 두 개의 차이를 얇은 양방향 sketch로 만들고 GPU에서 직교화한다. Control이 하나이면 해당 gradient 방향을 사용한다. Predictor 좌표 이동은 feature마다 두 rank 행렬곱을 수행해 O(feature × rank³) 연산으로 계산한다. Gram 행렬은 기저가 바뀔 때 갱신한다. 저장과 기저 교체 경계에서는 누적 A를 base에 합친다. Checkpoint는 가중치·U/V/A·predictor·데이터/표집/기저 RNG·학습 설정을 staging 디렉터리 안에 완성한 뒤 rename한다.

기울기 진단은 전체 parameter의 제어 gradient와 residual의 중심화한 분산을 기록한다. 층화 관측의 이 값은 관측된 weighted control 집합의 진단량이다. 실제 언어 학습의 지표는 별도 검증 자료의 bits-per-byte-token과 그 지점까지 사용한 누적 연산이다. Clipping·진단 시점에 적용할 전체 gradient의 FP64 norm을 계산한다.

학습과 생성은 기존 명령으로 연결된다.

```bash
source scripts/env.sh
cargo build --release -p sejong-cli
target/release/sejong train --config configs/sejong.toml --data data/corpus --out runs/sejong
target/release/sejong generate --model runs/sejong/model --prompt 'Sejong' --max-bytes 128
```

Native value 자료 생성에는 같은 모델 설정을 전달할 수 있다. 생성된 `native-b.json`의 model 필드에 설정 전체가 들어가며 `train-value`가 계층 본체를 학습한다. 연결할 원본은 assembly에서 기존 `kind: native`로 선택한다.

```bash
python scripts/create-native-data.py --out data/native-hierarchy --value-model-config configs/sejong.toml
target/release/sejong train-value --spec data/native-hierarchy/native-b.json --out runs/native-hierarchy-value
```

성능 비교는 실제 학습 경로를 실행하는 `benchmark` 명령을 사용한다. 설정마다 같은 자료·초기 seed·평가 관측을 사용하고, 모델 파라미터 수와 optimizer 설정을 결과에 기록한다. 학습 자료와 검증 자료의 fingerprint가 같으면 입력을 다시 지정하도록 오류를 반환한다.

```bash
target/release/sejong benchmark \
  --config configs/sejong-dense.toml \
  --config configs/sejong-hierarchy.toml \
  --config configs/sejong-sparse.toml \
  --config configs/sejong-control-variate.toml \
  --data data/train --validation data/validation --out runs/compute-comparison \
  --warmup-steps 2 --measure-steps 4 --evaluation-batches 2 --decode-bytes 16
```

`metrics.jsonl`은 첫 step·`log_every`·최종 step의 wall time과 관측량을 버퍼링해 기록한다. `benchmark`는 매 step 기록과 실행 계측을 활성화해 실제 shape의 GEMM/attention FLOPs·launch·컴파일 수를 수집한다. 일반 학습은 계측 상태를 `metrics_collected`로 기록하고 선택한 기울기 진단과 계측 수치를 채운다. `execution.json`은 기저 초기화·refresh·체크포인트와 마지막 A 합산까지 포함한 전체 실행을 기록한다. FLOPs의 logical 열은 유효 원소, issued 열은 커널 타일을 계산하며 FMA를 2로 센다. Pointwise·정규화·routing·전송·저장 비용은 전체 시간에 포함한다. `benchmark.json`에는 컴파일이 발생한 step, 컴파일 없는 step의 분포, 검증 loss 변화, prefill과 단일 byte decode 시간, full/cached logit 차이, source·실행 파일 fingerprint가 들어간다.

검증용 실행 파일은 GPU의 [hierarchy_check](../crates/sejong-gpu/examples/hierarchy_check.rs), 모델의 [hierarchy_check](../crates/sejong-model/examples/hierarchy_check.rs), 학습의 [control_variate_contract](../crates/sejong-train/examples/control_variate_contract.rs)다. 이들은 Rust 참조식으로 커널 미분을 대조하고, partial patch·여러 배치·인과성·캐시·NativeValue 저장·CV 기저 갱신·재개를 실제 GPU에서 확인한다.

[미분 연산 자체의 압축 연구](research/adjoint-compression.md)는 실제 본체의 cotangent spectrum, 비선형 미분의 rank 증가, 잔차 보정과 GPU VJP 시간을 분석한다. `adjoint_trace`는 진단용 전체 tape에서 선형 역전파 operand를 수집하고 `adjoint_replay`는 해당 FFN VJP를 재실행한다. [순전파 기저 parameter VJP](parameter-vjp.md)는 이 연구의 가중치 미분 경로를 실제 SGD·NativeValue 학습에 연결한다.

순전파의 QKV와 gate/up·SwiGLU는 공유 입력 tile의 다중 출력 커널을 사용한다. Graph의 Matmul·SwiGlu node 순서는 원래 미분·표집·checkpoint를 연결한다. RMSNorm backward는 최초 형상에서 선택한 partial·streaming reduction을 재사용하고, grouped attention은 query별 Delta를 dQ와 dKV 사이에 공유한다. [생산 통합 측정](measurements/2026-11-01-fused-applied/README.md)은 같은 본체의 학습·cached 추론·재개 결과를 담는다.

일반 GEMM과 결합 projection은 F16×F16 operand를 유지해 RDNA2 dot2로 계산하고 FP32에 누산한다. F32 operand를 포함한 곱은 F32 연산을 사용하며 dtype 선택은 Triton `constexpr`로 컴파일 시 확정한다. 기본 byte 프로파일은 활성값 보존·batch8·누적1로 기존 압축 VJP의 global QKV 적용 범위를 넓혔다. [현재 생산 실행](measurements/2026-11-01-architecture-fit/README.md)의 평균 학습 시간은111.436ms/step이다.
