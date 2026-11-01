# 원본 10개 × 일반 토큰 8B 비용 계산

이 문서는 원본 학습 통합 전 코드의 비용 계산을 기록한다. 후속 통합은 Memory query 캐시, NativeValue의 직접 SGD 갱신과 계층별 실행 계획을 추가한다. 아래 반복 query·기존 workspace 예약 수치는 통합 전 상태의 비교 기준이다.

사용자 지정 데이터량은 원본당 일반 토크나이저 기준 8B 토큰이다. 원본 10개의 총 학습 노출량은 80B 토큰이다. RX 6600 한 장에서 원본을 순차 학습하는 조건으로 계산한다. 실제 코퍼스의 바이트/토큰 비율과 요청 구성에 따라 계산 변수가 결정된다.

## 측정 근거

`kernel-probe.json`은 현재 원본 Triton 커널을 HIP으로 실행한 측정이다. 0으로 채운 입력, warmup 1회, 측정 3회 중앙값이며 할당과 컴파일은 측정 구간 밖에 둔다. 대표 순전파 GEMM은 1.43~2.48 TFLOP/s, 입력 역전파 GEMM은 0.40~0.48 TFLOP/s다. 128MiB pageable CPU→GPU 전송의 후반 3회 중앙값은 13.09GB/s다. 이 값들의 측정 범위는 개별 커널과 전송이다.

전체 학습 시간 환산에는 기존 `../2026-11-01-architecture-fit/applied.json`의 26,488,192 파라미터 byte-SGD 실행을 사용한다. 주요 연산 74.727GFLOP / 평균 0.111436초 = 0.67058TFLOP/s다. 아래 시간은 이 처리량을 큰 모델 및 다른 학습 경로에 적용한 시나리오다. NativeValue의 Adam, 배치 크기, Memory의 FP32 연산 및 CPU 왕복 비용이 실제 처리량을 결정한다.

## NativeValue 계산 예시

계산용 형상은 global 폭 1024, 13층, GQA 16/8, expert 8개 중 2개 활성화, expert FFN 768이다. local encoder/decoder는 각각 폭 256, 2층, FFN 768, window 128이며 patch 크기는 8이다. 출력 값 4096개, 입력 길이 1024바이트 위치를 적용한다. 학습 대상은 295,385,600 파라미터이며 10개 합계는 2.953856B다. 공유 분류기와 연결 구성은 별도다. NativeValue는 요청 마지막 위치에서 범주형 값을 학습하므로 여기서 토큰량은 입력 텍스트 노출량을 뜻한다.

구조의 patch와 top-k를 적용하면 바이트당 선형 가중치 사용량은 16,856,064다. 주요 학습 연산은 다음처럼 계산한다.

```
F/byte = 6 × 가중치 사용량
       + attention 순전파·역전파
       + global checkpoint 순전파 재계산
       + 요청별 head 및 BOS 투영
       ≈ 131,322,368 FLOP
Ftotal = 10 × 8e9 × r × F/byte
days = Ftotal / (R × 86400)
```

`r`은 코퍼스 UTF-8 바이트 수 / 일반 토큰 수다. 바이트 비율 2·4·6은 감도 분석용 입력이다. 계산 예시에서 local checkpoint는 둘 다 false이며 global checkpoint는 현재 NativeValue의 학습 그래프를 따른다. dense gradient의 주요 행렬 연산을 집계했다. parameter-VJP 정책의 rank, residual fraction과 배치별 적용 여부에 따른 변화는 해당 설정으로 재계산한다. 입력의 BOS/SEP와 padding은 실제 데이터에서 추가 집계한다.

| 바이트/일반 토큰 | 원본 1개 | 원본 10개 | 전력량: PC 200W 가정 | 전기 비용: 200원/kWh 가정 |
|---:|---:|---:|---:|---:|
| 2 | 36.3일 | 362.7일 | 1,741kWh | 34.8만원 |
| 4 | 72.5일 | 725.3일 | 3,481kWh | 69.6만원 |
| 6 | 108.8일 | 1,088.0일 | 5,222kWh | 104.4만원 |

전력과 단가는 예시 입력이다. 실제 비용은 벽면 측정 평균전력과 해당 전기요금으로 대입한다. 장비 구입, 데이터 준비, 저장장치와 추가 보정 학습은 별도 비용 항목이다. r=4에서 유효 처리량이 0.4/0.67/1.0TFLOP/s면 1,216/726/486일이다.

현재 NativeValue 배치 계획은 FP32 가중치, Adam 상태, gradient 3벌 예약, FP16 cast를 합쳐 대략 26N바이트를 계산한다. 위 형상에서 약 7.68GB이며, 길이 1024의 예제 한 개 workspace 예약이 약 2.31GB다. 합계 약 9.99GB가 RX 6600의 8GiB를 초과한다. parameter-VJP materializer를 켜면 dense gradient 버퍼도 추가된다. 위 시간표를 실행 계획으로 옮기려면 workspace 계획의 계층 구조 반영과 gradient/optimizer 메모리 축소가 필요하다. 이 값은 코드의 예약 계산이며 실행 peak 측정은 별도다.

## Memory 계산 예시

dimension=1024, layers=6, FFN=4096, hidden=109088로 놓으면 다음과 같다.

```
Pbody = L(4d² + 2df + 8d + f) + d² = 76,619,776
PKV = 2dh = 223,412,224
원본 본체+K/V = 300,032,000
affine port = (d+1)d = 1,049,600
readout = (d+1) × 값 개수
```

Memory는 요청 벡터와 context, 목표 query, 값 레이블을 입력으로 받는다. 일반 토큰 수를 이 학습량으로 바꾸려면 요청 하나에 배정한 일반 토큰 수와 학습 반복량을 정한다. context 위치 T=128은 벡터 시퀀스 길이이며 일반 토크나이저의 길이와 독립적인 설정이다.

```
Fbody = 2TL(4d²+2df) + 4LT²d + 2d² ≈ 19.732 GF/request
Ffact = Fbody + 5PKV ≈ 20.849 GF/request visit
Fquery ≈ 3 views × 3Fbody ≈ 177.589 GF/base visit
```

query의 3배는 forward, 입력 gradient, 가중치 gradient의 주요 행렬 연산 근사다. 세 view 활성화를 가정한다. 같은 0.67058TFLOP/s 환산에서 query 기본 요청 100만 회는 약 73.56시간, fact 요청 100만 회는 약 8.64시간이다. fact_steps 한 번은 전체 레이블 순회다. query 횟수는 query_steps × 실제 batch다.

원본당 요청 수를 M, query 기본 방문 수를 Q, fact_steps를 E라고 하면:

```
10개 시간(hours) ≈ 10 × [73.56 × Q/1e6 + 8.64 × M×E/1e6]
                  + port fitting·calibration·저장 시간
```

8B 토큰을 요청당 평균 a개 토큰으로 묶는 데이터 구성에서는 M=8e9/a다. 겹치는 요청과 추가 합성 예제를 만들면 해당 요청 수를 함께 집계한다. 예를 들어 a=128, Q=M, E=1이면 10개 주요 연산 환산은 약 2,141일이다. 이 예시의 요청 구성과 방문 수는 계산 입력이며 사용자가 지정한 학습 설정은 원본당 8B 일반 토큰이다. 이 구성의 원본당 요청 6,250만 개를 현재 FP64 context 배열로 펼치면 M×T×d×8=65.536TB다. 대규모 데이터 구성에는 context 생성·스트리밍과 저장 형식 변경이 필요하다.

fact_update는 고정 body와 고정 context에서 query를 매번 다시 계산한다. 위 형상에서는 body가 fact 주요 FLOPs의 94.64%다. query를 한 번 계산해 재사용하는 구현은 반복 fact 연산을 20.849GF에서 약 1.117GF로 줄인다. 비율은 약 18.66배다. query 캐시 FP32 저장량은 요청당 4d=4096바이트이며 100만 요청당 4.096GB다. 캐시의 전처리 시간과 저장/전송 비용을 실행 계획에 포함해야 한다. query 학습과 port 보정 비용은 위 식에 계속 포함된다.

## 합성 및 적재

현재 connect는 원본과 계약을 로드하고 연결·봉인·저장한다. 합성 단계에 전체 원본의 공동 역전파는 없다. 시간표에서 원본 10개 비용을 합산한다.

현재 Memory는 FP64 가중치를 적재한다. 위 본체+K/V 10개가 약 24.00GB이며 port/readout이 추가된다. 현재 Composite는 원본을 같은 GPU context에 함께 적재한다. 8GB 환경의 합성 실행에는 필요 원본 적재 또는 direct 경로용 K/V 분리 구현이 필요하다.

direct 경로의 K/V GEMM 실측 합계는 요청 1개, 원본 1개당 15.43ms다. 계획 중인 CPU 상주 K/V+port paging을 가정하면, 원본 10개 전체를 한 번씩 옮기는 전송량은 약 17.96GB이며 측정 대역폭 환산은 1.37초다. 여기에 K/V GEMM 약 0.154초, port/GELU/readout 및 런타임 비용을 더한다. 이 수치는 10개가 모두 필요한 경로의 직렬 계산 예시다. 실제 호출 원본 수, 캐시 적중률, CPU RAM 용량과 스트리밍 구현이 합성 지연을 결정한다.

## 재계산과 소스

`python3 docs/measurements/2026-11-02-composite-cost/calculate.py`로 `estimate.json`의 수치를 재계산한다. `kernel-probe.py`는 프로젝트 루트에서 `PYTHONPATH=python .venv/bin/python`으로 실행하며 측정 출력은 `.cache/composite-cost`에 기록한다. 원래 측정의 출력 디렉터리는 먼저 생성한다.

- 구조 파라미터 및 checkpoint: `crates/sejong-model/src/architecture.rs`
- checkpoint 재실행: `crates/sejong-gpu/src/graph.rs`
- attention 연산 집계: `crates/sejong-gpu/src/counters.rs`
- NativeValue 출력과 학습 대상: `crates/sejong-model/src/native/value.rs`
- Memory 배치, query/fact 단계, NativeValue Adam: `crates/sejong-train/src/memory.rs`
- Memory 파라미터 및 port: `crates/sejong-model/src/native/memory.rs`
- Composite 적재와 연결: `crates/sejong-runtime/src/native/mod.rs`
