# Parameter VJP의 준비 비용을 줄인 실행

2026-11-01. [초기 구현](../2026-11-01-parameter-vjp/README.md)의 기저 생성·표집·투영 준비를 분리해 측정하고, 압축 경로와 자원 수명을 변경했다. 원본은 [초기 HIP event 계측](initial-profile.json), [최종 benchmark](benchmark.json), [학습·재개 검증](verification.json)에 저장했다.

초기 3-step 진단 실행에서 launch 앞뒤 HIP event 구간의 step당 합은 기저 생성 15.73ms, 표집 9.44ms, 입력 투영·잔차 준비 15.13ms, 인수 갱신 6.89ms였다. 이 계측에는 각 launch의 event 기록을 적용했다. 아래 일반 학습 시간은 기본 release 빌드에서 별도로 측정했다.

구현 변경:

- `N≥입력 폭`인 긴 입력과, 여러 parameter 행렬이 같은 입력을 사용하는 묶음에 압축을 적용한다. P8의 짧은 global 경로와 단독 행렬은 정확 미분을 실행한다. 남은 후보의 형상 비용은 기존 GEMM·QR 타일 식으로 비교한다.
- Graph의 parameter 입력 묶음을 역전파 시작 때 한 번 순회해 작성한다.
- 같은 관측 안에서 `(행 수, rank)`가 같은 입력들은 token 기저를 공유한다. 최초 묶음의 가장 넓은 현재 순전파 출력을 donor로 사용한다.
- 입력 폭이 같은 층은 feature 선택표를 공유한다. 각 입력의 `XᵀQ`와 선택 잔차는 개별 계산한다.
- Checkpoint 재계산 경계는 기저·선택표·표집 RNG를 전달한다. 다음 관측의 Graph는 새 자원을 생성한다.
- HIP event 계측은 `kernel-profile` 빌드 feature와 `SEJONG_PROFILE_KERNELS` 환경 변수로 켠다. 기본 빌드는 일반 launch 경로를 실행한다.

공유 기저 Q와 입력별 `Aᵢ=XᵢᵀQ`에 대해 `Eᵢ=Xᵢ−QAᵢᵀ`다. 따라서 `Aᵢ(QᵀDᵢ)+EᵢᵀDᵢ=XᵢᵀDᵢ`가 성립한다. 같은 feature 선택표를 여러 층이 사용할 때 각 행의 포함 확률은 p이며, 잔차를 `1/p`로 보정한 각 parameter 미분의 표집 평균이 원래 미분과 일치한다. 공유 표집은 층 사이의 표집 상관을 만든다. 아래 검증은 해당 공유 정책의 실제 학습 결과다.

RX 6600, Rust release, 26,488,192 parameters, step당 4,096 bytes, 66번 갱신 중 첫 2 step 이후 64 step을 집계했다. 평가 자료와 학습 자료는 [초기 측정](../2026-11-01/README.md)의 별도 문서다. 초기·최종 검증은 512-byte 배치 8개를 사용한다. 세 구성 모두 측정 구간의 신규 커널 컴파일 수가 0이며, full/cached 마지막 logit 최대 차이는 0이었다.

| 현재 소스의 구성 | 평균 초/step | Issued GFLOPs/step | Launch/step | Peak MiB | 최종 BPB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 기존 정확 SGD | 0.263385 | 109.71519 | 14,728.63 | 465 | 7.935028 |
| 전역 타일 SGD·정확 미분 | 0.256955 | 109.71696 | 14,334.89 | 465 | 7.935005 |
| 기저·선택표 공유의 기본 설정 | 0.259391 | 107.18642 | 14,253.63 | 472 | 7.934876 |

초기 압축 구현의 0.278218초와 비교하면 시간은 6.77% 줄었다. 같은 실행의 기존 SGD 대비 시간 감소는 1.52%이며, 전역 타일 SGD의 정확 미분 대비 시간 증가는 0.95%다. 짧은 global과 단독 행렬에 정확 미분을 적용해 issued FLOPs는 초기 압축 구현의 99.33에서 107.19 GFLOPs로 늘었다. 현재 소스의 정확 SGD 대비 주요 issued FLOPs 감소는 2.30%다.

Step당 기저 생성은 초기 구현의 약 540회에서 8회로 줄었다. 최종 기본 구성은 입력 투영 묶음 64개에서 parameter 행렬 160개를 압축한다. Peak GPU 메모리는 초기 577에서 472 MiB로 줄었다. 평가한 범위는 위 66번 갱신과 validation 4,096 bytes다.

전체 byte 모델의 6-step 학습과 checkpoint 3에서의 재개를 대조했다. 최종 가중치 최대 차이는 2.84×10⁻¹⁴, 저장된 optimizer RNG 차이는 0이었다. 전체 잔차 관측의 정확 SGD 대조는 최대 4.66×10⁻¹⁰이었다.

NativeValue는 [동일 본체의 spec](../2026-11-01-parameter-vjp/native-b.json)으로 2-step 학습과 checkpoint 1 재개를 실행했다. Step마다 기저 1개를 8개 입력 묶음과 parameter 행렬 20개가 공유했다. 미분 RNG는 일치했고, 재개 가중치 최대 차이는 1.19×10⁻⁷이었다. 봉인·재로딩 가중치와 32개 요청 전체의 재개 출력 확률 최대 차이는 각각 0이었다. Rust workspace library tests 5개, all-targets 검사와 계측 feature 검사도 통과했다.

```bash
source scripts/env.sh
cargo build --release -p sejong-cli --locked
target/release/sejong benchmark \
  --config configs/sejong-sparse.toml \
  --config configs/sejong-parameter-exact.toml \
  --config configs/sejong.toml \
  --data docs/measurements/2026-11-01/train.txt \
  --validation docs/measurements/2026-11-01/validation.txt \
  --out runs/parameter-vjp-reused-rx6600 \
  --warmup-steps 2 --measure-steps 64 \
  --evaluation-batches 8 --decode-bytes 16
```

진단 설정의 `training.steps=3`, `checkpoint_every=0`으로 launch 구간을 계측할 수 있다.

```bash
cargo build --release -p sejong-cli --features sejong-gpu/kernel-profile --locked
SEJONG_PROFILE_KERNELS=1 target/release/sejong train \
  --config .cache/profile.toml \
  --data docs/measurements/2026-11-01/train.txt \
  --out runs/parameter-kernel-profile
```

`kernel-profile.json`이 연산별 launch 수와 HIP event 구간을 기록한다. 일반 성능 비교는 기본 release 빌드를 사용한다.
