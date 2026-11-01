# Memory·NativeValue 학습 경로 통합

RX 6600(gfx1032)에서 원본 학습·개정·저장·재개를 실행했다. 이전 코드는 `7956bab9621fff129d316784af25a64a2dcd9e5c`에서 별도로 빌드했다. 원자료는 [summary.json](summary.json)에 기록했다. 입력은 개발용 회귀·실행 시간 fixture다.

## 연결한 동작

- Memory query는 기존 결합 QKV 커널을 사용한다. FP64 경로는 기존 FP64 행렬곱을 사용한다.
- fact 학습과 revision은 고정 query를 한 번 계산하고 재사용한다. query와 fact의 배치는 각 작업 공간으로 정한다. 캐시가 GPU 예산에 들어오면 GPU에, 더 큰 캐시는 임시 FP32 파일에 보관한다.
- NativeValue의 `optimizer_kind=sgd`는 byte 학습과 같은 `GradientPlan`으로 인수·선택 행·dense gradient를 직접 갱신한다. 기본 AdamW는 기존 moments를 사용한다.
- NativeValue의 배치 계획은 local/global 길이, GQA, 활성 expert, 실제 순전파 dtype과 FP32 adjoint, VJP buffer 수명 및 allocator 반올림을 계산한다. 자동 모드는 활성값 보존을 우선하고, 한 예제도 보존하기 어려우면 checkpoint 재계산을 선택한다.
- 체크포인트는 선택된 batch·checkpoint 정책·VJP RNG를 보존한다. 이전 checkpoint의 생략된 정책은 기존 재계산 동작으로 복원한다.
- `create-native-data.py --value-model-config configs/sejong.toml`은 모델·VJP와 함께 SGD·checkpoint·물리 batch 설정을 NativeValue 규격에 전달한다.

## 실행 측정

학습·개정 시간은 초기화, checkpoint와 최종 저장, Memory 보정을 포함한다. query-only와 동일 gradient 갱신은 해당 연산만 측정한다. 표의 커널 컴파일 횟수는 0이며, 이미 컴파일된 모듈을 준비하는 시간은 포함한다.

| 실행 | 이전 | 적용 후 | 관측 |
|---|---:|---:|---|
| Memory: query 4회, fact 16순회 | 0.30786초 | 0.27967초 | 시간 9.16%, 주요 FLOPs 52.18% 감소 |
| 같은 원본 revision 128순회 | 0.39461초 | 0.21587초 | 시간 45.30%, 주요 FLOPs 96.14% 감소 |
| 고정 query 128순회 | 0.28867초 | 0.28596초 | launch 25,600→23,040, checksum 동일 |
| NativeValue Adam 64 step | 8.75764초 | 8.25665초 | 시간 5.72%, 주요 FLOPs 19.04% 감소 |
| 동일 NativeValue gradient 갱신 1회 | 복원 후 갱신 4.210ms | 직접 갱신 3.071ms | 전체 가중치 비트 일치 |

Memory 형상은 dimension 64, context 32, 2층, hidden 256, 요청 65개, batch 16이다. NativeValue는 26,472,736 파라미터의 계층·GQA·MoE 본체, batch 8, 입력 512바이트 위치다. NativeValue의 16-step SGD 전체 호출은 2.848초, GPU peak 603.0MB를 기록했다. Adam과 SGD는 각 optimizer의 학습 궤적을 사용한다.

64-step Adam fixture의 마지막 batch loss는 이전 2.3823, 적용 후 3.0221이었다. 이어서 동일 가중치·동일 배치의 dense 및 압축 gradient를 대조했다. 두 checkpoint 모드의 loss, VJP RNG와 압축 행렬 수가 같았고, `native_embedding`을 제외한 모든 gradient가 비트 단위로 같았다. 압축 경로의 embedding 최대 차이는 4.17e−7이며, 동일 checkpoint 경로 반복의 차이는 6.56e−7이었다. 해당 embedding backward는 FP32 atomic 합산을 사용한다. 위 시간표는 고정 실행량의 실행 비용을 기록한다.

## 약 0.3B 형상 실행

291,220,000 파라미터 NativeValue를 같은 RX 6600에서 실행했다. global 폭 1024, 13층, GQA 16/8, expert 8개 중 2개, FFN 768, local 폭 256·각 2층, patch 8, 입력 1024바이트 위치다. 값 head는 fixture의 32개 범주다.

- SGD 업데이트 3회, checkpoint·최종 저장까지 완료.
- 자동 실행 계획: **batch 8, 활성값 보존**.
- 첫 step: **압축 행렬 59개**, 공유 basis 2개, projection group 21개.
- 최대 GPU 할당: **4,941,625,600 bytes**, 약 **4.60GiB**.
- 초기화·업데이트·저장을 포함한 전체 호출: **12.693초**.

초기 FP32 상한 예약에서는 batch 6이었다. 순전파 FP16과 adjoint FP32를 각각 반올림해 합산하면서 batch 8이 선택됐고, global QKV의 압축 조건도 충족했다. 준비·저장 비용이 포함된 이 3-step 측정의 용도는 확대 형상의 실제 실행과 메모리 확인이다.

## 저장·재개 및 재현

- Memory revision, 파일 캐시 경로, fact/revision 중간 재개의 최종 가중치가 각각 기준 실행과 비트 단위로 일치했다.
- NativeValue Adam·SGD 재개의 RNG와 출력 확률이 일치했다. 가중치 최대 차이는 7.45e−9였다.
- 기존 NativeValue checkpoint의 본체 가중치가 이전 실행과 일치했다.
- `memory_contract`, Rust 학습 패키지 단위 테스트 3개, workspace 전체 target 검사, release CLI 빌드를 실행했다.

```bash
python3 scripts/bench-native-integration.py prepare
cargo build --release -p sejong-train --example native_training_benchmark
source scripts/env.sh
python3 scripts/bench-native-integration.py run \
  --executable target/release/examples/native_training_benchmark \
  --tag measured --mode memory
python3 scripts/bench-native-integration.py run \
  --executable target/release/examples/native_training_benchmark \
  --tag measured-scale --mode value \
  --spec .cache/native-integration/specs/value-scale.json
```

NativeValue의 동일 gradient 갱신 비교는 `native_sgd_update_check`, checkpoint 경계의 고정 배치 gradient 대조는 `native_gradient_equivalence`, 저장·재개 비교는 `native_parameter_vjp` 예제로 실행한다. 생성된 학습 산출물은 `.cache/native-integration/` 아래에 둔다.
