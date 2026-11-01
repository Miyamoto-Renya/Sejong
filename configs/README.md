# 실행 설정

`sejong.toml`은 계층 byte 모델·GQA·전문가·순전파 기저 parameter VJP의 기본 실행 예산을 담는다. `sejong-parameter-vjp.toml`은 같은 값을 명시한 비교 프로파일이다. `sejong-dense.toml`, `sejong-hierarchy.toml`, `sejong-sparse.toml`은 정확 SGD의 구조별 비교 설정이다. `sejong-control-variate.toml`은 공유 encoder 교정 SGD를 사용한다. `donor-a.toml`과 `donor-b.toml`은 서로 다른 ID와 초기값을 적용한다. 기존 모델의 생략 필드는 `sejong-core` 기본값을 사용하며, 미분 표집과 교정 SGD의 예산은 모든 필드를 지정한다.

기본 두 프로파일은 활성값 보존과 `batch=8`, `accumulation=1`로 step당4,096 bytes를 처리한다. Global 선형층의 행 수512에서 기존 비용 선택기가 QKV의 압축 VJP를 선택한다. 이 배치는 현재26.49M·sequence512의 [RX 6600 실측](../docs/measurements/2026-11-01-architecture-fit/README.md)으로 정했다. 다른 모델 형상은 메모리 예산과 실행 시간에 맞춰 batch·누적·checkpoint 설정을 지정한다.

`sejong-sparse-small-batch.toml`은 step당 정확 미분을 1,024 bytes로 맞춘 교정 SGD 비교군이다. `sejong-dense-scaled.toml`과 `sejong-scaled.toml`은 head 폭·stage 비율을 유지하며 모델 폭을 2배로 늘린 dense와 교정 SGD 설정이다. [RX 6600 측정](../docs/measurements/2026-11-01/README.md)에 각 설정의 실제 파라미터 수·학습 시간·FLOPs·검증 손실을 기록했다.

Native 역할·사실 학습과 편입 설정은 `examples/native/`의 JSON 형식을 따른다. 모델별 실행 규격은 저장된 artifact에 함께 들어간다.

[계층 본체와 교정 학습](../docs/compute.md)에 설정의 의미, 저장·재개와 실제 성능 측정 명령을 기록한다.

`sejong-parameter-vjp.toml`은 현재 순전파 기저와 feature 잔차 표집으로 가중치 미분을 계산한다. `sejong-parameter-exact.toml`은 같은 전역 SGD 갱신 경로의 정확 미분 비교군이다. [실제 parameter VJP 학습](../docs/parameter-vjp.md)에 인수 gradient·norm·표집 RNG·NativeValue 연결을 기록했다.

[미분 변경의 RX 6600 측정](../docs/measurements/2026-11-01-parameter-vjp/README.md)은 기본 설정과 두 정확 미분 비교군의 학습 시간·주요 FLOPs·검증 BPB·재개 결과를 담는다.

[준비 비용을 줄인 후속 측정](../docs/measurements/2026-11-01-parameter-vjp-overhead/README.md)은 기존 B1/A8 구성의 고정 압축 경로와 관측 안의 기저·선택표 공유를 검증한다. 현재 B8/A1 기본 구성의66-step 실행은 step당 기저2개·입력 투영 묶음16개를 사용했다. Expert routing으로 압축 대상이 늘면 투영 묶음 수도 증가한다.

[검사 비용 정리 후 측정](../docs/measurements/2026-11-01-check-pruning/README.md)은 기본 clipping·진단 비활성 경로와 두 정확 SGD 비교군을 같은 66-step 예산으로 실행한다. 일반 학습의 FLOPs 계측은 명시한 계측 경로에서 활성화하고, `benchmark`는 모든 측정 step의 실행 수치를 수집한다.
