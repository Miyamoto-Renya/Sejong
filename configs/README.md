# 실행 설정

`sejong.toml`은 계층 byte 모델·GQA·전문가·정확 SGD의 기본 실행 예산을 담는다. `sejong-dense.toml`, `sejong-hierarchy.toml`, `sejong-sparse.toml`은 정확 SGD의 구조별 비교 설정이며, 기본 설정과 `sejong-sparse.toml`의 실행 값이 같다. `sejong-control-variate.toml`은 공유 encoder 교정 SGD를 사용한다. `donor-a.toml`과 `donor-b.toml`은 서로 다른 ID와 초기값을 적용한다. 기존 모델의 생략 필드는 `sejong-core` 기본값을 사용하며, 교정 SGD의 예산은 모든 필드를 지정한다.

`sejong-sparse-small-batch.toml`은 step당 정확 미분을 1,024 bytes로 맞춘 교정 SGD 비교군이다. `sejong-dense-scaled.toml`과 `sejong-scaled.toml`은 head 폭·stage 비율을 유지하며 모델 폭을 2배로 늘린 dense와 교정 SGD 설정이다. [RX 6600 측정](../docs/measurements/2026-11-01/README.md)에 각 설정의 실제 파라미터 수·학습 시간·FLOPs·검증 손실을 기록했다.

Native 역할·사실 학습과 편입 설정은 `examples/native/`의 JSON 형식을 따른다. 모델별 실행 규격은 저장된 artifact에 함께 들어간다.

[계층 본체와 교정 학습](../docs/compute.md)에 설정의 의미, 저장·재개와 실제 성능 측정 명령을 기록한다.
