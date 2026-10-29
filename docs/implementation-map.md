# Sejong v6 구현 위치

Native graph는 로컬 의미 학습, 조건부 프로그램 해석, 원본의 full/direct 읽기와 확률 합성을 연결한다. 구현은 Rust 패키지와 Python/Triton 커널로 구성한다.

| 구성 요소 | 코드 | 작동 방식 |
| --- | --- | --- |
| 공통 규격·정책 | `crates/sejong-core/src/native.rs` | 동적 의미 크기, Program 레이블, 원본 registry와 경로, 손실·배치·optimizer·예산 정책 |
| 수치 오차 | `crates/sejong-core/src/numeric.rs` | IEEE-754 연산 수와 값 크기에 따른 반올림 오차, 상·하향 인접값 |
| 동결 언어 표현 | `crates/sejong-model/src/native/bert.rs` | 공식 MiniLM·WordPiece, 가변 개체 span pooling, 위치 table에 따른 입력 길이 |
| 의미·질문·어휘 원본 | `crates/sejong-model/src/native/classifier.rs` | 개체 수에 따른 의미 분포, 조건부 Program 접두사·STOP 채점, 가변 byte 어휘 encoder |
| 로컬 파서 학습 | `crates/sejong-train/src/roles.rs` | 정답 CE, 좌표 정렬 JS, softplus 대비, 실제 배치와 optimizer·RNG 재개 |
| Native 사실 값 | `crates/sejong-model/src/native/value.rs` | 요청 SEP hidden의 value head와 값 분포 |
| Memory 원본 | `crates/sejong-model/src/native/memory.rs` | 설정 기반 query 본체·K/V·prototype readout, 스트리밍 QR 입구 피팅, 선언 행의 FP64 full/direct 보정 |
| Memory 학습·개정 | `crates/sejong-train/src/memory.rs` | 선언한 손실·배경·optimizer 정책, 메모리에 맞춘 청크 계산, K/V 학습과 재개 |
| 의미 quotient | `crates/sejong-runtime/src/native/semantic.rs` | 개체 N명에 대해 2N개 채널과 160N개 충분 상태 |
| 어휘 last-use | `crates/sejong-runtime/src/native/lexical.rs` | 반복 의미 선택 공유, 마지막 사용에서 상태 합산, 살아 있는 선택의 바이트 예산 |
| 사건 실행 | `crates/sejong-runtime/src/native/events.rs` | N²+NONE 사건, 분리된 술어의 scan kernel, 재진입의 공유 선택 MDD |
| 공동 법칙 | `crates/sejong-runtime/src/native/partition.rs` | 입력 크기와 차수에 따른 equality partition 순차 생성, tensor pushforward·pullback·읽기 오차 전파 |
| 확률 frontier·원본 연결 | `crates/sejong-runtime/src/native/mod.rs` | 접두사 확장, 이름 기반 원본·경로, 남은 질량·읽기 오차·수치 오차와 최빈 답 판정 |
| 작업 공간 계산 | `crates/sejong-runtime/src/native/resources.rs` | 형상 곱·tensor 거듭제곱의 checked 계산, 바이트 사전 계산과 용량 오류 |
| 연속 읽기·개정·회계 | `crates/sejong-runtime/src/native/mod.rs` | 경로별 full/direct 입력, 단위 공 투영, 원본 서명·개정 편입, 공유 encoder를 반영한 비용 집계 |
| GPU tensor·연산 | `crates/sejong-gpu/src/` | HIP 메모리·stream·Triton 실행, 역전파 그래프, 장치 메모리에 따른 배치 여유량 |
| Triton 커널 | `python/sejong_kernels/ops/` | 일반 tensor·Transformer 연산, 관계 attention, gradient와 Adam |

확률 `goal_routes`의 각 경로는 서로 다른 원본 ID를 사용한다. 연속 `WeightRequest.route`는 같은 원본을 여러 번 연결할 수 있다. 두 경로 모두 등록된 이름과 실제 입출력 규격으로 연결한다.

`native_system` 예제는 로컬 학습, 원자 사실·질문 gold, 좌표 순열·이름 변경, NONE, full/direct 구간, native 원본 혼합, 공동 법칙, 저장·복원·샘플링·개정을 연결하고 `answers.json`을 저장한다.

`parser_system`은 가변 의미 좌표·길이·프로그램 선택과 파서의 미분을 대조한다. 공식 ONNX 참조 출력은 `scripts/parser-reference.py`로 생성한다. `parser_resume`은 checkpoint에서 이어 학습한 가중치·optimizer·RNG를 비교한다. `memory_contract`는 선언 행의 보정과 실제 사실 학습·개정·재개를 묶어 실행한다. `input_sized_exact_composition`은 의미 quotient·8단계 공유 사건·5차 공동 법칙·pullback·어휘 last-use를 독립 열거 결과와 대조한다.
