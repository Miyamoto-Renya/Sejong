# Sejong

Sejong은 Rust와 Python/Triton으로 구현한 확률 합성 모델이다. Rust가 모델·역전파·공유 사건·원본 편입·저장을 관리하고, Triton 커널을 HIP으로 실행한다. 통합 저장 형식은 `sejong-native-composite-v6`다.

의미 파서는 동결 MiniLM과 역할별 학습 head를 사용한다. 인원이 N명이면 scene은 `N²+1`, instruction은 `(N+1)²`, request는 `2(N+3)+1`개의 결과를 계산한다. 질문 파서는 anchor·goal을 선택하고 관계와 STOP을 조건부로 출력한다. 탐색 깊이는 질문에 등장한 술어 횟수에서 계산한다. 어휘 파서는 가변 길이 UTF-8 byte 입력을 처리한다.

## 실행

```bash
source /home/renya/Sejong/scripts/env.sh
python scripts/fetch-minilm.py
python scripts/create-native-data.py --out data/native-v6
bash scripts/train-native.sh runs/native-v6 data/native-v6

target/release/sejong query --model runs/native-v6/model --request data/native-v6/story-two.json
target/release/sejong sample --model runs/native-v6/model --request data/native-v6/query.json --seed 91
target/release/sejong read-weights --model runs/native-v6/model --request data/native-v6/weights.json --compare
```

자료 생성기의 기본 recipe는 인원 6명·술어 5개·최대 3단계 감독 경로다. `--people`, `--predicates`, `--max-depth`, `--query-examples`로 감독 자료를 정한다. 프로그램 감독은 목표 `Program`을 JSON으로 저장하고, 경로 길이별로 관측 수를 배분해 지정한 수의 프로그램을 추출한다. 모델 차원·학습량·학습률·readout 온도도 생성기 인자로 지정한다. 실제 사용한 설정은 `recipe.json`에 저장한다.

`fetch-minilm.py`는 공식 모델의 고정 revision과 SHA256으로 파일을 취득한다. Rust `tokenizers`가 자산의 `tokenizer.json`을 실행하고 Rust·Triton이 BERT를 실행한다. 현재 MiniLM의 위치 embedding 용량은 512토큰이다. 같은 GPU context에서 동일 encoder의 동결 가중치를 공유한다.

## 자원과 학습 정책

Host 예산은 Linux `MemAvailable`과 cgroup 잔여량, GPU 예산은 HIP의 실제 가용량을 사용한다. `budgets.memory_bytes`, 학습 `batch.memory_bytes`와 `batch.max_examples`, `gpu.memory_mb`로 별도 예산을 지정한다. 배치는 텐서 shape와 allocator의 실제 할당 단위에서 계산한다.

Native 학습은 활성 손실 항의 정규화된 가중 평균을 사용한다. `loss.weights`와 `query_loss.weights`로 비중을 지정한다. Memory 배경은 선언된 관측에서 독립 재표집한다. 정규분포 배경은 `background` 정책에서 표준편차와 zero 확률을 명시한다.

Optimizer의 평균화 길이는 관측 수와 예정 업데이트에서 계산한다. `optimizer`에서 평균화할 관측 수·epsilon·weight decay·gradient clipping을 지정할 수 있다. 체크포인트에 실제 배치·optimizer 설정·moments·RNG·자료 fingerprint를 저장한다. `checkpoint_every=0`은 최종 체크포인트 저장을 뜻한다.

## 원본과 확률 실행

| 명령 | 동작 |
| --- | --- |
| `train-role --spec --out [--resume]` | 의미·조건부 프로그램·어휘의 로컬 감독과 paired 학습 |
| `train-memory --spec --out [--resume]` | query 본체·affine 입구 준비와 K/V 사실 학습 |
| `train-value --spec --out [--resume]` | byte 요청의 SEP hidden으로 값 분포 학습 |
| `connect --spec --out` | 등록한 의미·어휘·사실 원본을 통합 |
| `revise --model --stage --spec --out [--resume]` | 이름으로 선택한 원본의 K/V 개정과 새 snapshot |
| `prepare --model --out` | 현재 가중치의 full/direct 수치 경계 재계산 |

Assembly의 `extra`로 사실 원본을 등록하고 `goal_routes`로 두 목적의 경로를 지정한다. 기본 경로는 `[a]`, `[a,b]`다. 사건의 독립 관계는 scan, 공유 사건 재진입은 MDD로 계산한다. 반복 어휘 선택은 마지막 사용까지 공유한다. Joint와 pullback은 입력 equality partition을 사용해 같은 행의 draw를 공유한다.

Memory는 선언된 context마다 동일한 FP64 실행 경로의 full/direct 분포를 대조해 TV 반경을 저장한다. K/V 개정 후 반경을 다시 계산한다. 답에는 `read_error`, `numerical_error`, `remaining`, 확률 구간과 최빈 답 판정을 함께 기록한다. 용량이 부족한 부분 탐색은 미처리 확률 질량을 보존한다.

값은 원시 byte열이다. JSON에서는 UTF-8 문자열 또는 `{"hex":"ff00fe"}`로 표현한다. `generate --raw`는 선택한 값을 그대로 출력한다. 독립 byte Transformer의 크기와 학습 설정은 `configs/*.toml`에 둔다.

## 코드와 확인

| 경로 | 내용 |
| --- | --- |
| `crates/sejong-core` | 동적 의미·프로그램·byte 값·정책·수치 규격 |
| `crates/sejong-gpu` | HIP 할당과 Triton 실행·역전파 |
| `crates/sejong-model` | BERT·의미 head·조건부 decoder·어휘·memory·native value |
| `crates/sejong-train` | 로컬 학습·자원 계산·개정·체크포인트 |
| `crates/sejong-runtime` | quotient·last-use·scan/MDD·partition·원본 registry |
| `scripts` | 자산 취득·자료 생성·학습·통합 확인 |

[구조](docs/architecture.md), [입력 규격](docs/native-data.md), [구현 대응표](docs/implementation-map.md), [GPU 경계](docs/gpu-boundary.md)에 실행 계약을 기록한다.

`bash scripts/check-system.sh`는 확장된 합성, 파서 학습·재개, memory 준비·개정을 세 묶음으로 확인한다. `native_system` 예제는 지정한 자료를 실제 학습하고 gold 질의 결과를 저장하는 전체 학습 실행이다. 환경 의존성은 `requirements-rocm.txt`와 `Cargo.lock`에 고정한다.
