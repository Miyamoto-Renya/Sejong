# Native 입력과 학습 규격

규격은 `crates/sejong-core/src/native.rs`에 정의한다. 편입 경로는 spec 파일의 디렉터리를 기준으로 해석한다. 학습 결과는 `<run>/model/`에 저장한다. Role의 최종 재개 상태는 `<run>/checkpoint/`, 주기별 role 상태와 memory·value의 재개 상태는 `<run>/checkpoints/`에 저장한다.

## 역할과 프로그램

`train-role --spec <JSON> --out <run> [--resume <checkpoint>]`를 사용한다. `role`은 scene·instruction·request·query·lexical이며 `model`에 폭·층·head·FFN·정규화 설정·seed를 지정한다. 의미 head는 `encoder`의 동결 MiniLM을 공유한다.

`examples`는 `text`, `people`, `predicates`, `label`, `program`으로 구성한다. 목록의 순서가 해당 관측의 좌표다. 인원 수 N에서 다음 레이블을 사용한다.

| 역할 | 레이블 |
| --- | --- |
| scene | `agent×N+patient`, NONE=`N²` |
| instruction | `focus×(N+1)+excluded`, N은 좌표 유지, NONE=`(N+1)²−1` |
| request | `goal×(N+3)+selector`; selector는 개체 0…N−1, agent=N, patient=N+1, focus=N+2; NONE=`2(N+3)` |
| query | `program`의 목표 구조, `null`은 NONE; `label`은 0 |
| lexical | `meanings` 목록의 인덱스 |

```json
{
  "text": "Where does the first person whom alice called work?",
  "people": ["alice", "bob", "carol", "david", "erin", "frank"],
  "predicates": ["called", "helped", "visited"],
  "label": 0,
  "program": {"goal": 0, "anchor": 0, "steps": [{"predicate": 0, "reverse": false, "last": false}]}
}
```

Goal 0은 장소, goal 1은 물품이다. `reverse`는 관계 방향, `last`는 마지막 일치 사건 선택이다. 조건부 decoder는 root와 각 관계 이후의 다음 step·STOP을 학습한다. 입력에서 발견한 술어 횟수가 프로그램의 최대 step 수다.

`pairs`의 각 항목은 `left`, `right`, `equivalent`다. 같은 의미의 관측은 좌표를 정렬한 Jensen–Shannon 손실, 다른 의미의 관측은 정답 log 확률의 logistic 대비 손실을 사용한다. `loss.weights`의 이름은 `supervised`, `consistency`, `contrast`다. 빈 설정은 활성 항의 동일 비중을 뜻한다.

## 학습 자원과 optimizer

공통 설정은 `steps`, `learning_rate`, `optimizer`, `batch`, `checkpoint_every`, `resume`다. Role의 `batch_size=0`은 자동 크기, 양수는 명시적인 관측 수 예산이다.

```json
{
  "batch": {"max_examples": 64, "memory_bytes": 1073741824},
  "optimizer": {"momentum_examples": 4096, "variance_examples": 65536, "epsilon": 0.0, "weight_decay": 0.0},
  "checkpoint_every": 0
}
```

이 값들은 사용자가 지정하는 정책의 예시다. 생략한 배치 크기는 관측 수·실제 가용 GPU 메모리·텐서별 할당량에서 계산한다. 평균화 구간 H는 지정한 관측 수를 배치 수로 나눈 올림이며 `β=(H−1)/(H+1)`로 변환한다. 기본 momentum 구간은 관측 전체 한 회분, variance 구간은 예정 업데이트 전체다. 기본 epsilon과 weight decay는 0이며 gradient clipping은 `clip_norm`을 지정하면 적용한다.

실제 배치·optimizer 설정·moments·RNG·학습 fingerprint를 체크포인트에 저장한다. `checkpoint_every=0`은 최종 저장, 양수는 해당 업데이트 간격의 추가 저장이다.

## Memory와 native 값

`train-memory`의 `config`에는 `id`, `dimension`, `tokens`, `layers`, `heads`, `hidden`, `ffn_width`, `norm_epsilon`, `readout_temperature`, `residual_scale`, `seed`를 지정한다.

- `requests`, `values`: 고유한 입력·출력 byte 값 목록.
- `contexts`: 요청마다 `tokens×dimension` 길이의 context. 첫 token이 연속 request다.
- `target_queries`: 요청마다 dimension 길이의 query 감독 좌표.
- `value_targets`: 값마다 dimension 길이의 목표 벡터.
- `labels`: 요청별 목표 값 인덱스.
- `query_steps`, `query_learning_rate`, `min_query_learning_rate`: query 준비와 cosine 일정.
- `fact_steps`, `learning_rate`: K/V 학습량과 학습률.
- `query_loss.weights`: `target`, `background`, `midpoint` 항의 정규화 비중.
- `background`: 기본 `{"kind":"empirical"}`은 선언된 배경에서 독립 재표집한다. Gaussian은 `standard_deviation`, `zero_probability`를 명시한다.

`[request,1]`의 관측 행렬은 dimension+1 열의 rank를 가져야 한다. 입구는 streaming Givens QR로 피팅한다. Readout은 온도 T에서 `H[:,y]=2×target[y]/T`, `b[y]=−||target[y]||²/T`다. 준비는 각 선언 context의 full/direct FP64 값 분포와 정규화의 반올림을 반경에 반영한다.

`train-value`는 `model`, `requests`, `prompts`, `values`, `labels`, `steps`, `learning_rate`와 공통 정책을 받는다. SEP hidden에서 값 head를 학습한다. 값은 UTF-8 문자열 또는 `{"hex":"ff00fe"}` 형태다.

## 원본 편입과 개정

Assembly는 `scene`, `instruction`, `request`, `query`, `lexica`, `a`, `b`, `catalog`와 추가 원본·경로·예산을 받는다.

```json
{
  "extra": {"archive": {"kind": "memory", "path": "archive/model"}},
  "goal_routes": [["a"], ["a", "b", "archive"]],
  "budgets": {"memory_bytes": 4294967296}
}
```

사실 원본의 kind는 `memory` 또는 `native`다. 각 경로의 첫 원본은 catalog 입력을, 다음 원본은 앞 원본의 출력 값을 받아야 한다. 목적별 확률 경로는 서로 다른 owner들로 구성한다. 연속 `read-weights` 경로는 owner를 반복 사용할 수 있다.

Host 예산 기본값은 Linux `MemAvailable`과 cgroup 잔여량의 작은 값이다. 사건 노드·살아 있는 어휘 분기·joint tensor는 자료구조의 byte 비용을 계산한다. 부족한 용량은 `Capacity` 오류로 나타내고, 부분 frontier는 미처리 질량을 유지한다.

`revise --stage <등록 이름>`은 해당 memory의 K/V를 로컬 복제본에서 학습한다. 개정 spec은 `labels`, `steps`, `learning_rate`와 공통 정책이다. 본체·입구·입출력 규격을 확인하고 현재 가중치의 반경으로 새 snapshot을 저장한다.

## 질의와 결과

일반 질의는 `question`, `people`을 받는다. 생략한 people은 catalog에서 질문에 등장한 이름으로 구성한다. 이야기 질의는 `story`와 `predicates`를 함께 준다. `modes`는 등록 원본 이름별 `full`·`direct` 선택이다. `full:true`는 전체 확률 분기 계산을 요청한다. `max_rows`, `max_components`로 처리할 frontier 양을 지정한다.

Joint는 `{"joint":["alice","alice","bob"],"goal":1}` 형식이다. 같은 요청의 반복은 같은 draw를 공유한다. Tensor의 차수는 joint 입력 수, shape는 최종 값 수에서 계산한다.

`read-weights`는 `inputs`, `route`, `modes`를 받는다. 첫 full 입력은 펼친 context, 첫 direct 입력은 request 좌표다. 다음 원본은 앞 출력의 단위 공 투영을 받는다. `--compare`는 같은 입력의 full 경로와 수치 차이를 계산한다.

답의 `values`, `probabilities`, `lower`, `upper`는 같은 순서이며 마지막 확률 항은 NONE이다. `read_error`는 준비된 직접 읽기의 TV 반경 누적, `numerical_error`는 실제 합성 연산량에서 계산한 반올림 여유, `remaining`은 미처리 질량이다. `tolerance`로 추가 판정 여유를 지정한다. `certified`, `prediction`, `reason`에 최빈 terminal 판정을 기록한다.
