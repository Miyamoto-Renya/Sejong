# Sejong v6 구조

Sejong의 native graph는 의미·질문·어휘·사실 원본과 연결 규격을 저장한다. 통합 artifact 형식은 `sejong-native-composite-v6`, 로컬 파서 형식은 `sejong-role-dynamic-v3`, 준비된 memory 형식은 `sejong-prepared-memory-v3`다.

## 파서와 학습

의미 파서는 공식 MiniLM 가중치와 WordPiece를 Rust·Triton으로 실행한다. 의미 입력의 개체에 등장 순서의 별칭을 부여하고 원래 좌표를 span으로 유지한다. 별칭은 이름 목록과 번호를 사용해 확장한다. Token offsets로 개체별 hidden을 pooling하고 유효 token의 평균을 문장 표현으로 사용한다. 입력 길이는 로드한 encoder의 위치 embedding 수를 따른다.

개체 수가 N일 때 scene은 N²+1, instruction은 (N+1)², request는 2(N+3)+1개의 의미를 가진다. 마지막 항목은 NONE이다. Head는 종류 활성도와 개체별 역할·목적·선택자를 분해해 이 분포를 구성한다. 투영 폭과 층 수는 모델 설정에 들어간다.

Query 원본은 가변 길이 입력의 relational encoder와 조건부 프로그램 decoder를 사용한다. 처음에 목적·anchor의 2N개 선택과 NONE을 채점한다. 각 접두사에서 술어별 방향·first/last 조합과 STOP을 채점한다. 등장한 개체와 술어를 mask에 반영하고, 질문의 술어 출현 횟수를 관계 단계의 문법 길이로 사용한다. 학습 레이블은 `Program { goal, anchor, steps }`이며 조건부 선택의 log 확률을 합산한다. 실행기는 접두사를 요청할 때 다음 선택을 계산한다.

어휘 원본은 가변 길이 byte 입력을 relational encoder로 처리하고 선언된 의미를 출력한다. 관계 attention은 거리의 방향·로그 크기와 같은 identity 여부의 학습 bias를 사용한다.

로컬 학습은 정답 CE, 같은 의미의 좌표 정렬 JS, 서로 다른 의미의 log 확률 차이에 대한 softplus 대비 손실을 사용한다. Query의 JS는 대응하는 접두사 조건부 분포에 적용한다. `LossPolicy`는 사용 중인 손실 항목의 선언 가중치를 합계 1로 정규화한다. `OptimizerPolicy`는 관측 수·배치·학습 계획 또는 명시한 예제 단위 기억 길이에서 Adam 계수를 계산한다. 배치는 tensor별 할당량과 사용 가능한 장치 메모리로 정한다.

## 사실 원본

`NativeValue`는 byte와 BOS·SEP·EOS를 구분하는 Transformer다. 요청 SEP 위치의 hidden과 학습한 value head에서 값 분포를 계산한다.

Memory는 query Transformer, affine 입구 C, K/V와 prototype readout을 가진다. 층·head·FFN 폭·정규화 epsilon·잔차 배율·readout 온도 T는 `MemoryConfig`가 지정한다.

```text
full:   q = QueryBody(context)
direct: q = [request, 1] × C
read:   h = GELU(q × K) × V
row:    softmax(h × H + b)
H[:, y] = 2 × target[y] / T
b[y] = −||target[y]||² / T
```

Query 학습은 목표 query, 배경을 바꾼 같은 요청의 일관성, midpoint affine 조건을 결합한다. 배경은 선언된 context의 경험적 분포 또는 명시한 Gaussian 정책에서 생성한다. 입구 C는 query 관측을 한 행씩 넣는 Givens QR로 피팅한다. QR의 보관 크기는 affine 좌표 차원으로 결정한다. 이어 K/V가 사실 목표 벡터를 학습한다.

## 의미·사건·원본 연결

의미 실행기는 목적·개체의 2N개 채널마다 80개 충분 상태를 유지한다. 채널당 scene 역할 일치, focus, 허용 여부와 관련 요청을 보관해 총 160N개 상태로 계산한다.

같은 어휘 owner와 phrase의 의미 선택은 사용 구간 동안 공유한다. 마지막 사용에서 선택 축을 합산해 제거한다. 살아 있는 선택들의 결합 상태와 연산 작업 공간을 바이트로 계산한다.

사건은 occurrence별 N²+NONE 분포를 가진다. 질문에 등장한 술어의 사건을 모으고 같은 신경망 입력을 캐시한다. 서로 다른 술어를 지나는 프로그램은 scan kernel을 곱한다. 같은 술어로 돌아오는 프로그램은 ordered decision diagram에서 기존 사건 선택을 재사용한다.

`Assembly`는 기본 원본 `a`, `b`와 이름을 붙인 `extra` 원본들을 등록한다. 두 의미 목적의 `goal_routes`가 실제 원본 경로를 지정하며 기본 경로는 `[a]`, `[a,b]`다. 확률 경로의 각 원본 ID는 경로 안에서 고유하다. 첫 원본은 catalog를 입력으로 받고, 다음 원본은 앞 원본의 모든 값을 입력으로 받는다. 연속 입력용 `WeightRequest.route`는 등록된 원본들을 지정한 순서로 연결하며 같은 원본을 반복할 수 있다.

여러 요청의 공동 법칙은 equality partition별 tensor 축 수축으로 전달한다. 같은 원본의 같은 행은 하나의 draw를 공유한다. 서로 다른 행과 원본은 독립 선택을 가진다. Partition은 입력 행 수에 맞춰 필요한 항목을 순차 생성한다. `pullback`은 최종 관측 함수를 입력 좌표로 옮긴다. Tensor 크기는 각 축의 값 수와 요청 차수로 계산한다.

## 자원과 확률 판정

호스트 예산은 `MemAvailable`과 cgroup 잔여량 중 작은 값에서 정한다. `Budgets.memory_bytes`를 지정하면 그 범위 안에서 사용한다. 실행기는 보관할 벡터·접두사·행 캐시·사건 그래프와 작업 공간의 바이트 수를 할당 전에 계산한다. 부분 질의는 예산에 도달한 접두사와 행의 질량을 `remaining`에 유지한다. 전체 실행의 용량 오류에는 필요한 바이트와 사용 가능한 바이트가 들어간다.

Memory의 경험적 보정은 선언된 요청마다 추론과 같은 FP64 `forward_row`로 full과 direct를 실행한다. Query 차이의 norm과 두 행 법칙의 total variation을 방향 반올림으로 계산한다. `row_radius` ρ는 이 선언된 FP64 full 행 법칙에 대한 direct 오차의 보정값이다. 저장된 값은 원본 snapshot에 포함되며 로드 시 같은 행을 다시 실행해 대조한다.

도달 질량 w의 direct 행을 읽으면 `E ← min(1,E+wρ)`로 읽기 오차를 누적한다. `sejong-core::numeric`는 IEEE-754 정밀도, 누적 연산 수와 절댓값 합에서 `numerical_error` δ를 계산한다. 의미 상태 전이·선택 축 병합·사건·frontier 계산이 이 값에 반영된다.

계산한 terminal 질량 ℓ와 남은 질량 r의 구간은 `[max(0,ℓ−E−δ), min(1,ℓ+r+E+δ)]`다. 일등·이등 질량 차이가 `r+2E+2δ+τ`를 넘으면 최빈 답을 판정한다. τ는 사용자가 지정한 추가 tolerance다. NONE은 별도 terminal이다.

공동 법칙은 고유 행 집합 U마다 `1−∏(1−ρᵣ)`를 계산하고 입력 질량으로 가중한다. 같은 행의 반복은 한 항으로 들어간다. 연속 경로의 full/direct 대조는 같은 입력의 두 실행 결과와 최대 절대 차이를 반환한다.

## Snapshot과 개정

봉인 서명에는 실제 가중치·입출력·context·입구·readout·보정값·원본 이름·경로·예산이 들어간다. 질의 입구에서 현재 서명을 대조한다. 개정은 원본을 로컬 복제해 K/V를 학습하고, 본체·입구·입출력·readout 계약을 확인한 뒤 보정값을 다시 계산해 편입한다.

Checkpoint는 phase·optimizer moments·RNG·실제 배치와 자료·학습 정책 fingerprint를 저장한다. Artifact는 safetensors와 JSON을 전용 디렉터리에 담는다. 저장은 staging 파일을 완성한 뒤 목적지로 rename하며 JSON은 FP64 왕복 정밀도를 보존한다.
