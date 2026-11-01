# 검사·검증 실행 비용 감사

이 문서의 줄번호·SHA-256과 비용 경로는 검사 비용 정리 전 소스를 가리킨다. 변경 후 동작은 [F01–F38 구현 대응표](check-overhead-changes.md), 실제 실행 수치는 [후속 측정](../measurements/2026-11-01-check-pruning/README.md)에 기록했다.

2026-11-01. 현재 작업 트리의 Rust·Python 소스 64개에서 검사·검증·전송·크기 계산 관련 위치 871개를 수집하고, 성능 후보를 38개 묶음으로 추렸다. [전체 위치 목록](check-overhead-sites.json)은 파일·줄·검색 종류·소스 SHA-256을 담는다. 위치 목록의 항목은 소스 탐색 후보이며, 아래 표가 실제 호출 경로와 활성 조건을 분류한다.

기본 구성은 `sgd + parameter_vjp`, `clip_enabled=false`, `gradient_diagnostics_every=0`, accumulation 8이다. 아래 분석은 소스의 호출 관계, 저장된 모델 header와 기존 측정 자료를 사용한다. 이 감사의 산출물은 후보 목록과 수학적 반례다.

## 먼저 처리할 항목

| ID | 위치 | 활성 경로·실행 비용 | 발견 내용·조치 후보 |
| --- | --- | --- | --- |
| F01 | [parameter_vjp.rs:655](../../crates/sejong-gpu/src/parameter_vjp.rs#L655) | 기본 SGD, parameter별 검사 | FP64 Frobenius 상한에 `f32::MAX`를 적용한다. 원소와 갱신값이 모두 FP32 finite인 행렬을 거절하는 조건이다. 아래 반례에 따라 이 cap을 제거할 수 있다. |
| F02 | [parameter_vjp.rs:612](../../crates/sejong-gpu/src/parameter_vjp.rs#L612), [호출:720](../../crates/sejong-train/src/control_variate.rs#L720) | 기본 SGD 매 step, component 전체 FP64 norm·D2H·동기화 | clipping과 diagnostics가 꺼져 있어도 gradient norm 상한을 계산한다. 이때 값은 finite 감지와 로그에 사용한다. 갱신 결과의 finite flag와 목적을 통합하고 norm은 요청한 경로에서 계산하는 후보. 정확 norm을 요청한 때도 component norm·상한을 먼저 계산한 뒤 전체 gradient norm을 다시 계산한다. |
| F03 | [materialize:384](../../crates/sejong-gpu/src/parameter_vjp.rs#L384), [pack:449](../../crates/sejong-gpu/src/parameter_vjp.rs#L449) | NativeValue의 parameter VJP·Adam, 매 step | `pack`이 norm 전용 pointer-dedup HashMap·components·길이·descriptors를 만들고 GPU metadata에 붙인다. `GradientMaterializer`는 `tiles(MATERIALIZE)`만 실행해 이 norm metadata의 소비처가 없다. 갱신 descriptor와 norm descriptor를 분리할 직접 후보. |
| F04 | [parameter.py:282](../../python/sejong_kernels/ops/parameter.py#L282), [update:687](../../crates/sejong-gpu/src/parameter_vjp.rs#L687) | 기본 SGD 매 step, 모든 갱신 원소의 abs·finite 비교·타일 reduction, flag 초기화와 host 회수 | 실제 갱신 결과를 검사한다. F02와 finite 감지 목적이 겹친다. flag를 step 요약과 함께 회수하고 검사 정책을 한곳에 모으는 후보. 선행 검사를 줄이면 오류 검출 시점과 오류가 난 step의 가중치 상태를 함께 정해야 한다. |
| F05 | [Adam:210](../../crates/sejong-train/src/lib.rs#L210), [dense SGD:710](../../crates/sejong-train/src/control_variate.rs#L710) | Adam·기존 dense SGD, 각 gradient마다 `squared_norm→to_f32` | clipping이 꺼져도 모든 gradient를 순회하며 tensor별 reduction·동기화·D2H를 수행한다. NativeValue·roles·memory Adam의 공통 경로다. clipping·진단과 경량 finite 감지를 분리하고 global reduction으로 합칠 후보. 현재 기본 compact SGD는 F02를 사용한다. |
| F06 | [SGD loss:552](../../crates/sejong-train/src/control_variate.rs#L552), [Adam loss:369](../../crates/sejong-train/src/lib.rs#L369) | 기본 step당 8회, 매회 512개 loss F32를 D2H | CPU에서 평균과 finite 검사를 한다. backward seed는 별도로 구성한다. GPU에서 loss 합계·finite를 축약하고 step 끝에 한 번 회수하는 후보. |
| F07 | [SGD:493](../../crates/sejong-train/src/control_variate.rs#L493), [SGD:823](../../crates/sejong-train/src/control_variate.rs#L823), [Adam:356](../../crates/sejong-train/src/lib.rs#L356) | 기본 학습, step 종료 sync 다음의 다음 step 시작 sync | 완료된 동일 stream을 연속 동기화한다. 최초 실행 경계와 step 종료 경계로 정리할 후보. |
| F08 | [CV norm:365](../../crates/sejong-train/src/control_variate.rs#L365), [호출:738](../../crates/sejong-train/src/control_variate.rs#L738) | CV 프로파일의 block마다 매 step, 2 GEMM+2 dot, dot 결과 host 회수 2회 | 교정 성분과의 cross term을 포함한 전체 norm이다. clipping·diagnostics가 꺼진 경로에서는 수치 감지용 경량 경로로 분리할 후보. 기본 구성의 CV는 비활성이다. |
| F09 | [query:628](../../crates/sejong-runtime/src/native/mod.rs#L628), [final:864](../../crates/sejong-runtime/src/native/mod.rs#L864) | Native `q.full=true`, frontier 매 반복 | `remaining=sum(pending)`과 `rounding_error(known,pending)`를 계산한다. 루프 내 사용처는 `!q.full` 조기인증 분기다. full 모드의 사용되지 않는 반복 계산을 최종 1회로 이동할 직접 후보. |
| F10 | [live_bytes:1304](../../crates/sejong-runtime/src/native/mod.rs#L1304), [호출:651](../../crates/sejong-runtime/src/native/mod.rs#L651) | Native frontier 매 반복에 pending과 cached law 전체 순회 | T회 탐색에서 frontier·cache가 성장하면 누적 O(T²) accounting이 된다. 삽입·삭제·할당 시 증분 byte 수를 갱신하고 비교에는 저장한 합계를 사용하는 후보. |
| F11 | [story:511](../../crates/sejong-runtime/src/native/mod.rs#L511), [lexical:171](../../crates/sejong-runtime/src/native/lexical.rs#L171) | event·clause 추가마다 기존 저장량 전체 재합산 | 기존 laws·scene cache, clauses·operator cache를 매번 다시 센다. event·clause 수에 대한 이차 순회 부분을 증분 accounting으로 바꾸는 후보. |

## 전체 모델 해시와 로드·추론 경계

| ID | 위치 | 활성 경로·실행 비용 | 발견 내용·조치 후보 |
| --- | --- | --- | --- |
| F12 | [signature:10](../../crates/sejong-model/src/native/mod.rs#L10), [to_f64:283](../../crates/sejong-gpu/src/lib.rs#L283) | P개 원소·K개 tensor의 검증 1회마다 FP64 host 복사 8P bytes, 동기화·D2H K회 | metadata JSON과 모든 parameter를 해시한다. FP32 tensor는 FP64 변환 kernel도 실행한다. load·revision·save 경계의 인증과 추론용 검증된 handle을 분리할 최우선 추론 후보. |
| F13 | [Composite verify:301](../../crates/sejong-runtime/src/native/mod.rs#L301), [current_signatures:269](../../crates/sejong-runtime/src/native/mod.rs#L269) | `query`, `weights`, `joint`의 요청 시작 | fact별 `verify→signature` 뒤 `current_signatures`에서 같은 fact를 다시 해시한다. 요청 경계의 fact 전체 해시가 2회다. digest 한 번을 seal과 assembly 계약 대조에 재사용할 후보. 요청 경로 밖의 fact도 검사한다. |
| F14 | [NativeValue row:178](../../crates/sejong-model/src/native/value.rs#L178), [Memory row:311](../../crates/sejong-model/src/native/memory.rs#L311) | 실제 row 조회마다 `verify→signature` | 요청 안에서 같은 fact의 서로 다른 R개 row를 읽으면 F13을 합쳐 전체 해시가 `2+R`회가 된다. 요청 시작에 인증한 read session을 row 조회에 전달할 후보. |
| F15 | [Classifier signature:1357](../../crates/sejong-model/src/native/classifier.rs#L1357), [Bert verify:440](../../crates/sejong-model/src/native/bert.rs#L440) | 모든 classifier의 signature마다 frozen encoder 전체 재해시 | 동일 `Rc<BertEncoder>`를 공유하는 head들이 encoder를 각각 다시 검사한다. encoder identity의 dedup과 head/encoder digest 분리 후보. |
| F16 | [Memory load:674](../../crates/sejong-model/src/native/memory.rs#L674), [calibration:385](../../crates/sejong-model/src/native/memory.rs#L385) | load마다 선언 요청 R개의 Full forward R회+Direct forward R회, query·logit 회수와 TV 계산 | 저장 signature 대조 뒤 전체 calibration을 재실행하고 다시 signature를 만든다. 인증된 bounds를 로드하는 경로와 명시적 재교정 경로를 분리할 후보. calibration의 실제 산출물은 `query_radius`, `row_radius`다. |
| F17 | [Memory save:617](../../crates/sejong-model/src/native/memory.rs#L617), [Value save:208](../../crates/sejong-model/src/native/value.rs#L208), [Classifier save:1372](../../crates/sejong-model/src/native/classifier.rs#L1372) | seal·verify·save의 중첩 호출 | 읽기만 수행하는 저장 함수 안에서 같은 signature를 반복 계산한다. 한 snapshot의 digest 재사용, 저장용 D2H와 hash용 D2H 통합 후보. |
| F18 | [Bert load_shared:147](../../crates/sejong-model/src/native/bert.rs#L147) | encoder cache hit도 config·tokenizer·전체 safetensors 파일 읽기·해시 | 같은 encoder를 여러 head가 로드할 때 동일 파일을 반복 스캔한다. 이미 인증한 artifact/encoder handle을 loader에 전달할 후보. |
| F19 | [Classifier features:657](../../crates/sejong-model/src/native/classifier.rs#L657) | frozen feature cache hit마다 JSON key·hash·파일 읽기·parse·전체 finite scan | 같은 학습 example이 반복되어도 disk cache에서 다시 읽고 검사한다. 최초 검증한 Features를 메모리 cache로 재사용할 후보. |
| F20 | [host_admit:260](../../crates/sejong-model/src/native/classifier.rs#L260), [host_available:185](../../crates/sejong-core/src/native.rs#L185) | encoder·root·transition마다 `/proc/meminfo`와 cgroup 파일 2개 읽기·parse | 요청·배치에서 확보한 host budget을 전달하고 실제 할당은 fallible allocation으로 처리할 후보. `Budgets::resolve`도 같은 OS 조회를 반복한다. |
| F21 | [Classifier prefix:1282](../../crates/sejong-model/src/native/classifier.rs#L1282) | prefix 확장마다 root/context와 전체 steps를 다시 검사 | 길이 D 경로의 검사 합계가 O(D²)다. 검증된 prefix와 새 step의 내부 확장, 외부 Program 검증 경계를 구분할 후보. |
| F22 | [lexical bind:76](../../crates/sejong-runtime/src/native/lexical.rs#L76), [assemble:176](../../crates/sejong-runtime/src/native/mod.rs#L176) | query마다 Use/Ignore ABI·namespace 검사와 lowercase registry 재구축 | assemble에서 같은 metadata의 ABI와 namespace를 이미 확인한다. compiled registry를 assembly에 보관할 후보. |
| F23 | [normalized:35](../../crates/sejong-runtime/src/native/resources.rs#L35), [partition:124](../../crates/sejong-runtime/src/native/partition.rs#L124), [lexical:213](../../crates/sejong-runtime/src/native/lexical.rs#L213) | law/kernel의 각 row·prior를 생성 뒤 실행 경계에서 재검사 | normalized가 mass·finite/nonnegative·tolerance용 absolute sum을 여러 번 순회한다. 인증한 Law/Kernel 타입을 내부에 전달하고 외부 입력 검증과 분리할 후보. |

F12–F15와 F17은 같은 서명 비용이 발생하는 서로 다른 호출 계층이다. 저장된 실제 NativeValue artifact는 서명 대상 201 tensors·26,472,736개 원소다. 전체 해시 한 번의 FP64 D2H는 **211,781,888 bytes, 약 201.97 MiB**다. 한 fact에서 R개 row를 읽는 요청의 해당 fact 해시 D2H는 **`8P(2+R)` bytes**이며, metadata 직렬화가 추가된다.

`Memory.parameters/metadata/port`, `NativeValue.body/embedding/head/bias/metadata`, `BertEncoder.parameters/config/tokenizer_json`은 공개 필드다. Tensor clone은 allocation을 공유하고 Engine 갱신 API가 내용을 바꾼다. 요청 간 digest cache는 **검증된 immutable snapshot 또는 generation을 추적하는 sealed handle**과 함께 설계해야 변경 검출 계약을 보존한다. 요청·저장 함수 하나 안에서 동일 digest를 재사용하는 F13·F17은 해당 읽기 흐름의 중복을 직접 합칠 수 있다.

## 반복되는 정적 검사와 작은 중복

| ID | 위치 | 호출·비용 | 조치 후보 |
| --- | --- | --- | --- |
| F24 | [Graph config:97](../../crates/sejong-gpu/src/graph.rs#L97), [prepare:156](../../crates/sejong-gpu/src/parameter_vjp.rs#L156) | 같은 parameter VJP policy를 Graph 구성과 투영 준비마다 재검사 | 학습 진입에 검증한 policy를 내부 경로에 전달. 공개 API의 최초 검증과 분리. |
| F25 | [gradient accumulate:33](../../crates/sejong-gpu/src/parameter_vjp.rs#L33) | 기본 200 parameters×accumulation 8=1,600회/step 같은 scale의 finite 검사, term별 재검사 | 관측당 `control_weight/loss_scale`을 한 번 검증하고 내부에 전달. 임의 scale을 받는 공개 API의 계약은 별도로 유지. |
| F26 | [pack:459](../../crates/sejong-gpu/src/parameter_vjp.rs#L459), [pack:487](../../crates/sejong-gpu/src/parameter_vjp.rs#L487), [pack:508](../../crates/sejong-gpu/src/parameter_vjp.rs#L508) | parameter·term마다 dtype/shape/scale 확인, pointer dedup 앞의 dtype 재검사 | 검증한 gradient layout에 동적 pointer·scale을 채우는 구조로 분리. component 재사용 때 같은 dtype을 다시 확인하는 부분도 합칠 후보. |
| F27 | [Graph accumulate:489](../../crates/sejong-gpu/src/graph.rs#L489), [Engine add:483](../../crates/sejong-gpu/src/lib.rs#L483), [add_into:489](../../crates/sejong-gpu/src/lib.rs#L489) | gradient 병합에서 같은 shape 계약을 Graph→add→add_into가 확인 | 출력 y는 바로 x.shape로 생성한다. prepared shape 연산과 공개 입력 경계를 나누는 후보. hierarchical hidden·encode의 sequence 검사, 토큰마다 불변 Sampling 옵션 검사도 같은 부류다. |
| F28 | [Corpus batch:78](../../crates/sejong-train/src/lib.rs#L78) | 기본 step당 8회 같은 corpus 길이 조건 검사 | trainer 진입에서 검증한 corpus/sequence 조합을 내부 batch 경로에 전달. |
| F29 | [query loss:513](../../crates/sejong-train/src/memory.rs#L513), [pair loss:263](../../crates/sejong-train/src/roles.rs#L263) | 매 query step·선택 pair마다 같은 loss weights를 Vec·BTreeMap 조회·합계·finite 검사로 정규화 | 학습 시작에 목적 조합별 weights와 active loss 계획을 준비. |
| F30 | [Gaussian background:448](../../crates/sejong-train/src/memory.rs#L448) | query 학습에서 같은 정책 범위를 3×batch회/step 검사 | 검증한 background sampler를 학습 진입에 준비. |
| F31 | [pair remap:58](../../crates/sejong-train/src/roles.rs#L58), [pair:261](../../crates/sejong-train/src/roles.rs#L261), [program:286](../../crates/sejong-train/src/roles.rs#L286) | 고정 pair를 뽑을 때마다 inventory·program 정합성, permutation과 class 범위를 다시 계산·검사 | remap의 선형 검색은 O(n²+p²), prefix/program 검사와 encoded 재계산도 반복한다. 자료 준비 시 compiled supervision과 검증된 mapping을 저장. |
| F32 | [midpoint:487](../../crates/sejong-train/src/memory.rs#L487), [pair consistency:284](../../crates/sejong-train/src/roles.rs#L284) | 해당 loss weight가 0인 설정에서도 입력 생성·추가 forward·D2H·zero-seed backward 실행 | active loss 계획으로 0 가중치 목적의 계산·검사 하위 경로를 통째로 생략. 기본 균등 loss는 양수 항을 사용한다. |
| F33 | [expert counts:355](../../crates/sejong-gpu/src/hierarchy_ops.rs#L355), [count range:361](../../crates/sejong-gpu/src/hierarchy_ops.rs#L361) | packing마다 합 O(E)·개별 범위 검사 뒤 Tensor.rows가 같은 범위를 재검사 | K개의 유효 ID가 주어지면 count 합은 항등식이다. NaN/sentinel 때문에 누락된 slot을 검출하는 역할이 있어 진단 분리 후보다. 개별 count 범위는 slice 경계와 합칠 수 있다. counts D2H는 실제 expert GEMM 행 수 결정에 사용한다. |

이 표의 scalar·shape 검사는 CPU 비교 몇 회 수준이다. 전수 scan·D2H·전체 해시·OS 파일 읽기 뒤에 처리할 순서다. 공개 Tensor shape/dtype도 변경할 수 있으므로 내부 검사를 준비 단계로 옮길 때 검증된 metadata 경계가 필요하다.

## 검사 주변의 상시 관측·준비 비용

| ID | 위치 | 호출·비용 | 조치 후보 |
| --- | --- | --- | --- |
| F34 | [count_execution:151](../../crates/sejong-gpu/src/counters.rs#L151), [position loop:182](../../crates/sejong-gpu/src/counters.rs#L182) | attention launch마다 T 길이 CPU 순회. 기본 352 launches에서 합계 65,536 position/step | FLOPs 계측용 계산이다. 형상별 사전 계산·등차수열 닫힌식·진단 빌드 분리 후보. 모든 launch의 인자 탐색과 parameter update의 FLOPs 재합산도 같은 계측 경로다. |
| F35 | [launch:349](../../crates/sejong-gpu/src/lib.rs#L349), [JSON key:353](../../crates/sejong-gpu/src/lib.rs#L353) | 기본 약 14,254 launches/step마다 grid 검사, signature String/JSON, 직렬화, rejected/module cache 조회 | 준비된 kernel handle과 launch 계약을 재사용할 후보. JSON·cache 준비는 검사 주변의 dispatch 비용이다. |
| F36 | [SGD log:852](../../crates/sejong-train/src/control_variate.rs#L852), [Adam log:398](../../crates/sejong-train/src/lib.rs#L398) | `log_every=10`이어도 metrics JSON과 flush는 매 step | 버퍼링·기록 cadence와 flush cadence를 적용할 후보. |
| F37 | [ensure:26](../../crates/sejong-core/src/lib.rs#L26), [allocation:218](../../crates/sejong-gpu/src/hip.rs#L218) | `ensure`에 `format!`을 넘긴 곳은 성공 경로에서도 문자열을 먼저 만든다 | 오류 메시지 지연 생성 후보. 대표 warm allocation의 pool hit는 해당 format 경로 앞에서 복귀하므로 cold allocation 위치와 구분한다. 고정 문자열 ensure의 String 변환은 오류 분기에서 실행한다. |
| F38 | [benchmark decode:134](../../crates/sejong-cli/src/benchmark.rs#L134), [D2H:151](../../crates/sejong-cli/src/benchmark.rs#L151) | teacher-forced benchmark에서 warmup 포함 2×decode회 전체 logits D2H, 사용되는 full/cached 대조는 마지막 값 | 최종 결과 회수로 축소할 후보. 토큰 전후 sync와 download 내부 sync도 정리 대상. source/executable hash·환경 패키지 순회와 corpus 재해시는 benchmark 준비 비용이다. |

## 수학적 반례와 규모

F01에서 `M=f32::MAX`, dense `G∈R⁴ˣ⁴`, 모든 `Gᵢⱼ=M/2`, W=0, decay=0, lr=2⁻¹⁰으로 두면:

- FP64 Frobenius norm은 `2M=6.805646932770577×10³⁸`으로 유한하다.
- 각 SGD 결과는 `−M/2048=−1.6615348956959417×10³⁵`로 FP32에서 유한하다.
- 현재 `bound≤M`은 이 갱신을 거절한다.

[반례 수치](check-overhead-counterexample.json)는 Python의 IEEE-754 변환과 FP64 계산으로 저장했다. 이 조건은 norm의 저장 형식과 개별 갱신 원소의 범위를 혼동한다.

기본 저장 모델은 200 tensors·26,488,192개 원소다. 압축된 20개 weights의 원소 수는 589,824개다. 나머지 dense gradient가 모두 관측된 step에서 F02의 scan은 다음과 같다.

| 입력 | F32 원소 수 |
| --- | ---: |
| Dense gradient | 25,898,368 |
| 공유 left | 131,072 |
| Right | 589,824 |
| Tail | 1,179,648 |
| 합 | **27,798,912, 약 106.04 MiB/step** |

고유 component는 564개이며 norm 결과 회수는 4,512 bytes다. 실제 scan 크기는 `gradient=None`과 pointer 중복 구성에 따라 달라진다. F04 갱신 grid는 52,644 tiles·26,953,728 padded lanes다. F05의 200개 gradient가 모두 존재하면 norm만으로 최대 200번 reduction·200번 host 동기화/회수가 발생한다.

[기존 3-step 계측](../measurements/2026-11-01-parameter-vjp-overhead/initial-profile.json)의 초기 압축 경로에서 `parameter_component_norm` HIP event 구간 합은 5.337ms였다. 이 값은 당시 세 step의 kernel 구간이며 현재 구현 전체의 제거 후 시간은 해당 경로를 변경한 측정으로 구한다.

## 일반 실행과 분리된 검증 및 알고리즘 계약

- `native/checks.rs`의 oracle·전수 확률 비교는 [cfg(test)](../../crates/sejong-runtime/src/native/mod.rs#L1363) 아래에 있다. `examples/*check`, `*contract`, 연구 스크립트는 명시한 실행 진입점에서 작동한다. 기본 학습·추론의 반복 비용은 위 표의 production 호출에서 발생한다.
- `kernel-profile`의 HIP event 수집은 Cargo feature로 분리되어 있다. 기본 release는 일반 launch 경로를 사용한다.
- `tl.debug_barrier`는 이름과 함께 실제 CTA 동기화를 수행한다. 표집 scratch와 streaming QR의 store/load 순서를 지키는 역할이 있다.
- Triton의 padding mask와 인덱스 경계, rejection sampling의 범위 조건, QR의 작은 방향 판정은 각각 주소·균등 표집·수치 알고리즘에 사용한다.
- equality partitions·distinct 처리와 event MDD의 재진입 분기는 같은 row draw의 상관을 계산한다. semantic mass·outward rounding·error-bound 생성은 해당 출력의 수학적 계약이다.
- `roles.consistency`의 `is_finite`는 log-probability −∞, 즉 확률 0의 Jensen–Shannon 항을 처리한다.
- 처음 받는 파일·프로그램·shape·dtype·allocation 크기·RNG 상태·checkpoint 설정은 입력 경계에서 확인한다. 내부 반복은 검증된 handle과 layout으로 줄일 수 있다.

수정 순서는 F01·F03·F09의 과한 조건/미사용 계산, F02·F05와 F12–F15의 큰 검사, F10·F11·F20·F21의 반복 순회, F24–F38의 정적 계약·관측 비용 순으로 잡을 수 있다.
