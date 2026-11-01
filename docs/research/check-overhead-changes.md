# 검사 비용 정리 구현

2026-11-01. [실행 비용 감사](check-overhead-audit.md)의 F01–F38을 아래와 같이 변경했다. 감사 문서의 줄번호와 [위치 목록](check-overhead-sites.json)의 SHA-256은 변경 전 소스를 가리킨다. 아래 링크는 변경 후 구현을 가리킨다. RX 6600 실행 결과와 재개 대조 수치는 [측정 기록](../measurements/2026-11-01-check-pruning/README.md)에 담았다.

준비 단계는 파일·정책·배치·프로그램을 읽고 실행에 필요한 layout과 목적 계획을 만든다. 반복 실행은 그 계획과 봉인된 snapshot의 digest를 재사용한다. Clipping과 기울기 진단을 선택하면 실제 gradient norm을 계산하고, `benchmark`가 FLOPs 계측을 활성화한다.

| ID | 변경 방식 | 최종 구현 |
| --- | --- | --- |
| F01 | 삭제 | [GradientPlan](../../crates/sejong-gpu/src/parameter_vjp.rs#L497)의 FP32 최대 norm 제한과 component 상한을 제거했다. 요청한 norm은 FP64 실제 합산 gradient로 계산한다. |
| F02 | 옵션 계산 | [SGD](../../crates/sejong-train/src/control_variate.rs#L723)는 clipping·진단 시점에 `plan.norm`을 실행한다. 기본 갱신은 포장된 인수를 바로 소비한다. |
| F03 | 삭제 | [pack](../../crates/sejong-gpu/src/parameter_vjp.rs#L422)은 갱신·복원용 pointer, term, rank map만 만든다. Norm 전용 component 목록과 pointer 중복 제거 HashMap을 제거했다. |
| F04 | 삭제 | [parameter_gradient_update](../../python/sejong_kernels/ops/parameter.py#L265)는 gradient 합성과 SGD 저장을 수행한다. 갱신 원소의 finite scan, flag 초기화·reduction·host 회수를 제거했다. |
| F05 | 옵션·합산 | [Adam](../../crates/sejong-train/src/lib.rs#L199)과 dense SGD는 clipping·진단에서 [Engine.gradient_norm](../../crates/sejong-gpu/src/lib.rs#L490)을 호출한다. 모든 gradient의 norm을 하나의 launch와 한 번의 host 회수로 합친다. |
| F06 | GPU 요약 | [byte 학습](../../crates/sejong-train/src/lib.rs#L401)과 [SGD](../../crates/sejong-train/src/control_variate.rs#L564)는 관측별 평균 loss를 FP64 GPU summary에 기록하고 accumulation 끝에 회수한다. |
| F07 | 삭제 | [Adam](../../crates/sejong-train/src/lib.rs#L386)과 [SGD](../../crates/sejong-train/src/control_variate.rs#L502)는 최초 실행 경계와 step 종료에서 동기화한다. Step 시작의 반복 동기화를 제거했다. |
| F08 | 옵션 계산 | [CV 갱신](../../crates/sejong-train/src/control_variate.rs#L746)의 correction norm·cross term·진단용 mean target 합산은 clipping·진단 조건에서 실행한다. Predictor 학습과 실제 correction 갱신은 각 step에 수행한다. |
| F09 | 최종 경계 이동 | [query](../../crates/sejong-runtime/src/native/mod.rs#L666)는 부분 탐색에서 조기인증용 remaining·rounding을 계산한다. Full 탐색은 종료 시 최종 인증값을 계산한다. |
| F10 | 증분 계산 | [Frontier](../../crates/sejong-runtime/src/native/mod.rs#L123)는 task 삽입·삭제 때 실제 prefix 용량을 반영한 bytes를 갱신한다. Fact row cache도 삽입 때 저장량을 더한다. |
| F11 | 증분 계산 | [story](../../crates/sejong-runtime/src/native/mod.rs#L534)와 [lexical bind](../../crates/sejong-runtime/src/native/lexical.rs#L136)는 law·scene cache·clause·operator 저장량을 추가 시점에 누적한다. |
| F12 | 경계·전송 변경 | [signature](../../crates/sejong-model/src/native/mod.rs#L10)는 F32/F16 parameter를 F32로 회수한 뒤 CPU에서 기존 FP64 hash 표현으로 변환한다. 봉인 후 signature 조회는 저장한 digest를 반환한다. |
| F13 | 명시 실행으로 이동 | [Composite](../../crates/sejong-runtime/src/native/mod.rs#L318)의 전체 `verify`를 명시한 검증·저장 경계에 둔다. Query·weights·joint는 준비된 구성과 snapshot을 사용한다. |
| F14 | 삭제 | [NativeValue.row](../../crates/sejong-model/src/native/value.rs#L187)와 [Memory.row/read](../../crates/sejong-model/src/native/memory.rs#L314)는 seal을 확인하고 해당 행을 계산한다. 행마다 전체 parameter 재해시를 제거했다. |
| F15 | 공유·분리 | [Composite.verify](../../crates/sejong-runtime/src/native/mod.rs#L318)는 encoder allocation identity를 모아 공유 encoder를 한 번 검사한다. 각 classifier head는 `verify_head`로 검사한다. |
| F16 | 로드·교정 분리 | [Memory.load](../../crates/sejong-model/src/native/memory.rs#L650)는 저장 digest와 calibration metadata를 확인하고 반경을 복원한다. [calibrate](../../crates/sejong-model/src/native/memory.rs#L369)는 명시한 재교정 경로에서 full/direct 실행을 수행한다. |
| F17 | Digest 재사용 | [Memory.save](../../crates/sejong-model/src/native/memory.rs#L633), [NativeValue.save](../../crates/sejong-model/src/native/value.rs#L217), [Classifier.save](../../crates/sejong-model/src/native/classifier.rs#L1445)는 snapshot digest를 재사용하고 저장에 필요한 tensor를 회수한다. |
| F18 | 캐시 우선 | [BertEncoder.load_shared](../../crates/sejong-model/src/native/bert.rs#L147)는 context·정규화한 경로와 선언 digest로 공유 encoder를 먼저 찾는다. 첫 적재가 파일 읽기와 encoder 준비를 수행한다. |
| F19 | 메모리 캐시 | [Classifier.features](../../crates/sejong-model/src/native/classifier.rs#L670)는 최초 준비한 동결 feature를 `Rc`로 보관한다. Cache hit는 메모리 객체를 반환하고, 저장량이 host 예산을 넘으면 이전 항목을 제거한다. |
| F20 | 준비·할당 경계 이동 | [classifier host vector](../../crates/sejong-model/src/native/classifier.rs#L268)는 fallible reserve로 할당한다. 반복 `host_admit` OS 조회를 제거하고 runtime binding에는 요청에서 정한 budget을 전달한다. Feature cache는 최초 생성 때 host 예산을 읽는다. |
| F21 | 준비 경로 | [query_expand_prepared](../../crates/sejong-model/src/native/classifier.rs#L1345)는 query plan이 만든 root·action을 확장한다. 공개 `query_expand`가 외부 prefix를 확인하고 runtime 내부 탐색은 준비 경로를 사용한다. |
| F22 | Assembly 이동 | [Registry](../../crates/sejong-runtime/src/native/lexical.rs#L15)는 assembly에서 ABI·namespace와 lowercase lookup을 준비한다. [bind_prepared](../../crates/sejong-runtime/src/native/lexical.rs#L120)가 이를 요청 간 재사용한다. |
| F23 | 단일 순회·준비 경로 | [normalized](../../crates/sejong-runtime/src/native/resources.rs#L35)는 한 번 순회해 mass를 구한다. 내부 law·kernel은 prepared 실행으로 전달하고 공개 partition·lexical 입력 경계에서 분포를 검사한다. |
| F24 | 준비 경계 이동 | [prepare_parameter_vjp](../../crates/sejong-gpu/src/parameter_vjp.rs#L133)는 공개 입력과 policy를 확인한다. Graph 내부 공유 준비는 `prepare_parameter_vjp_reusing`을 사용하고 반복 policy·shape 대조를 제거했다. |
| F25 | 설정 경계 이동 | [TrainConfig](../../crates/sejong-core/src/lib.rs#L413)는 loss scale과 역수의 유한성을 확인한다. [ParameterGradient.accumulate](../../crates/sejong-gpu/src/parameter_vjp.rs#L31)의 parameter·term별 scale 반복 검사를 제거했다. |
| F26 | 삭제 | [pack](../../crates/sejong-gpu/src/parameter_vjp.rs#L422)은 내부 gradient layout에서 metadata를 작성한다. 각 term의 dtype·shape·scale 재검사를 제거했다. |
| F27 | 중복 경계 통합 | Graph gradient 합산과 [Engine.add](../../crates/sejong-gpu/src/lib.rs#L567)의 중복 shape 검사를 제거하고 `add_into`에서 입력 경계를 처리한다. Model hidden wrapper는 sequence를 한 번 확인하고, 생성은 sampling 정책을 진입 때 준비한다. |
| F28 | 학습 진입 이동 | [Corpus.batch](../../crates/sejong-train/src/lib.rs#L72)는 준비된 corpus/sequence로 표집한다. Trainer·benchmark가 corpus 길이를 진입 때 확인한다. |
| F29 | 목적 계획 | [PairPlan](../../crates/sejong-train/src/roles.rs#L265)과 [QueryObjective](../../crates/sejong-train/src/memory.rs#L444)가 loss 가중치를 학습 시작에 정규화해 저장한다. |
| F30 | 준비 경계 이동 | [QueryObjective.prepare](../../crates/sejong-train/src/memory.rs#L449)가 Gaussian 배경 정책을 한 번 확인한다. `query_context`는 준비된 표준편차와 zero 확률을 사용한다. |
| F31 | 자료 준비 이동 | [PairPlan.prepare](../../crates/sejong-train/src/roles.rs#L270)는 inventory·program 정합성, class 좌표와 대체 program을 준비한다. 학습 루프는 준비된 mapping·target을 사용한다. |
| F32 | 활성 목적 실행 | [query_update](../../crates/sejong-train/src/memory.rs#L503)는 양수 가중치에 필요한 alternate·midpoint view를 생성하고 미분한다. Pair consistency·contrast도 활성 목적에 따라 추가 평가와 seed를 구성한다. |
| F33 | 삭제 | [expert packing](../../crates/sejong-gpu/src/hierarchy_ops.rs#L356)은 실제 expert GEMM 행 수를 회수하고 `Tensor.rows`로 행을 선택한다. Count 합과 같은 count 범위의 중복 검사를 제거했다. |
| F34 | 옵션·닫힌식 | [Engine.enable_execution_metrics](../../crates/sejong-gpu/src/lib.rs#L484)가 실행 계측을 활성화한다. [attention FLOPs](../../crates/sejong-gpu/src/counters.rs#L174)와 parameter QR 비용은 위치·rank 순회 대신 닫힌식을 사용한다. |
| F35 | Kernel 캐시 | [launch](../../crates/sejong-gpu/src/lib.rs#L362)는 op·인자 signature·상수의 hash와 동등성으로 컴파일 결과를 재사용한다. Signature JSON은 cache miss에서 만들고 반복 grid 대조·재직렬화·중복 cache 조회를 제거했다. |
| F36 | 기록 주기 | [Adam](../../crates/sejong-train/src/lib.rs#L423)과 [SGD](../../crates/sejong-train/src/control_variate.rs#L836)는 첫 step·`log_every`·최종 step에 JSON을 만든다. `BufWriter`가 기록을 모으고 학습 종료에 flush한다. |
| F37 | 오류 분기 생성 | [ensure](../../crates/sejong-core/src/lib.rs#L26)는 `Display`를 받아 실패 시 문자열을 만든다. 동적 메시지는 `format_args!`를 사용하고 [GPU allocation](../../crates/sejong-gpu/src/hip.rs#L220)의 Capacity 정보도 실패 분기에서 만든다. |
| F38 | 최종 회수·입력 공유 | [benchmark decode](../../crates/sejong-cli/src/benchmark.rs#L134)는 마지막 cached logits를 회수해 full 결과와 비교한다. 학습은 미리 연 corpus를 공유하고 환경 정보는 Python/Triton/ROCm metadata에서 읽는다. |

`Memory`, `NativeValue`, `Classifier`의 `signature`는 봉인 digest를 반환하고 `verify`는 현재 parameter를 다시 읽는다. Classifier 학습 진입의 `open_training`과 Memory 교정·개정 진입은 seal을 열어 변경된 가중치로 새 digest를 만든다. Snapshot 저장은 이 생명주기를 사용한다.

일반 학습은 계측 활성 상태를 `execution.json`의 `metrics_collected`로 기록한다. 기울기 norm·FLOPs 계측을 선택한 시점에 해당 수치를 기록하며, 나머지 metrics 필드는 `null`을 사용한다. `benchmark`는 각 step의 계측과 기록을 명시적으로 활성화한다.
