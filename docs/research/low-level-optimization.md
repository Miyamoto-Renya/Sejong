# RX 6600 저수준 최적화 조사

2026-11-01. 검사 비용 정리 후 Sejong 본체의 실제 학습·HIP event·Triton IR·gfx1032 ISA를 조사했다. 현재 기본 본체는 FP32 master 200개·26,488,192 parameters, FP16 계산, sequence 512 × accumulation 8이다. [측정 원본](../measurements/2026-11-01-low-level/README.md)에 전체 모델 비교와 독립 GEMM 실험을 담았다.

전체 모델에서 global 활성값을 보존하는 실행을 먼저 비교했다. 같은 자료·seed·optimizer·66-step 일정에서 평균 시간은 **259.633→217.728ms/step**, 주요 issued FLOPs는 **107.186→88.484GF/step**, peak 메모리는 **472→483MiB**였다. 시간은 16.14%, 주요 FLOPs는 17.45% 감소했다. 최종 검증 BPB는 두 실행 모두 7.9348758879였고, 저장된 전체 가중치의 최대 절대차는 3.73×10⁻⁹였다. 이 실험은 `training.activation_checkpointing=false`로 기존의 활성값 보존 경로를 사용했다.

별도 3-step HIP event 계측은 총 43,005 launches를 기록했다. 연산별 event 구간의 비중은 GEMM 57.4%, add 11.4%, copy 6.46%, RMSNorm backward 4.51%였다. 이 계측의 각 launch에는 event 기록이 붙으며, 구간에는 host 제출 간격도 들어간다. 실제 학습 시간 비교에는 일반 release 빌드의 64-step 측정값을 사용했다.

| 우선 | 대상·현재 근거 | 구체적인 교체 방향 | 줄이는 비용·적용 범위 |
| --- | --- | --- | --- |
| 1 | [global checkpoint](../../crates/sejong-model/src/architecture.rs#L294)가 forward를 backward에서 재실행한다. 현재 본체의 활성값 보존은 peak 483MiB에서 실행됐다. | 실제 tensor 수명·allocator 크기·GPU budget으로 보존할 활성값과 재계산할 block을 정한다. 저장 byte당 재계산 비용을 실행 계획에 연결한다. | 현재 형상의 전체 모델 실측: 18.703GF/step과 41.906ms/step 감소. 큰 형상은 보존·재계산을 block별로 선택한다. |
| 2 | [parameter binding](../../crates/sejong-model/src/lib.rs#L214)이 관측마다 모든 master를 F16으로 변환한다. 기본 200 tensors × 8 = 1,600 casts/step이다. | Step 동안 같은 FP16 mirror를 공유하고, optimizer가 master와 mirror를 함께 저장한다. Graph에는 master leaf와 준비된 cast view의 미분 연결을 둔다. | Step 공유만으로 1,400 launches와 raw read/write 1,112,504,064 bytes/step 감소 후보. Mirror 크기는 52,976,384 bytes다. GPU cache의 실제 traffic은 별도 측정한다. |
| 3 | [GEMM](../../python/sejong_kernels/ops/kernels.py#L35)이 F16 입력도 F32로 풀어 IEEE dot를 실행한다. | F16×F16 operand를 유지하고 FP32에 누산해 gfx1032 packed dot2를 사용한다. Mixed FP32 operand는 해당 정밀도 경로로 실행한다. | 실제 대표 GEMM 3개에서 중앙값 1.235–1.327×. 수학적 GEMM FLOPs는 같고 명령·변환·LDS 사용이 바뀐다. |
| 4 | [gradient accumulate](../../crates/sejong-gpu/src/graph.rs#L485)가 GEMM 결과를 새 tensor에 다시 더한다. 전체 profile의 add는 7,405 launches다. | Gradient buffer의 소비 횟수·alias를 준비하고 GEMM destination과 beta epilogue로 직접 누적한다. | 별도 add·중간 tensor·메모리 왕복. Residual의 두 부모가 같은 dy allocation을 공유하는 관계를 수명 계획에 반영한다. |
| 5 | [expert routing](../../crates/sejong-gpu/src/hierarchy_ops.rs#L329)이 counts를 host로 읽어 GEMM 행 수를 정한다. 기본 checkpoint 구성은 128회/step이다. | GPU counts·offsets·tile queue를 grouped GEMM이 읽게 한다. Checkpoint를 쓰는 block에는 route ID·packing metadata를 보존한다. | D2H 대기와 작은 GEMM dispatch. 실제 선택된 행의 tile만 queue에 넣어 capacity padding의 연산 증가를 제어한다. |
| 6 | [upload](../../crates/sejong-gpu/src/lib.rs#L273)가 stream 대기 뒤 동기 복사를 한다. 입력·라벨만 최소 16회/step이고, 라벨은 encoder 실행 뒤 업로드한다. | Step의 입력·라벨 32KiB를 pinned staging에 함께 준비해 같은 stream에 비동기로 올린다. 다음 step은 별도 staging buffer에 준비한다. | Host/GPU 파이프라인 단절. Staging 수명은 복사 완료 event에 연결한다. |
| 7 | [RMSNorm backward](../../python/sejong_kernels/ops/kernels.py#L61)가 모든 token에서 같은 dW 주소에 atomic add한다. | Row tile에서 partial dW를 축약하고 채널별로 합산한다. Residual add와 norm 입력 준비도 결합한다. | Atomic 경합·중간 memory traffic. Tile과 한 번/두 번 reduction 선택은 실제 shape·register 예산·실측 시간으로 정한다. |
| 8 | [attention dKV](../../python/sejong_kernels/ops/hierarchy.py#L99)가 key마다 같은 query의 `sum(DO*O)`를 다시 계산한다. | 먼저 실행하는 dQ가 Delta를 저장하고 dKV가 읽는다. Query tile과 GQA group이 K/V tile을 공유하도록 확장한다. | 기본 유효 pair 기준 약 0.602G scalar 연산/step과 반복 O load 절감 후보. Delta 공유는 기존 두 launch에 연결한다. |
| 9 | [launch 인자](../../crates/sejong-gpu/src/lib.rs#L187)가 Tensor를 clone하고, launch마다 값·pointer Vec를 만든다. | Borrowed pointer argument와 재사용 저장소, 준비된 kernel handle·grid를 사용한다. | 약 14,260 launches/step의 shape heap 복사·Rc·Vec 할당과 상수 hash. |
| 10 | [allocator](../../crates/sejong-gpu/src/hip.rs#L204)가 free allocation 전체를 선형 탐색한다. | 기존 2의 거듭제곱 allocation 크기를 size-class bucket으로 사용하고, 고정 tensor는 arena slot에 배정한다. | 할당당 pool 탐색·소유 객체 생성. 실제 수명 계획이 중간 buffer 재사용 시점을 정한다. |
| 11 | [block 구성](../../crates/sejong-model/src/architecture.rs#L244)이 실행마다 이름·HashMap·Vec를 재구성한다. | 모델 로드 때 typed block·parameter index·미분·checkpoint 계획을 준비한다. 고정 주소의 GPU 구간은 HIP graph로 재실행한다. | CPU Graph 해석·metadata·개별 launch 제출. GPU routing과 비동기 복사 준비가 graph replay에 앞선다. |
| 12 | Q/K/V, gate/up이 같은 입력을 별도 GEMM으로 읽고 각각 dX를 만든다. | Compute weight를 묶고 projection·RoPE/activation epilogue와 입력 adjoint reduction을 결합한다. | 입력 재독출·launch·dX 합산. Parameter VJP의 공유 Q·입력 투영·선택 잔차를 같은 묶음에 연결한다. |

Packed dot2는 [AMD RDNA2 ISA의 FP16 두 곱·FP32 누산 명령](https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture)에 근거한다. 실제 compiler probe는 입력 변환 두 줄을 제거해 `v_dot2c_f32_f16` 생성을 확인했다. NN 형상의 LDS는 4,096→3,072 bytes, 정적 barrier는 10→7로 바뀌었다. NN의 VGPR은 111→114, NT는 123→98이었다. 정적 명령 수·register 자원과 실제 실행 시간은 함께 기록했다.

독립 GPU 실험은 F16 입력·출력과 FP32 누산을 사용했다. `(M,N,K)`가 `(512,384,128)`, `(512,128,384)`, `(64,512,512)`인 GEMM에서 기존→dot2 중앙값은 각각 **80.340→65.075µs**, **87.668→66.046µs**, **73.888→56.684µs**였다. 전체 출력의 상대 L2 차이는 9.48×10⁻⁶–1.19×10⁻⁵였다. 이 수치의 적용 범위는 해당 독립 GEMM이며, 생산 적용 때 전체 학습의 누산 순서 변화와 loss·재개를 대조한다.

GEMM의 다음 단계는 실제 `(shape,dtype,전치,layout)`에 따라 tile·warps·K 특수화를 선택하는 것이다. 현재 BM16·BN32·BK32·warps4를 공유한다. 연속성과 배수 정보를 전달하면 scalar half load와 layout 변환을 줄일 후보가 생긴다. 후보 집합은 wave 크기·register·LDS 한도에서 만들고 측정한 선택을 compiler·GPU 식별자와 함께 cache에 저장한다. [Triton 공식 GEMM 문서](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)는 L2 재사용을 위한 tile 순서와 shape별 선택을 설명하고, [AMD 성능 지침](https://gpuopen.com/learn/rdna-performance-guide/)은 VGPR 압력·LDS bank·coalescing을 다룬다.

현재 attention은 online softmax와 제한된 local window를 사용한다. 다음 구현은 여러 query의 K/V 재사용과 Delta 공유를 결합한다. [FlashAttention 연구](https://arxiv.org/abs/2205.14135)의 SRAM tile 재사용 원리를 적용하되 local causal window·GQA와 실제 FP32/FP64 실행 경로를 각각 유지한다. Delta 공유는 유효 pair의 정확한 곱·합 기준 약 0.602G 산술을 줄인다. 현재 counter의 dKV 계수 10D에는 이 내적이 포함되며, 공유 경로의 계수는 8D다. FMA=2 규칙으로 계산한 절감량은 약 0.607GFLOPs/step이다.

실행 계획은 step 동안 고정된 master·mirror·gradient buffer와 block별 activation 수명을 한 번 준비하는 구조가 적합하다. 이 구조에 grouped experts와 비동기 복사를 연결한 뒤 HIP graph를 적용한다. [HIP graph 규격](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/hipgraph.html)은 stream 연산 capture와 재실행을 설명한다. [HIP 비동기 실행 문서](https://rocmdocs.amd.com/projects/HIP/en/develop/how-to/hip_runtime_api/asynchronous.html)에 따라 pinned staging과 stream 순서로 복사·계산의 의존성을 연결한다.

구현 순서는 **메모리 예산 기반 활성값 보존 → step별 FP16 mirror → F16 dot2 GEMM → gradient epilogue·grouped experts → arena·고정 실행 계획·HIP graph**로 잡는다. 앞의 두 항목은 같은 step의 고정 상태를 재사용하고, dot2는 현재 GPU의 실제 명령을 활용한다. Delta 공유·RMSNorm reduction·QKV/gate-up 결합은 해당 실행 계획에 차례로 연결한다.
