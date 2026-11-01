# 최신 학습 가속 논문과 Sejong 적용 순서

조사일: 2026-11-01. 2025–2026년 연구를 중심으로 attention·MoE·저정밀 연산·활성값 보존·역전파·optimizer·byte 계층·학습 일정·자료 선택에 관련된 논문과 연구 발표59편을 조사했다. 아래 목록은 원문 초록으로 선별한 연구와 방법·실험·부록까지 읽은 주요 후보를 함께 기록한다. 최근 공개일은 2026-10-28까지 포함한다. 링크는 논문 원문, 학회 proceedings, 저자 연구 페이지다.

**후속 전체 모델 실험을 거쳐 활성값 보존·실행 batch 통합·RDNA2 dot2를 기본 경로에 적용했다.** [현재 구조에 맞춘 실측과 적용 결과](/home/renya/Sejong/docs/measurements/2026-11-01-architecture-fit/README.md)의 독립520-step 비교는222.499→114.034ms/step, 검증 BPB6.769966→6.769952다. 적용 후 생산 경로의66-step 실행은111.436ms/step을 기록했다. Step별 compute weight 공유는 누적 횟수가2 이상일 때 추가 이득을 얻는다. 최종 B8/A1 구성의 expert counts CPU 회수는8회/step이므로 grouped MoE의 후속 비용도 이 기준으로 평가한다.

아래 논문별 비용 분석과 우선순위 표는 최초 조사 당시의 B1/A8·checkpoint 구성에 대응한다. 실제 적용 순서는 위 후속 실험에서 확정한 세 가지를 따른다. Layer dropout·잔차 보정 입력 VJP·S3R은 같은 검증 BPB까지의 총시간을 평가할 연구 후보로 둔다.

## 판단의 출발점

[현재 통합 실측](/home/renya/Sejong/docs/measurements/2026-11-01-fused-applied/README.md)은 RX 6600, 26,488,192 parameters, sequence512 × batch1 × accumulation8, SGD, parameter VJP rank16·잔차1/4 조건이다. Warmup2 뒤 compile 없는 64 step의 평균은 **226.306ms**, 주요 issued FLOPs는 **106.499GF/step**, logical FLOPs는 **94.126GF/step**, launches는 **12,908.125/step**, peak는 **472MiB**다. 이 문서의 GPU 수치는 저장된 실험에서 가져왔으며 이번 작업의 결과물은 논문 조사와 구현 설계다.

현재 구조는 다음 비용을 이미 줄였다.

- P8 계층으로 global 길이는 512 bytes에서 64 positions가 된다. Local encoder·decoder는 폭128, window128, 합계4층이다. Global은 폭512, GQA8/4, 8층이다.
- E4/top2 MoE는 global FFN의 선택된 전문가를 실행한다. 전문가 내부 폭은384다.
- QKV와 gate/up 순전파 결합, RMSNorm partial reduction, attention Delta 공유를 적용했다.
- 가중치 VJP는 공유 token 기저와 잔차 표집을 사용하고 SGD가 인수를 직접 소비한다. 입력 VJP는 `D Wᵀ`를 계산한다.

따라서 논문 속 dense/tokenized 모델의 이득을 현재 비용에 대응시킨다. 설계의 성공 지표는 **같은 자료에서의 bytes/second와 같은 검증 BPB까지 드는 총시간**이다. 정밀도·추정량·학습 일정이 바뀌는 후보에는 도달 BPB를 사용한다.

\[
T_{\mathrm{target}}=T_{\mathrm{prepare}}+\sum_{s=1}^{S_{\mathrm{target}}}T_s,
\qquad
C_{\mathrm{target}}=C_{\mathrm{prepare}}+\sum_{s=1}^{S_{\mathrm{target}}}C_s.
\]

## 우선순위와 실제 교체 범위

| 순서 | 후보 | 줄이는 비용 | 현재 근거와 구현 범위 |
| --- | --- | --- | --- |
| 1 | 활성값 보존 계획 + step별 FP16 mirror | 재실행 FLOPs·cast·launch | Fusion 전 전체 모델의 활성값 보존 실측은 259.633→217.728ms, peak472→483MiB. Mirror 공유의 코드상 절감 후보는1,400 launches/step이다. |
| 2 | dot2 + 하드웨어·형상별 kernel 계획 | 같은 GEMM의 명령·변환·메모리 이동 | 기존 독립 F16 GEMM3개는1.235–1.327×. 새 결합 projection까지 연결하고 F16 operand를 유지한다. |
| 3 | SonicMoE식 GPU grouped experts | counts readback·gather·작은 GEMM dispatch | 현재 구조의 counts CPU 회수는128회/step. GPU queue가 실제 expert tile을 실행한다. |
| 4 | Query tile attention + backward epilogue | K/V 반복 읽기·중간 gradient 쓰기 | 기존 GQA·local window·Delta를 유지한 채 여러 query가 tile을 공유한다. MoE dH·SwiGLU·router gradient도 결합한다. |
| 5 | Layer dropout의 학습 일정 | 선택된 block의 forward와 backward | 2026 논문은271M–8.2B에서 비임베딩 training FLOPs 최대25% 감소를 연구했다. Sejong의 층별 비용으로 목표 예산을 다시 계산한다. |
| 6 | 잔차 보정 입력 VJP | 현재 정확 dX GEMM의 곱셈량 | 기존 Q·B를 공유하는 새 설계. 준비·복원·표집 비용과 깊이를 따라 증가하는 분산을 함께 계산한다. |
| 7 | 작은 상태의 adaptive scaling / S3R | 도달 step 수 / 저정밀 역전파 원가 | APOLLO의 projection 통계와 S3R의 낮은 rank 오차 보정을 현재 factor update에 연결하는 후속 연구다. |

1–4는 현재 모델의 실행 경로를 중심으로 구현한다. 5–7은 학습 규칙 또는 정밀도를 함께 바꾸므로 동일 BPB까지의 전체 비용을 비교한다. 생산 루프는 준비된 계획·buffer·GPU metadata를 소비하고, 형상별 선택과 자원 계산은 준비 경계에 둔다.

## 1. 메모리가 허용하는 계산은 한 번 수행한다

[Adacc](https://arxiv.org/abs/2508.00806)는 tensor마다 보존·재계산·압축을 선택하며 V100 기반 실험에서 full recompute 대비1.01–1.37× throughput을 보고했다. [Meta/AMD의 2026-10 연구](https://arxiv.org/abs/2610.22230)는 장비의 메모리와 명령 특성을 함께 활용하는 실행 계획을 다룬다.

Sejong의 직접 접점은 [block checkpoint](/home/renya/Sejong/crates/sejong-model/src/architecture.rs:307)와 [관측별 parameter cast](/home/renya/Sejong/crates/sejong-model/src/lib.rs:206)다. 같은 optimizer step의8개 관측은 같은 master weight를 사용한다. FP16 mirror를 step 수명으로 두면200개 tensor의 변환을1,600회에서200회로 줄일 수 있다. Mirror는52,976,384 bytes이며, 절감되는 논리적 read/write는 `7 × parameter_count × (4+2)` =1,112,504,064 bytes/step이다. 물리적 DRAM traffic은 cache와 실행 순서에 따라 달라진다.

저장할 활성값 집합 S는 다음 실제 peak 조건 안에서 고른다.

\[
\max_t\{P+\sum_{i\in S,\,live_i(t)}b_i+workspace(t)\}\le M_{budget}.
\]

`b_i`에는 allocator의 실제 할당 단위와 alias를 반영한다. 재실행을 공유하는 DAG에서는 같은 연산의 절감량을 한 번 계상한다. 압축 후보의 조건은 `encode+decode+추가 dispatch < 회피하는 재계산 시간`이다. 기본472MiB 형상에서는 활성값 보존을 먼저 평가할 근거가 충분하다. [기존 전체 모델 실험](/home/renya/Sejong/docs/measurements/2026-11-01-low-level/README.md)의16.14%는 fusion 전 조건의 결과이며, 후속 통합 비교의 기준은226.306ms다.

Mirror는 원래 master leaf로의 cast 미분 연결을 유지하고 optimizer 갱신과 함께 새 version을 만든다. 이 소유권을 정하면 arena와 graph capture에서도 주소와 수명을 고정할 수 있다.

## 2. 최신 AMD 연구에서 가져올 핵심은 명령·tile 비용 모델이다

[Architecting for Hardware Diversity](https://arxiv.org/abs/2610.22230)의 Origami는 tile별 연산, cache traffic, 명령 issue, wave tail을 모델링한다. 논문의 장비는 MI350X/B200이고 대상은 production 추천 모델이다. 해당 연구의 전체1.5× 개선과 Sejong의 예상 시간을 구분한다.

현재 [일반 GEMM](/home/renya/Sejong/python/sejong_kernels/ops/kernels.py:24)과 [결합 projection](/home/renya/Sejong/python/sejong_kernels/ops/projection.py:7)은 F16 load 뒤 F32로 변환해 dot를 실행한다. [기존 compiler·GPU probe](/home/renya/Sejong/docs/measurements/2026-11-01-low-level/README.md)는 F16 operand를 유지할 때 `v_dot2c_f32_f16` 생성과 대표 GEMM의 개선을 확인했다. [RDNA2 ISA](https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture)는 FP16 두 곱을 FP32에 누산하는 이 명령을 정의한다.

계획의 key는 `(GPU, compiler, M,N,K,dtype,전치,layout,출력 수)`다. Dot2와 주소 계산이 사용하는 VALU, VGPR/LDS 점유, 실제 CU 수, 출력 tile의 wave tail을 함께 계산한다. 다중 accumulator를 쓰는 QKV·gate/up은 단일 GEMM과 다른 register 예산으로 계획한다. F16×F16 경로를 먼저 연결하고 FP32 cotangent를 사용하는 곱에는 해당 operand 정밀도를 유지한다.

현재 RMS 최초 준비는 형상당 최대774 launches를 사용한다. 새 계획은 측정 cache와 장비 비용 모델로 후보를 줄여 준비 시간도 함께 최적화할 수 있다. Steady-state에서는 선택 결과를 재사용한다.

## 3. SonicMoE: routing부터 backward까지 GPU에서 묶는다

[SonicMoE](https://arxiv.org/abs/2512.14080)는 IO를 줄이는 grouped GEMM과 backward 결합을 다룬다. H100의7B MoE kernel에서 ScatterMoE 대비1.86× throughput·45% activation 절감을 보고하며, 전체 분산 학습 결과도 별도로 제시한다. [ScatterMoE](https://arxiv.org/abs/2403.08245)의 scattered linear와 함께 현재 구현의 데이터 이동을 재설계하는 근거로 사용한다.

[expert_routes](/home/renya/Sejong/crates/sejong-gpu/src/hierarchy_ops.rs:331)는 GPU counts를 CPU로 읽고, [Block::apply](/home/renya/Sejong/crates/sejong-model/src/architecture.rs:201)는 expert마다 gather와 FFN을 호출한다. 기본 global8층 × 관측8회 × 순전파/재계산2회 =128회 counts 회수다. Global64 positions의 top2 routing은128개 expert assignments를 만들고, 평균 expert rows는32다.

제안 실행은 `counts → offsets → tile queue → grouped gate/up → grouped down → combine`이다. Gather를 gate/up의 input load에 넣고 expert weight offset은 레이어 준비 시 보관한다. Tile queue의 실제 길이는 `Σ_e ceil(M_e/BM)`이다. 총 assignments S, 전문가 E, `a=min(E,S)`일 때 안전한 queue capacity는

\[
a+\left\lfloor\frac{S-a}{BM}\right\rfloor
\]

로 잡을 수 있다. 실제 queue만 실행하면 empty capacity에 대한 GEMM을 줄인다. 현재 기본 expert의 `M_e≤64`는 parameter VJP의 긴 입력 조건보다 작아 정확 dW 경로와 연결된다. 큰 batch에서는 기존 압축 선택과 donor 순서를 다시 계획한다.

SonicMoE의 token rounding은 routing 선택을 바꾸는 별도 알고리즘이다. 첫 구현은 현재 top2 assignments를 보존하는 grouped 연산으로 정의한다.

Backward의 다음 결합은 실수 산술에서

\[
Y=A W_d,\quad Z=dO W_d^\top,\quad dA=sZ,\quad ds=\langle Z,A\rangle
\]

를 이용한다. 기존 `ds=⟨dO,Y⟩`와 연결되며 expert 내부 폭384에서 reduction을 처리한다. 기존 G/U로 SwiGLU의 dG/dU까지 epilogue에 붙이면 dA buffer와 별도 launch를 줄일 수 있다. FP16 중간 rounding 위치는 이 연산 순서 변경의 수치 차이를 결정한다.

## 4. Attention은 query 축의 재사용을 확장한다

[FlashAttention-2](https://arxiv.org/abs/2307.08691)와 [FlashAttention-4](https://arxiv.org/abs/2603.05451)의 주요 이전 대상은 query tile과 non-matmul 비용 분석이다. FA4의1613TFLOPs/s 및 cuDNN 대비1.3×는 B200/BF16 조건이다.

현재 [hierarchy attention](/home/renya/Sejong/python/sejong_kernels/ops/hierarchy.py:8)은 program 하나가 query 한 개와 BK32 keys를 처리한다. Online softmax와 key tile을 이미 사용하고, dQ가 만든 Delta를 dKV가 공유한다. 다음 구현에서는 `BQ×BK` scores를 계산해 여러 query가 K/V를 읽고 재사용한다. dQ tile과 dKV tile의 출력 소유권을 나눠 atomic 경합을 제어한다. FP32 softmax·accumulator·Delta를 유지한다.

현재 global의 key loop는 최대2회, local은 최대4회다. FA4의 조건부 rescaling보다 query 재사용을 우선하는 이유가 여기에 있다. BQ/BK 선택에는 causal/window 경계의 낭비, GQA 공유, VGPR/LDS, 실제 tile 수를 포함한다. 변경이 줄이는 주된 비용은 메모리 이동과 명령이며, 유효 attention pair의 수학적 곱셈량은 같은 구조를 따른다.

[Kimi Linear](https://arxiv.org/html/2510.26692v1#S6.SS3)의 core 식 `6Td²+3TCd+TC²`에 `d=C=64`를 넣으면40,960T다. 현재 globalT64 causal QK/PV는 `4×2080×64` FLOPs/head로, 위 KDA 식의 약1/4.923이다. LocalT512/W128은57,408 pairs,14,696,448 FLOPs/head이고 KDA 식은20,971,520으로1.427배다. 이는 해당 chunk 크기의 core 산술 비교다. Gating·projection·backward와 장비 실행 시간은 각 구현의 추가 비용으로 계산한다. 현재 짧은 global과 windowed local에는 기존 softmax의 tile 개선을 먼저 적용한다.

## 5. 새 FLOPs 절감 후보: 학습 중 실행할 층의 예산을 정한다

[Don't Drop Dropout](https://arxiv.org/html/2609.05275v1)은2026-09 공개, ICML2026 연구로 Cerebras CS-3에서271M–8.2B 모델의2,400회 이상 실험을 다룬다. 층이 깊을수록 dropout 확률을 키우고, 학습 후반에는 확률을 줄이는 조합과 training scale `1/(1-p)`를 연구했다. 최대25% 절감은 비임베딩 training FLOPs 조건이며, 모델 크기·일정별 품질 결과가 다르다.

Sejong에서는 global block을 대상으로 시작할 설계 가치가 있다. Byte local 경로의 인과 처리를 유지하면서 선택된 global block의 forward·backward·checkpoint 재실행을 함께 생략할 수 있다. Batch1의 각 관측에서 mask를 한 번 결정하면 sequence별 결정을 구현하기도 쉽다.

실측한 층별 평균비용을 고정해 예산을 계획할 때, 층 ℓ의 비용을 `Cℓ`, dropout 확률을 `pℓ,s`, 항상 실행하는 비용을 `Cfixed`라 두면 다음 근사를 사용한다. 실제 MoE의 tile 수는 해당 관측의 routing에 따라 결정된다.

\[
E[C_s]\approx C_{fixed}+\sum_\ell(1-p_{\ell,s})C_\ell,
\quad
\Delta C_s\approx\sum_\ell p_{\ell,s}C_\ell.
\]

층 수의 비율 대신 현재 local/global·MoE·VJP 비용으로 예산을 정한다. 총 목표 절감률과 depth/time schedule에서 확률을 계산하고, 표집 mask를 checkpoint 재실행과 재개에서 재사용한다. 실행한 block의 residual scale을 원래 backward에 연결한다. Mask와 residual scale을 고정한 그래프에 체인룰을 적용하고, parameter VJP는 해당 그래프의 dW를 기존 잔차 추정량으로 계산한다. Dense 모델과의 품질 관계는 전체 학습 곡선에서 비교한다.

조합의 핵심은 기존 parameter VJP가 더하는 분산과 layer dropout의 분산을 함께 다루는 것이다. 초기 비교는 dropout만 바꾼 경로와 현재 VJP 경로를 구분하고, 성공한 예산을 결합한다. 매 관측의 큰 norm scan 대신 step/관측 index로 결정한 mask와 scale을 소비한다.

## 6. 새 미분 연구: 현재 잔차 보정을 dX에도 적용한다

[INSTANT](https://proceedings.iclr.cc/paper_files/paper/2026/file/d368aba36f74776cc7a1079332a31973-Paper-Conference.pdf)는 activation과 gradient의 부분공간으로 역전파 곱을 줄인다. TinyLlama BoolQ 마지막 block 미세조정의 약13× FLOPs 절감과 정확도67.71→64.89를 보고했다. 부록I.2의 V100 EfficientFormer full fine-tuning에서는 epoch110.8→116.6–120.2초였다. 이 결과는 작은 곱으로 분해했을 때의 준비·실행 비용을 함께 계산할 이유를 제공한다.

다음은 Sejong의 기존 parameter VJP와 결합하는 **새 설계 수식**이다. `D∈R^(N×M)`, `W∈R^(K×M)`, 기존 token 기저 `Q∈R^(N×r)`에 대해

\[
B=Q^\top D,\quad E=D-QB,
\qquad
\widehat{dX}=Q(BW^\top)+\frac{1}{p}E_{:,C}W_{:,C}^\top.
\]

출력 feature block C의 포함 확률을 p로 두면, 현재 D/Q/W를 조건으로 표집 평균은 `DWᵀ`다. 기존 가중치 VJP가 이미 계산한 B를 공유할 수 있다. 선택한 E 열만 계산한다. 준비된 Q가 있는 조건의 logical 비용은

\[
2NMr+2rMK+2NrK+2NrpM+2NpMK
\]

이며 B를 공유하면 첫 항을 줄인다. 현재 local gate/up의 실제 형상 `N512,K128,M384,r16,p1/4`에서는 원래 dX가50.332MFLOPs, 위 식은24.117MFLOPs, B 공유 시17.826MFLOPs다. QR·패딩·launch·optimizer 비용은 별도 항이다. 이 수치는 산술 계산이며 GPU 시간의 측정 대상은 완성한 fused 경로다.

또 다른 후보는 출력 feature 기저 `U∈R^(M×r)`, `Z=WU`, `R=W-ZUᵀ`를 사용하는

\[
\widehat{dX}=(DU)Z^\top+D_{:,C}R_{:,C}^\top/p
\]

이다. Z를8개 관측에서 공유할 수 있지만 weight 갱신마다 재생성한다. 관측당 `2NMr+2NrK+2NpMK`, weight별 준비 `2KMr+2KrpM`에 기저 생성과 저장 비용을 더한다. 두 후보의 선택은 실제 shape와 공유 횟수로 결정한다.

현재 parameter 잔차 mask는 층 사이에서 공유한다. dX까지 표집하면 upstream D가 이전 표집에 의존하므로, 새 dX의 mask는 층·관측·용도별 독립 key를 갖도록 설계한다. 새 dX key는 기존 parameter-VJP 표집 key와도 분리한다. Checkpoint 재실행은 저장한 key를 소비한다. 각 단계에서의 조건부 평균과 깊이별 분산 증폭을 분석할 수 있는 구조다. [Graph의 Matmul backward](/home/renya/Sejong/crates/sejong-gpu/src/graph.rs)는 현재 dX를 먼저 계산하므로 B 준비와 실행 순서를 함께 바꿔야 한다.

[PRAC](https://arxiv.org/html/2602.23111v1)와 [GRASS](https://aclanthology.org/2024.emnlp-main.835.pdf)는 principal 성분과 표집 잔차, 필요한 gradient 행만 계산하는 설계의 비교 근거다. Residual block `Hc`에 독립 포함 확률 `pc`를 쓰면 parameter-gradient MSE는 `Σ(1/pc−1)‖Hc‖²`다. 비용 예산 아래 최적 확률은 `pc=min(1,‖Hc‖/sqrt(λ costc))` 형태다. 실제 구현은 기존 fixed-size 비복원 표집의 포함 확률을 정확히 사용하고, 이미 생성한 tail 통계나 이전 관측 통계의 재사용 비용을 계산한다.

## 7. 새 저정밀 연구: S3R의 구조적 오차를 인수로 처리한다

[S3R](https://arxiv.org/abs/2610.36587)는2026-10-27 공개됐다. Gradient 원소마다 난수를 만드는 비용을 token당 난수 하나로 줄이고, 공유 난수가 만든 오차의 평균 성분을 보정한다. TPU v5p의 dense LLM에서 전체 INT8 step1.24–1.32× BF16을 보고했다.

연구 접점은 원래 gradient와 복원한 quantized gradient의 차이 `E=Dq−D`다. Token별 평균 `e=E1/M`의 dX 기여는

\[
e(W1)^\top
\]

라는 rank1 항이다. 해당 항을 빼는 보정을 dX epilogue에 합칠 수 있다. 두 축의 평균을 보정하는 구조도 낮은 rank로 표현된다. 현재 factor gradient update와 공유할 수 있는지는 실제 인수의 축과 optimizer 소비 순서에 따라 결정한다.

RX 6600의 후보 명령은 INT8 dot4다. 준비된 weight quantization을 step에 공유하고 다음 조건을 만족하는 형상을 선택한다.

\[
T_{quantX}+T_{quantD}+T_{quantW}/A+T_{dot4}+T_{scale}+T_{correction}<T_{dot2}.
\]

GlobalM64·expert평균M32에서는 작은 행렬의 고정 비용이 크게 작용한다. 먼저 dot2를 전체 관련 경로에 연결한 뒤 그 시간을 비교 기준으로 삼는다. S3R의 난수와 VJP 표집 key는 역할별로 구성하고, FP32 master 갱신을 유지한다.

## 8. 학습 일정·optimizer·byte 경계의 후보

[Small Batch Size Training](https://arxiv.org/html/2507.07101v2)과 [2026-10 optimizer/batch 연구](https://arxiv.org/abs/2610.08975)는 작은 batch의 SGD와 실제 batch별 optimizer 비교를 뒷받침한다. 현재 batch1/accumulation8의 변경에는 두 실험 축이 있다.

- `(batch,accumulation)=(1,8),(2,4),(4,2),(8,1)`은 step당4,096 bytes를 유지하고 실행 묶음 크기를 바꾼다. 같은 데이터 순서와 실제 메모리 peak를 비교한다.
- `batch1,accumulation1`은 optimizer 갱신 빈도를8배 높인다. 학습률 일정은 처리 bytes 기준으로 맞추고 optimizer·cast 횟수까지 포함한 bytes/second와 BPB를 비교한다.

현재 [압축 선택기](/home/renya/Sejong/crates/sejong-gpu/src/parameter_vjp.rs:100)는 `N≥입력 폭`을 조건으로 삼는다. Batch8에서는 globalN이64→512로 바뀌어 압축 후보와 Q 공유가 달라진다. 실행 묶음 변경의 비교에서는 이 추정량 변화도 기록한다. 독립 sequence의 인과 경계는 각 sequence 안에서 유지한다.

[APOLLO](https://proceedings.mlsys.org/paper_files/paper/2025/file/437bc4ccafd3fc6d4289bd10940be42b-Paper-Conference.pdf)의 작은 random projection으로 adaptive scaling을 만드는 방식은 current factor와 연결 가능하다. `G=AB+scatter(T)`이면 `PG=(PA)B+P[:,C]T`로 계산하고 최종 scale은 SGD update epilogue에서 소비한다. Random P와 현재 data-dependent Q의 통계적 역할은 각각 정의한다. [2026-10의 Muon 소형 모델 실험](https://arxiv.org/pdf/2610.21555)은17.7M/33.6M·A100에서 optimizer 원가 때문에 equal-time 순위가 달라지는 결과를 제시했다. 현재 SGD에는 작은 상태·작은 추가 곱부터 비교하는 순서가 적합하다.

Byte 계층에서는 [BLT](https://arxiv.org/abs/2412.09871), [H-Net](https://arxiv.org/abs/2507.07955), [SOMBRERO](https://arxiv.org/abs/2601.22805), [2026-10 novelty patching](https://arxiv.org/abs/2610.22100)을 비교했다. 현재 P8의 global 비용을 `aL/P+b(L/P)²`로 놓으면 P16에서 projection/FFN은 절반, attention의 주요 quadratic 항은1/4이 된다. Local 경로와 byte head 비용을 합산해 전체 감소량을 계산한다. Novelty patching은12M/50M의 규모가 현재와 가깝고 two-byte LUT로 경계를 정하므로, 더 큰 평균 P에서 품질을 유지하는 경계 정책의 후보다. H-Net 원문은 현재 학습 구현에서 isotropic 대비 최대 약2배의 시간 증가도 기록했다. Dynamic packing·경계 학습·smoothing 원가를 함께 평가한다.

자료 선택도 총 학습 FLOPs를 바꾼다. [PreSelect](https://arxiv.org/html/2503.00808v1)의 headline10×는 RefinedWeb의 선택 전 baseline과 비교한 downstream 결과이며, scorer 제작에 대형 모델의 선행 계산을 사용했다. [BETR](https://arxiv.org/html/2507.12466v1)은 목표 과제와 자료의 일치를 연구한다. Sejong에서는 학습 루프 밖에서 자료별 점수를 준비하고 held-out 자료로 도달 시간을 비교하는 설계가 적합하다. [OPUS](https://arxiv.org/html/2602.05400v1)의 scoring overhead4.7%는 scoring512 tokens 대 update6,144/24,576 tokens 조건이다. 현재512-byte 관측에서는 후보 scoring 자체의 비용부터 계산한다.

## 현재 우선순위를 낮춘 항목

| 연구 | 현재 형상과 비용의 관계 |
| --- | --- |
| Native Sparse Attention | 논문의64K context와 현재global64/localwindow128 사이에 큰 길이 차이가 있다. 장문 global 확장에서 선택 예산을 다시 계산한다. |
| Cut Cross-Entropy | 현재 vocab257, FP32 logits512×257은526,336 bytes다. 큰 vocabulary의 수십GB logits 절감 조건과 현재 byte head의 비용을 구분한다. |
| DyT | RMSNorm partial reduction을 적용한 현재 기준으로 tanh·α gradient의 추가 비용과 품질을 비교한다. |
| Quartet/NVFP4 | 논문 속 FP4 가속은 native FP4 matrix 장비의 결과다. gfx1032는 dot2/dot4 명령과 packing·scale 비용을 기준으로 선택한다. |
| GaLore2/Lotus/SOAP | 현재 SGD는 SVD와 Adam moment를 소비하지 않는 구조다. 새 통계·기저 갱신의 비용과 수렴 이득을 함께 계산한다. |
| Dust/NDFA | Dust는 다수 draw/population, NDFA는 학습하는 surrogate feedback의 비용이 추가된다. 단일 GPU byte 사전학습의 compute-to-BPB를 새로 평가하는 연구다. |
| PlaneZip/SlackSwap | 보고 성능의 근거는 trace 기반 simulation이다. 현재472MiB 형상에서는 GPU 활성값 보존이 먼저다. |
| 추론 전용·분산 통신 중심 기법 | 훈련의 forward/backward·단일 GPU 실행에 실제로 남는 항을 골라 계산한다. |

## 조사 논문 목록

날짜는 최초 arXiv 공개일 또는 명시한 학회/열람 버전이다. `본문`은 방법·실험 또는 부록의 관련 구간까지 열람한 항목, `선별`은 공식 초록·저자 자료에서 적용 범위를 확인한 항목이다. 아래의 판단은 Sejong 코드 분석과 결합한 연구 결과다.

### 실행·정밀도·메모리

| 번호 | 논문·시점 | 확인 범위·핵심 조건 | 적용 판단 |
| --- | --- | --- | --- |
| S01 | [Architecting for Hardware Diversity](https://arxiv.org/abs/2610.22230),2026-10-15 | 본문; MI350X/B200 추천 모델·Origami | RDNA2 명령·tile 계획 |
| S02 | [Adacc](https://arxiv.org/abs/2508.00806),2025-08-01 | 본문; V100·보존/재계산/압축 | 활성값 계획 |
| S03 | [PyGraph](https://arxiv.org/abs/2503.19779),2025-03-25 | 본문; A6000·선별 capture·argument indirection | Arena/StepState/HIP graph |
| S04 | [Meganeura](https://arxiv.org/abs/2608.01563),2026-08-03 | 본문; 여러 vendor·typed graph·수명 alias | Rust 정적 실행 계획 |
| S05 | [S3R](https://arxiv.org/abs/2610.36587),2026-10-27 | 본문; TPUv5p·INT8·공유 난수/오차 보정 | Dot4 후속 연구 |
| S06 | [Quartet II](https://arxiv.org/abs/2601.22813),2026-01-30/v2 06-01 | 본문; Blackwell·MS-EDEN | 양자화 원가·수치 설계 |
| S07 | [Quartet](https://arxiv.org/abs/2505.14669),2025-05-20 | 선별; Blackwell FP4 | Native 명령 지원 기준 |
| S08 | [Pretraining LLMs with NVFP4](https://arxiv.org/abs/2509.25149),2025-09-29 | 선별; 12B/10T tokens | 긴 학습의 precision 경계 |
| S09 | [FP4 All the Way](https://arxiv.org/abs/2505.19115),2025-05-25/v2 08-10 | 선별; Gaudi2·7B·열람 버전200B tokens | SR·forward/backward 정밀도 |
| S10 | [Stable FP4 Training via Transposition-Invariant Block Quantization](https://arxiv.org/abs/2607.24953),2026-07-27 | 본문; software emulation·2D scale | 전치 간 scale 일관성 |
| S11 | [COAT](https://arxiv.org/abs/2410.19313),2024-10-25/ICLR2025 | 본문; H100 FP8·activation/Adam state | 확대 모델·NativeValue 후보 |
| S12 | [HALO](https://arxiv.org/abs/2501.02625),2025-01-05/v3 11-05 | 본문; low-precision fine-tuning·Hadamard | INT8 rotation 비용 |
| S13 | [Why Low-Precision Transformer Training Fails](https://arxiv.org/abs/2510.04212),2025-10-05/v4 2026-06-14 | 본문; BF16 attention rounding·Delta | Attention 누산·출력 정밀도 |
| S14 | [PlaneZip](https://arxiv.org/abs/2610.25033),2026-10-18 | 본문; trace-driven hardware simulation | 메모리 압박 시 codec 후보 |
| S15 | [Liger Kernel](https://arxiv.org/abs/2410.10989),2024-10-14 | 선별; Triton fusion·chunking | 남은 gradient epilogue |
| S16 | [SlackSwap](https://arxiv.org/abs/2610.13003),2026-10-09 | 선별; A100 trace simulation·link budget | 큰 activation offload 계획 |

### Attention·전문가·byte 구조

| 번호 | 논문·시점 | 확인 범위·핵심 조건 | 적용 판단 |
| --- | --- | --- | --- |
| A01 | [SonicMoE](https://arxiv.org/abs/2512.14080),2025-12-16/v2 2026-03-26 | 본문; Hopper/Blackwell·IO/tiling | GPU grouped MoE 우선 |
| A02 | [FlashAttention-4](https://arxiv.org/abs/2603.05451),2026-03-05 | 본문; B200·asymmetric hardware | Query tile와 softmax 비용 |
| A03 | [FlashAttention-3](https://arxiv.org/abs/2407.08608),2024-07-11 | 선별; H100·WGMMA/TMA | 장비별 pipeline 비교 |
| A04 | [FlashAttention-2](https://arxiv.org/abs/2307.08691),2023-07-17 | 선별; work partitioning | Query tile 기반 연구 |
| A05 | [ScatterMoE](https://arxiv.org/abs/2403.08245),2024-03-13 | 본문; scattered grouped linear | Gather와 GEMM 결합 |
| A06 | [MegaBlocks](https://arxiv.org/abs/2211.15841),2022-11-29 | 선별; block-sparse MoE | Dropless tile packing 기반 |
| A07 | [Native Sparse Attention](https://arxiv.org/abs/2502.11089),2025-02-16 | 본문; 27B·64K A100 kernel | 장문 global 후보 |
| A08 | [Gated Delta Networks](https://arxiv.org/abs/2412.06464),2024-12-09/ICLR2025 | 선별; chunk-parallel delta rule | 길이별 core 원가 비교 |
| A09 | [Kimi Linear](https://arxiv.org/abs/2510.26692),2025-10-30 | 본문; hybrid·1M decoding 수치 | 현재 T/W에 비용식 대입 |
| A10 | [Transformers without Normalization](https://arxiv.org/abs/2503.10622),2025-03-13 | 본문; DyT·H1007B | 최적화 RMS와 비교 |
| A11 | [Byte Latent Transformer](https://arxiv.org/abs/2412.09871),2024-12-13/ACL2025 | 선별; entropy patch·byte 계층 | P8 이후 추가 이득 평가 |
| A12 | [H-Net](https://arxiv.org/abs/2507.07955),2025-07-10/ICLR2026 | 본문; learned chunking·실제 시간 | 경계 정책과 packing 원가 |
| A13 | [SOMBRERO](https://arxiv.org/abs/2601.22805),2026-01-30 | 선별; learned boundary·byte smoothing | 큰 평균 P에서 품질 |
| A14 | [Scratchpad Patching](https://arxiv.org/abs/2605.09630),2026-05-10 | 선별; patch lag·주요 수치는 추론 | Global prefix 갱신 비용 |
| A15 | [Breaking the Token Ceiling](https://arxiv.org/abs/2609.12303),2026-09-11 | 본문; 1.28B·teacher distillation | Teacher 계산 포함 품질 연구 |
| A16 | [Patch where the text is new](https://arxiv.org/abs/2610.22100),2026-10-15 | 본문; 12M/50M·two-byte LUT | 현재 규모의 경계 후보 |
| A17 | [Toppling the Hierarchy](https://arxiv.org/abs/2609.00463),2026-08-31 | 선별; byte attention·문자 조작 | P 확대의 local 경로 |
| A18 | [AU-Net](https://arxiv.org/abs/2506.14761),2025-06-17 | 선별; byte/word 계층·compute 통제 | 계층 설계 비교 |
| A19 | [FlashMoE](https://openreview.net/pdf?id=EZfDHprhZM),NeurIPS2025 | 본문; 8H100·persistent distributed MoE | Device scheduling 원리 |

### 역전파·optimizer

| 번호 | 논문·시점 | 확인 범위·핵심 조건 | 적용 판단 |
| --- | --- | --- | --- |
| G01 | [INSTANT](https://proceedings.iclr.cc/paper_files/paper/2026/file/d368aba36f74776cc7a1079332a31973-Paper-Conference.pdf),ICLR2026 | 본문/부록; low-rank activation/gradient | 잔차 보정 dX 새 설계 |
| G02 | [PRAC](https://arxiv.org/html/2602.23111v1),2026-02-26 | 본문; principal/random feature 공간 | 잔차의 분산 예산 |
| G03 | [APOLLO](https://proceedings.mlsys.org/paper_files/paper/2025/file/437bc4ccafd3fc6d4289bd10940be42b-Paper-Conference.pdf),MLSys2025 | 본문; projected optimizer 통계 | Factor 기반 경량 scaling |
| G04 | [GaLore 2](https://arxiv.org/html/2504.20437v1),2025-04-29 | 본문; randomized SVD·7B/500B | 기저 품질/갱신 비용 |
| G05 | [Lotus](https://arxiv.org/html/2602.01233v1),2026-02-01 | 본문; randomized projection·refresh | 관측 사이 기저 재사용 |
| G06 | [GRASS](https://aclanthology.org/2024.emnlp-main.835.pdf),EMNLP2024 | 본문; 60M·1B·7B 크기별 속도 | 작은 GEMM과 표집 비용 |
| G07 | [SEBP](https://aclanthology.org/2026.eacl-srw.31/),EACL2026 | 선별; padding gradient 희소성 | Masked supervision 후보 |
| G08 | [OPUS](https://arxiv.org/html/2602.05400v1),2026-02-05 | 본문; Ghost/CountSketch·online selection | Scoring 전체 원가 |
| G09 | [Dust](https://qlabs.sh/research/dust),2026-10 | 저자 원문; 64–16k draws/update | Population 포함 FLOPs 연구 |
| G10 | [LionMuon](https://arxiv.org/html/2609.35297v1),2026-09-28 | 본문; 간헐 Muon·124M/355M | SGD 대비 update 원가 |
| G11 | [When Does Muon Actually Win?](https://arxiv.org/pdf/2610.21555),2026-10-15 | 본문; A100·17.7M/33.6M·equal-time | 현재 규모의 optimizer 비교 |
| G12 | [The Best Optimizer Depends on Batch Size](https://arxiv.org/abs/2610.08975),2026-10-06/v2 10-12 | 본문;128K–2M batch·순위 변화 | 실제 byte batch에서 판단 |
| G13 | [SOAP](https://arxiv.org/abs/2409.11321),2024-09-17/ICLR2025 | 선별; eigenbasis Adam·large batch | 수렴과 통계 갱신 비용 |
| G14 | [Drop-Muon](https://arxiv.org/html/2510.02239v1),2025-10-02 | 본문; v1 CNN·suffix update | 앞쪽 dX까지 줄이는 범위 |
| G15 | [Nonlinear Direct Feedback Alignment](https://arxiv.org/abs/2610.36718),2026-10-27 | 선별; ResNet/ViT·학습 feedback | Byte 사전학습 확장 연구 |
| G16 | [Deeper Backpropagation or More Updates?](https://arxiv.org/abs/2610.38422),2026-10-28 | 선별; looped LM·backward window | 예산별 도달 품질 비교 |

### 일정·자료·목적함수

| 번호 | 논문·시점 | 확인 범위·핵심 조건 | 적용 판단 |
| --- | --- | --- | --- |
| T01 | [Don't Drop Dropout](https://arxiv.org/abs/2609.05275),2026-09-04/ICML2026 | 본문; CS-3·271M–8.2B·depth/time schedule | Global block 실행 예산 |
| T02 | [Small Batch Size Training](https://arxiv.org/abs/2507.07101),2025-07-09/v2 08-27 | 본문; 작은 batch·SGD·Adam half-life | 누적 횟수와 실행 묶음 |
| T03 | [Critical Batch Size Revisited](https://arxiv.org/abs/2505.23971),2025-05-29/NeurIPS2025 | 초록; OLMo1B/7B·batch warmup | Step 수와 bytes를 함께 비교 |
| T04 | [PreSelect](https://arxiv.org/abs/2503.00808),2025-03-02 | 본문/부록;1B/3B·fastText·선행 scorer 계산 | Offline 자료 선택 |
| T05 | [Language Models Improve When Pretraining Data Matches Target Tasks](https://arxiv.org/abs/2507.12466),2025-07-16 | 본문; BETR·held-out task 비교 | 자료 분포와 목표 품질 |
| T06 | [Cut Your Losses](https://arxiv.org/abs/2411.09009),2024-11-13/ICLR2025 | 원문/초록; large vocabulary | V257의 실제 비용 기준 |
| T07 | [Curriculum-Guided Layer Scaling](https://arxiv.org/abs/2506.11389),2025-06-13/v5 2026-08-01 | 본문;100M/1.2B·깊이/자료 성장 | 전체 schedule 변경 후보 |
| T08 | [Progressive Residual Warmup](https://arxiv.org/abs/2603.05369),2026-03-05 | 본문; 층별 residual contribution | 같은 step 비용의 수렴 후보 |

## 구현과 비교 순서

첫 묶음은 활성값 계획·mirror·dot2다. 다음으로 GPU grouped MoE와 query tile attention을 연결한다. 이때 안정된 tensor 수명을 arena에 배정하고 static 구간부터 HIP graph를 재생한다. Graph 손익은 `반복 수 × (직접 실행−replay−state 갱신) > 준비 시간`으로 계산한다. [기존 add-chain 실험](/home/renya/Sejong/docs/measurements/2026-11-01-fused-execution/README.md)의 약12회 손익분기는 해당 chain의 결과다.

새 학습 규칙은 layer dropout과 입력 VJP를 개별 비교한 뒤 결합한다. 성능 측정은 현재 benchmark의 동일 자료·seed·처리 bytes·compile 분리 조건을 활용한다. 미분·정밀도 변경은 해당 연산의 수치 대조와 전체 모델의 BPB 곡선으로 확인한다. 준비 비용, optimizer, scoring, 양자화와 residual correction을 전체 시간에 포함한다. 실행 중에는 선택한 kernel과 계획을 재사용한다.
