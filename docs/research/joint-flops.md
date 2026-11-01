# Sejong의 토큰 압축과 희소 연산과 기울기 교정을 결합하는 연구

2026-11-01. [앞선 FLOPs 연구](flops-learning.md)의 기울기 교정에 **byte 구간 압축, encoder 공유, 저랭크 갱신 지연, GQA와 전문가 선택**을 결합했다. 중심 설계는 작은 encoder가 모든 자료를 읽고, 큰 본체는 선택한 자료의 압축 표현을 처리하는 구조다. 본체의 정확한 미분으로 예측 오차를 교정하고, 예측 갱신은 작은 행렬로 누적한다.

이 문서는 형상 기반 연구 가정과 계산 결과를 보존한다. 실제 Rust·Triton 구현은 [계층 본체와 교정 학습](../compute.md), 실행 결과는 [RX 6600 측정](../measurements/2026-11-01/README.md)에 기록했다. 현재 predictor는 `[mean(E), mean(target_embedding), mean(E * target_embedding), 1]`의 `3 × encoder_width + 1`차원을 직접 rank² 계수에 사상한다. 아래 비용 모델은 64차원 feature projection을 사용한 설계다. 실행 결과의 FLOPs는 실제 predictor와 커널 형상을 집계한다.

동일한 4,096 raw bytes를 처리하는 step의 형상 계산에서, P=8 압축과 공유 encoder와 기울기 교정의 주요 연산은 **0.7952 GFLOPs/byte**다. GQA와 16개 중 2개 전문가를 실행하는 구조를 더하면 **0.3458 GFLOPs/byte**다. 비교하는 dense byte 모델은 **24.3753 GFLOPs/byte**다. 아래 수치는 학습할 파라미터 형상과 실행할 연산의 산술량이다. 목표 언어 품질까지의 절감은 구조별 학습량과 기울기 잔차를 측정해 결정한다.

| 산출물 | 계산한 내용 |
| --- | --- |
| [결합 비용 계산기](../../scripts/research-joint-flops.py) | 약 4B 형상의 구성별 FLOPs, GEMM 타일, 전문가 부하 분포, 기저·예측기·SGD 비용 |
| [계산 결과](joint-flops-results.json) | 후보 18개, source SHA256, 실행 환경, 항목별 비용 |
| [행렬 항등식 검산](../../scripts/research_joint_identities.py) | 층화 표집, 내부 토큰 표집 분산, 9회 지연 SGD, 과거 통계로 고정한 예측 계수 |

실행은 `OPENBLAS_NUM_THREADS=1 python3 scripts/research-joint-flops.py --out docs/research/joint-flops-results.json`이다. Python과 NumPy로 형상 계산과 FP64 항등식을 실행했다. 최대 절대 오차는 **2.67×10⁻¹⁵**였다. Rust 학습 루프와 Triton 커널의 결합 구현은 아래 실행 순서와 평가 기준을 대상으로 한다.

비용의 공통 조건은 raw 512 bytes × M=8, 정확한 본체 미분을 계산할 제어 표본 m=2, rank 32, 기저 갱신 주기 32 step, SGD다. rank는 행렬의 작은 차원으로 제한한다. FMA=2로 세고 GEMM·attention 내적·투영·복원·기저 갱신·기울기 통계·누적·optimizer를 합산했다. QR은 `4(a+b)r²`의 계산 예산을 사용한다. exp·sigmoid·norm·top-k 비교·gather/scatter·형 변환·메모리 전송은 별도 연산 종류로 계측한다. 주 표의 global checkpoint는 꺼진 조건이다. 타일 수치는 현재 GEMM의 BM16·BN32·BK32를 적용하며, local window와 GQA는 이 문서에서 정한 커널 실행 형상을 사용한다.

| 결합 구성 | 논리적 GFLOPs/byte | 타일 포함 GFLOPs/byte | 타일 포함 dense byte 기준 / 후보 |
| --- | ---: | ---: | ---: |
| dense byte 전체 미분 | 24.3494 | 24.3753 | 1.00 |
| P8 dense 전체 미분 | 3.0703 | 3.0737 | 7.93 |
| P8 + 독립 proxy + 매 step 저랭크 복원 | 0.9406 | 0.9426 | 25.86 |
| P8 + 공유 encoder + 매 step 저랭크 복원 | 0.8500 | 0.8513 | 28.63 |
| **P8 + 공유 encoder + 저랭크 지연 갱신** | **0.7940** | **0.7952** | **30.65** |
| 위 구성에 global checkpoint 적용 | 1.0476 | 1.0490 | 23.24 |
| **P8 + 위 결합 + GQA + E16 top-2** | **0.2082** | **0.3458** | **70.49** |
| P16 + 위 결합 + GQA + E16 top-2 | 0.1175 | 0.2921 | 83.45 |

각 감소 배수는 **동일한 raw byte 관측량의 주요 연산**을 비교한다. 표에는 본체의 전체 파라미터에 대한 SGD와 gradient 누적도 들어간다. 이전 연구의 24.3319에는 이 두 항을 더하기 전의 주요 forward/backward가 들어 있다. 평균 4 bytes/token을 가정한 약 4B BPE 기준의 주요 forward/backward는 5.8932 GFLOPs/byte다. 실제 학습 자료의 bytes/token과 bits-per-byte를 함께 기록해 비교 기준을 고정한다.

첫 번째 결합은 **encoder를 본체와 기울기 예측기가 공유하는 것**이다. P=8이면 작은 encoder E가 512 bytes를 처리하고 큰 본체 G가 64개 구간 표현을 처리한다. decoder D는 구간별 G 출력을 조건으로 각 byte를 예측한다. E는 d256·2층, G는 d2560·32층, D는 d512·2층이고 E와 D의 causal window는 128 bytes다. 본체와 encoder·decoder·bridge의 총 파라미터는 3,995,575,808개다.

목표 구간 j의 G 입력은 이전에 완료된 구간들의 encoder 표현이다. 첫 구간에는 학습 가능한 BOS를 넣는다. decoder의 byte 입력과 encoder skip은 목표 byte 이전의 prefix에서 얻는다. 구간 안의 여러 byte loss가 같은 G 출력을 사용하므로, 출력 미분을 구간 안에서 합친 뒤 G를 역전파한다. 이 계층 구조의 정확한 연쇄법칙으로 본체의 실행 위치 수를 줄인다. byte 구간과 global/local 모델의 결합은 [MEGABYTE](https://arxiv.org/abs/2305.07185), 내용별 patch 길이는 [BLT](https://arxiv.org/abs/2412.09871), 학습하는 causal chunking은 [H-Net](https://arxiv.org/abs/2507.07955)을 참조한다. 고정 P=8은 비용과 인과 관계를 직접 대조하는 첫 구성이다.

기울기 예측용 feature는 `mean(h_E)`와 `mean(h_E ⊙ embedding(target))`를 결합해 64차원으로 투영한다. 이 optimizer 경로에서 h_E와 target embedding은 값으로 읽는다. coefficient loss는 별도 feature projection과 예측 head의 파라미터 φ를 학습한다. 언어 모델의 encoder는 제어 표본의 언어 loss로 학습한다. target 정보는 optimizer feature 경로에 두고, 모델의 출력·routing은 causal prefix를 사용한다.

이 분리의 비용은 다음과 같다. F는 순전파, B는 역전파, C는 둘의 합이다.

\[
F_{\rm step}
=M F_E+m(B_E+C_G+C_D+C_{\rm bridge})
+C_{\rm predictor}+C_{\rm correction}+C_{\rm update}.
\]

E의 활성값을 M개에 대해 보존한 뒤 제어 표본을 고르면 선택된 E의 forward를 재사용할 수 있다. 활성값을 먼저 해제하는 스케줄은 mF_E를 추가한다. 계산 결과의 `encoder_replay_if_cache_released`가 이 추가 비용이다. 별도 d512·4층 proxy의 forward/backward를 모두 실행하는 구성에서 공유 encoder로 바꾸면 P8의 타일 포함 비용이 0.9426에서 0.8513으로 내려간다.

step 시작의 θ, φ, 기저 U·V를 고정하고, 각 관측의 예측을 cᵢ, 실제 gradient를 gᵢ라 두면:

\[
\widehat G=\frac1M\sum_{i=1}^M c_i
+\frac1m\sum_{i\in C}(g_i-c_i),
\qquad \mathbb E_C[\widehat G]=\frac1M\sum_i g_i.
\]

예측 c가 현재 θ의 encoder feature에 의존해도, θ와 c를 제어 표본 선택에 대해 고정하면 이 평균 항등식이 성립한다. coefficient target, mixing 계수와 기저 갱신 결과는 다음 step부터 적용한다. 기본 구성의 encoder·embedding·norm·BOS에는 c=0을 쓴다. 이 그룹은 m개 실제 gradient의 평균으로 학습하므로, 전체 M개 평균과 비교한 표집 분산 비율이 M/m=4다. 그룹별 잔차 통계로 이 비용을 드러낸다.

두 번째 개선은 **예측 gradient를 큰 행렬로 복원하는 시점을 늦추는 것**이다. a×b 가중치의 예측을 `cᵢ=UBᵢVᵀ`로 두고, U와 V는 정규직교 열 r개를 갖는다. 현재 유효 가중치를 다음처럼 저장한다.

\[
W_{\rm eff}=W_0+UAV^\top,
\quad W_0\leftarrow W_0-\eta\,\overline g_C,
\quad A\leftarrow A-\eta(\overline B-\overline B_C).
\]

같은 gradient 추정량으로 매 step 큰 행렬 W를 갱신하는 SGD와 대수적으로 동일하다. U와 V는 32 step 동안 고정하고, 기저를 교체하기 직전에 `W₀←W₀+UAVᵀ, A←0`을 실행한다. 학습 checkpoint는 W₀·U·V·A와 예측기·표집 상태를 함께 저장한다. 단일 가중치 형식으로 내보낼 때도 이 합산을 수행한다.

모델의 실제 계산에는 누적된 A를 즉시 반영한다.

\[
Y=XW_0+(XU)AV^\top,
\qquad dX=\delta W_0^\top+(\delta V)A^\top U^\top.
\]

여기서 `XU`와 `δV`는 계수 target `K=UᵀgV=(XU)ᵀ(δV)`에도 쓴다. 이 재사용을 포함하면, 매 step 큰 행렬 복원 비용 `2abr+2min(a,b)r²`를 32로 나누고, 제어 표본마다 `2Tr(a+b)+4Tr²`의 forward·dX 비용을 더한다. P8 dense의 타일 계산에서는 복원이 **0.2568 TF/step**에서 flush **0.0080**과 추가 forward·dX **0.0191 TF/step**으로 바뀐다. 합계 **0.2296 TF/step**을 줄인다. 토큰 압축이 추가 저랭크 경로의 T까지 줄이므로 두 개선이 잘 결합된다.

이 항등식의 optimizer 조건은 SGD다. 선형 momentum과 weight decay도 상태를 같은 두 성분으로 분해해 확장할 수 있다. Adam의 원소별 제곱·나눗셈과 전체 gradient clipping은 합쳐진 gradient에 의존하므로 별도의 full-gradient 처리 비용과 상태 설계를 추가해야 한다. 비교 실험은 같은 optimizer에서 시작한다.

세 번째 결합은 **큰 FFN의 전체 용량을 여러 전문가에 분산하고 일부만 계산하는 것**이다. 약 4B byte 기준의 파라미터 수를 정확하게 맞춘 다음 형상을 사용한다.

| 본체 구성 요소 | dense | GQA와 E16 top-2 |
| --- | ---: | ---: |
| Q/K/V/O | 838,860,800 | 503,316,480 |
| 전체 FFN | 3,145,728,000 | 3,479,961,600 |
| Router | 0 | 1,310,720 |
| Embedding·norm | 1,482,240 | 1,482,240 |
| 고유 모델 파라미터 | **3,986,071,040** | **3,986,071,040** |
| 위치당 활성 파라미터 | 3,986,071,040 | **941,104,640** |

Query 40 heads와 head 폭 64를 유지하고 KV를 8 heads로 공유한다. 각 층의 16개 SwiGLU 전문가 중 2개를 실행한다. 전문가의 FFN 폭은 21개 층에서 896, 11개 층에서 864다. GQA에서 줄인 projection 용량을 전문가와 router에 재배분한 결과다. GQA의 attention QK·AV는 40개 query heads를 계산하고, K/V projection이 줄어든다. 현재 attention 커널의 동일 Q/K/V stride를 GQA indexing으로 확장하고 공유 dK·dV의 합산을 구현해야 한다. 계산기는 이 합산의 덧셈도 포함한다. [GQA 원논문](https://aclanthology.org/2023.emnlp-main.298/)

계층 모델에 이 본체를 넣어도 총 파라미터는 3,995,575,808개로 같다. 학습 보조 파라미터와 기저는 별도로 집계한다. 예를 들어 P8 dense의 coefficient·feature head는 15,892,480개 파라미터이고, 기저 원소는 69,050,400개다. 모델 크기와 활성 연산을 분리하는 방향의 선행 근거는 작은 전문가를 검색하는 [PEER](https://arxiv.org/abs/2407.04153)다. 이번 E16 구성은 작은 dense router의 비용을 직접 계산할 수 있다.

희소 본체에는 공통 Q/K/V/O·router·decoder·bridge의 gradient만 우선 예측한다. 전문가에는 c=0을 사용한다. 적은 토큰을 받은 전문가에서 rank-32 행렬을 복원하면 실제 `dW=Xᵀδ`보다 예측 비용이 커질 수 있기 때문이다. 이후 전문가별 predictor를 더할 때는 전체 예측 관측에서 쓸 지원 집합 Aᵢ를 제어 표본 C 선택 전에 정한다. 실제 갱신 집합은 `∪ᵢAᵢ ∪ ∪ᵢ∈C Sᵢ`이며 Sᵢ는 본체의 실제 route다. 제어 표본에서 발견한 Sᵢ로 모든 예측을 사후에 자르면 평균이 바뀐다. 계산 예로 `g₁=(1,0), g₂=(0,1), c₁=(1,1), c₂=(0,0), M=2,m=1`에서 그 mask의 평균은 `(0.25,0.75)`이고 원래 평균은 `(0.5,0.5)`다.

압축과 전문가 선택의 큰 상호작용은 **전문가당 토큰 수와 타일 크기**다. 전문가 e의 토큰 수를 nₑ라 하면, 폭 d와 fₑ가 32의 배수인 현재 GEMM의 SwiGLU 학습 산술량은:

\[
C_e=df_e\{12\lceil n_e\rceil_{16}+6\lceil n_e\rceil_{32}\},
\qquad \lceil0\rceil_b=0.
\]

forward와 dX의 토큰 축은 16으로, dW의 누적 축은 32로 패딩된다. 균등하고 위치별 독립인 top-2 시나리오에서 `nₑ∼Binomial(mN,1/8)`이다. 이 분포를 직접 합산한 결과는 다음과 같다.

| 압축 P | control 두 개의 실행 | 전문가당 평균 토큰 | FFN 타일 산술 / 논리 산술 |
| --- | --- | ---: | ---: |
| 8 | 각 control을 따로 실행 | 8 | 2.669 |
| 8 | 두 control을 전문가별로 함께 packing | 16 | 1.623 |
| 16 | 각 control을 따로 실행 | 4 | 5.259 |
| 16 | 두 control을 전문가별로 함께 packing | 8 | 2.669 |

주 비교표는 **각 control을 따로 실행**하는 비용을 사용한다. 이 방식은 g₁·g₂와 그 차이를 직접 얻어 기울기 진단과 기저 갱신에 사용할 수 있다. 두 control을 합쳐 dW를 한 번 계산하면 합 g₁+g₂를 얻는다. 차이를 사용하는 진단은 signed factors `X=[X₁;X₂], Δ=[δ₁;−δ₂]`에서 `DΩ=Xᵀ(ΔΩ)`처럼 별도 contraction을 계산한다. packing의 최종 손익에는 이 진단과 기저 갱신 비용을 더해야 한다. 표의 Binomial 시나리오는 실제 router의 expert별 nₑ histogram으로 교체한다.

P8에서 P16으로 늘리면 논리적 비용은 0.2082에서 0.1175로 줄고, 타일 비용은 0.3458에서 0.2921로 줄어든다. 압축이 깊어질수록 작은 전문가의 빈 타일이 절감 효과를 잠식한다. 우선 P8과 control packing을 결합하고, P16은 작은 M/K 타일 또는 ragged kernel을 함께 평가하는 순서가 적합하다. rank 8·16도 기존 BK32에서 K축 32를 실행하므로 rank 축소의 실제 산술은 커널 타일과 함께 계산한다.

네 번째 개선은 **같은 제어 수로 잔차 분산을 줄이는 표집과 보정**이다. encoder feature로 자료를 strata h에 나누고, 각 크기를 Mₕ, 비중을 aₕ=Mₕ/M, 제어 수를 mₕ로 두자. 각 stratum에서 균등하게 선택하면:

\[
\widehat G=\bar c+\sum_h a_h\overline{(g-c)}_{C_h},
\quad
\operatorname{tr Var}_C(\widehat G)
=\sum_h a_h^2\frac{1-m_h/M_h}{m_h}S_{r,h}^2.
\]

같은 예측값과 실제 gradient에 대해 평균 항등식을 유지하면서, 서로 다른 자료군 사이의 잔차 평균 차이를 표집 분산에서 제거한다. M=8,m=2의 첫 구성은 4개씩 두 그룹에서 하나씩 고른다. 그룹별 실제 계산 비용 Fₕ와 잔차 표준편차 Sₕ의 과거 추정값을 사용하면 연속 완화의 비용 최적 배분은 `mₕ∝aₕSₕ/√Fₕ`다. 정수화할 때 각 비어 있지 않은 그룹에 최소 한 개를 배정하고 전체 m을 맞춘다. 현재 표의 균등 expert 부하 가정은 strata 선택 후의 실제 부하로 다시 집계한다.

예측의 크기도 과거 통계로 조정할 수 있다. `cᵢ=αUBᵢVᵀ`의 최적 스칼라는 중심화한 `Cov(g,c₀)/Var(c₀)`이며, 공분산은 작은 계수 `Kᵢ=UᵀgᵢV`와 Bᵢ에서 계산할 수 있다. α를 이전 step의 통계로 정하고 현재 step 동안 고정하면 조건부 평균 항등식이 유지된다. 기저가 포착한 변동, coefficient 예측의 오차, α의 보정 효과를 각각 기록한다. 층화·α의 검산은 가능한 제어 집합을 전부 열거해 평균과 분산을 대조했다.

내부 토큰의 dW 표집은 추가 후보로 남긴다. 정확한 forward와 dX를 계산하고 `dW=Σₜxₜδₜᵀ`의 항을 `pₜ∝||xₜ||||δₜ||`로 표집하면, 중요도 보정으로 평균을 유지하면서 dW의 논리적 연산을 줄일 수 있다. token 비율 ρ일 때 선형층의 forward+dX+dW 비율은 `(2+ρ)/3`이다. 외부 제어 표집에 더해지는 분산은 `ΣᵢV_inner,i/(mM)`이므로 두 표집의 잡음도 함께 계산한다. P16의 N=32에서 절반인 16개를 골라도 현재 dW의 BK32 타일 수가 같아, 압축 뒤에는 이 방법의 우선순위를 낮춘다.

같은 품질까지의 판정은 세 비교를 분리한다. 먼저 동일 자료에서 dense byte와 P8 dense와 P8 sparse를 모두 전체 미분으로 학습해 구조 변경의 효과를 측정한다. 다음으로 **같은 구조**에서 전체 M개 미분과 제어 교정을 비교한다. 마지막으로 공유 encoder·지연 갱신·packing을 추가해 개별 비용을 확인한다. 저랭크 SGD 동치는 실제 X·δ의 크기와 정밀도에서 다시 확인하고, 모든 구성의 데이터와 optimizer 조건을 맞춘다.

기울기 예측 단계의 선별 지표는 전체 파라미터에서 중심화한 잔차의 trace variance 비율 q와 같은 구조 대비 FLOPs 비율 γ다. 독립 관측과 고정 예측 함수를 조건으로 원래 M개 평균 대비 분산 비율은 `φ=1+(M/m−1)q`이고, 계산×분산 점수는 γφ다. P8 dense 결합의 γ는 0.2587이다. 아래 q는 측정 결과와 대조할 가정값이다.

| q | 계산×분산 점수 γφ |
| ---: | ---: |
| 0.10 | 0.3363 |
| 0.25 | 0.4528 |
| 0.50 | 0.6468 |
| 1.00 | 1.0349 |

이 점수의 손익 경계는 q≈0.955다. 같은 encoder와 sparse expert의 c=0 그룹이 전체 gradient 분산에서 차지하는 비중을 ρ₀, 예측 그룹의 잔차 비율을 qₛ라 두면 `q=ρ₀+(1−ρ₀)qₛ`다. 예를 들어 ρ₀=0.8,qₛ=0.1이면 q=0.82다. 공통 경로의 높은 예측 정확도와 함께 전문가·encoder의 분산 비중도 측정해야 하는 이유다. 이 점수는 표집 잡음과 연산의 선별 기준이고, 목표 품질까지의 관측 수는 실제 학습 곡선에서 얻는다.

앞선 문서가 인용한 gradient prediction 논문은 버전별 방법이 달라졌다. [v1](https://arxiv.org/html/2511.05187v1)은 본체의 마지막 hidden을 사용한 저랭크 예측과 ViT 분류 실험을 제시한다. [2026년 9월의 v2](https://arxiv.org/html/2511.05187v2)는 본체의 저정밀 reverse-mode 예측을 사용하고, 외부 inference 자원의 시간에 할인율을 적용한 비용 장부로 LLM의 짧은 학습 구간을 평가한다. Sejong의 단일 RX 6600 비교는 모든 실제 연산을 동일하게 합산한다. 공유 encoder만으로 큰 본체의 gradient를 예측하는 성능은 Sejong의 연구 가설이며, q와 목표 bits-per-byte까지의 누적 FLOPs로 검증한다.

결합 구현의 한 step은 다음 순서로 구성한다.

1. θ·φ·U·V·A·α와 표집 정책을 고정하고 M개 자료를 읽는다. E를 실행해 causal 표현과 detached optimizer feature를 얻고 Bᵢ를 예측한다.
2. 사전에 정한 그룹에서 제어 표본을 뽑는다. 선택한 자료에 대해 G·D와 bridge의 forward를 실행하고 언어 loss를 계산한다. 각 선형 연산에서 W₀와 누적 UAVᵀ를 함께 사용한다.
3. 정확한 언어 gradient와 XU·δV를 계산한다. 계수 target, 그룹별 잔차 통계와 control gradient를 누적한다. sparse expert는 선택된 token을 모두 처리한다.
4. W₀에 control gradient를 적용하고 A에 coefficient correction을 누적한다. E와 c=0 그룹은 control gradient로 갱신한다. 현재 step의 target으로 φ·α·표집 통계를 갱신해 다음 step에 적용한다.
5. 32 step마다 A를 W₀에 합친 뒤 기저와 coefficient 좌표를 갱신한다. 체크포인트에는 난수·자료 위치와 유효 가중치를 재현할 상태를 함께 저장한다.

우선 구현 순서는 **P8 계층 전체 미분 → 공유 encoder 교정 → 지연 저랭크 SGD → 같은 용량의 GQA·E16 top-2 → control packing과 층화 선택**이다. 각 단계에서 목표 validation bits-per-byte까지의 누적 FLOPs, 실제 커널 타일, predictor·refresh·optimizer 비중, 그룹별 q, 전문가별 token 수를 기록한다. warmup·통계 수집·기저 초기화·검증·추가 학습 관측의 연산을 총합에 넣어, 다음 단계에 배정할 연산을 결정한다.
