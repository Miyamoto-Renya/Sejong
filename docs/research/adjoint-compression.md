# Sejong의 미분 연산 자체를 줄이는 연구

2026-11-01. 연구 대상은 **역전파가 전달하는 cotangent와 각 연산의 vector–Jacobian product(VJP)**다. 현재 Sejong의 26,488,192개 파라미터 모델에서 실제 역전파 operand를 수집하고, 압축 오차·부분공간 이동·확률적 보정·GPU 실행 비용을 분석했다.

현재 가장 유망한 구현 순서는 **순전파 출력으로 현재 입력의 기저를 얻기 → 입력 VJP와 파라미터 VJP를 분리하기 → 파라미터 VJP의 큰 성분을 인수로 계산하기 → 잔차만 확률적으로 계산하기 → 인수와 선택 행을 optimizer까지 전달하기**다. 이 경로는 현재 loss의 정확한 입력 미분을 유지하며, 큰 가중치 미분을 만드는 연산을 줄인다. 입력 VJP까지 압축하는 확장에서는 현재 미분 방향을 따라가는 기저의 생성 비용이 중심 연구 문제다.

| 산출물 | 실제 수행한 내용 |
| --- | --- |
| [GPU trace 수집기](../../crates/sejong-model/examples/adjoint_trace.rs) | 초기 모델과 66-step 저장 모델의 실제 선형 역전파 입력·가중치·출력·cotangent 저장 |
| [GPU VJP 실행기](../../crates/sejong-model/examples/adjoint_replay.rs) | 실제 FFN operand로 정확 VJP와 채널 블록 VJP, packing·full gradient 복원 비용 측정 |
| [분석 코드](../../scripts/research-adjoint.py) | FP64 spectrum·VJP 오차·확률적 보정의 평균과 분산·기저 이동·순전파 기저 재사용 계산 |
| [수치 결과](adjoint-results.json) | 설정·tensor fingerprint·연산별 spectrum·후보 비용·국소 분산·GPU 시간과 참조식 오차 |
| 학습 실행 경로 | 기존 Rust 수동 역전파와 Triton 커널 실행 |
| 혼합 VJP 학습 설계 | 아래 수식·gradient 형식·커널 계약·비용과 분산을 결합하는 설계 |

후속 [실제 parameter VJP 학습](../parameter-vjp.md)은 5절의 순전파 기저·잔차 보정을 Rust Graph, Triton 커널, 인수 SGD와 NativeValue Adam 경로에 구현한다. 4절의 입력 cotangent까지 압축하는 확장은 해당 절의 비용·분산 설계를 따른다.

GPU는 RX 6600이다. 초기 모델은 seed 42, 저장 모델은 [앞선 측정](../measurements/2026-11-01/README.md)의 정확 SGD 66-step 가중치다. 입력은 저장한 validation 문서의 512 bytes이며 offset 0과 1,024를 사용했다. 첫 입력의 초기·학습 모델 trace는 각각 167개·158개 선형 연산, 두 번째 입력의 학습 모델 trace는 164개 연산이다. 이 수의 차이는 선택된 전문가의 실제 사용량에서 생긴다. Spectrum과 기저 실험은 NumPy FP64, GPU VJP는 원래 F16 순전파 operand와 F32 미분을 사용한다. Trace 실행은 전체 tape를 구성해 원래 연산의 Var를 보존한다.

## 1. 줄일 연산과 비용의 하한

선형층을 `Y=XW`, `X∈R^(N×a)`, `W∈R^(a×b)`, 출력 미분을 `D∈R^(N×b)`로 두자.

\[
G_X=DW^\top,\qquad G_W=X^\top D.
\]

일반 GEMM 산술은 forward `2Nab`, 두 VJP의 합 `4Nab`다. 이 식의 `G_W`는 이미 두 인수의 곱으로 실행된다. 새 압축이 줄여야 하는 축은 행렬곱의 공통 축 N과 출력 미분의 유효 차원이다.

`D=ABᵀ`, `A∈R^(N×r)`, `B∈R^(b×r)`를 알고 있으면:

\[
G_X=A(WB)^\top,\qquad G_W=(X^\top A)B^\top.
\]

두 VJP의 비용은 `4ar(N+b)`다. 기저를 주어진 값으로 두고 D를 투영하는 비용 `2Nbr`도 더한다. 현재 신호에서 기저를 얻는 연산, 비선형 경계의 변환, 메모리 이동은 후속 항이다.

같은 dense forward를 유지한 `F+2F` 모델에서 backward를 전부 줄여 얻는 산술 배수의 극한은 3이다. Forward 재계산을 한 번 유지한 `2F+2F` 모델에서는 2다. 이는 forward·재계산 조건을 고정한 비용식이다. 구조 변경·재계산 스케줄·학습량 변화는 각각의 항으로 더한다.

현재 P8 global의 N은 64이고, 실제 전문가의 N은 이보다 작아진다. 현 GEMM의 BM16·BN32·BK32에서는 rank 8과 16도 K축 32를 실행한다. 구간 압축이 이미 줄인 N과 미분 압축의 공통 축을 함께 계산해야 한다. [실제 커널](../../crates/sejong-gpu/src/lib.rs)의 타일을 분석기에 전달했다.

## 2. 비선형층에서 보존되는 것과 커지는 것

정확한 순전파 값 x를 고정한 비선형 연산 `y=f(x)`의 VJP는 `J_f(x)ᵀD`다. 이 연산은 들어오는 미분 D에 선형이다. 따라서:

\[
\mathbb E[\widehat D\mid D,x]=D
\quad\Longrightarrow\quad
\mathbb E[J_f(x)^\top\widehat D\mid D,x]=J_f(x)^\top D.
\]

각 지점에서 현재 들어온 신호를 조건으로 새로운 표집을 하면 이 평균식을 그래프의 역순으로 적용할 수 있다. [RAD](https://arxiv.org/abs/2007.10412)와 [2026년 VJP 연구](https://arxiv.org/html/2602.14701v1)는 확률적 미분과 VJP별 분산 전파를 다룬다. [활성값 압축 연구](https://arxiv.org/html/2605.01255v1)는 저장한 순전파 활성값을 바꾸었을 때 선형·비선형 연산에서 생기는 오차를 분석한다. 이 연구에서는 정확한 순전파 값에서 Jacobian을 평가한다.

SwiGLU의 `H=SiLU(G)⊙U`를 보면:

\[
 D_G=D_H\odot U\odot\operatorname{SiLU}'(G),\qquad
 D_U=D_H\odot\operatorname{SiLU}(G).
\]

`rank(D_H)=1`이어도 오른쪽 원소별 곱의 rank는 `min(N,f)`까지 커진다. 예를 들어 `D_H=11ᵀ`, `G=0`, `U=2I`이면 `D_G=I`다. 8×8 검산에서 rank는 1에서 8로 바뀌었다. 저랭크 행렬 두 개의 원소별 곱은 각각 rank r·s일 때 최대 rs개 인수로 표현된다. 비선형 미분의 인수를 유지하려면 이 증가를 다시 압축하는 연산과 분산을 계산한다.

RMSNorm에서는 `v_t=(mean(x_t²)+eps)^(-1/2)`일 때:

\[
 D_X=\operatorname{diag}(v)(D_Y\odot w)
 -\operatorname{diag}\!\left(\frac{v^3}{d}
       \langle D_Y\odot w,X\rangle_{\rm row}\right)X.
\]

첫 항은 저랭크 인수를 유지한다. 두 번째 항은 현재 X의 행별 배율이므로 X의 rank를 함께 가진다. Attention의 score 미분은 `P⊙(D_P−row_sum(P⊙D_P))`와 causal mask를 포함한다. Head 안의 token 결합과 원소별 곱을 별도 연산으로 다룬다. GQA의 공유 K/V 미분은 선택한 query head의 기여를 합산한다.

## 3. 실제 미분 신호의 압축률

아래 표는 66-step 모델·offset 0의 **각 연산에 들어오는 정확한 D를 독립적으로 SVD 압축**한 결과다. 에너지 rank와 버린 에너지 비율은 연산별 값의 중앙값이다.

| 경로 | 선형 연산 수 | 90% 에너지 rank | 99% 에너지 rank | rank 16의 잔차 에너지 |
| --- | ---: | ---: | ---: | ---: |
| Encoder | 14 | 63 | 110.5 | 41.75% |
| Decoder | 14 | 42 | 87 | 35.11% |
| Global 공유 Q/K/V/O·router | 40 | 8.5 | 25.5 | 3.04% |
| Global 전문가 | 84 | 19 | 33 | 13.00% |

전체 선형 연산에서 현재 SVD 기저를 주어진 값으로 둔 rank-16 후보와 원래 VJP 중 산술이 작은 경로를 고르면, backward issued FLOPs는 55.84%, forward·global 재계산까지 합친 선형 GEMM은 75.42%다. 이때 연산별 `dW` 오차 제곱합 / 원래 `dW` 제곱합은 0.07373이다. 이 수치는 각 연산을 독립적으로 압축한 오차와 기저가 주어진 GEMM 비용이다. 압축 신호를 전체 그래프에 연결한 분산은 6절의 식으로 계산한다.

현재 SwiGLU Jacobian 인수도 분석했다. Decoder 마지막 FFN의 `U⊙SiLU'(G)`는 99% 에너지에 rank 144, `SiLU(G)`는 rank 108이 필요했다. Encoder 마지막 FFN에서는 각각 192·158이었다. 저랭크 cotangent와 이 인수들을 곱할 때 생기는 rs 비용을 명시적으로 처리해야 한다.

## 4. 정확한 큰 성분과 확률적 잔차를 결합하는 VJP

고정하거나 현재 자료에서 만든 feature 기저 `Q∈R^(b×r)`에 대해:

\[
 H=DQ,\quad R=D-HQ^\top,
 \qquad
 \widehat D=HQ^\top+
 \sum_{j\in C}\frac{R_{:,j}}{p_j}e_j^\top.
\]

여기서 C는 channel block 표집이며 p는 해당 block의 포함 확률이다. Q와 현재 D를 조건으로 표집하면 `E[D_hat]=D`다. 모든 관측의 현재 `DQ`를 직접 계산한다. 미분 예측기를 학습하는 비용은 이 식의 항과 구분된다.

이 미분을 선형 VJP에 넣으면:

\[
\begin{aligned}
\widehat G_X&=H(WQ)^\top+(R_C/p)W_{:,C}^\top,\\
\widehat G_W&=(X^\top H)Q^\top+
        \operatorname{scatterColumns}_C(X^\top(R_C/p)).
\end{aligned}
\]

전체 R을 복원하는 대신 `R_C=D_C−H Q_Cᵀ`만 계산한다. 선택 channel에 대응하는 W의 열도 직접 읽는다. 독립 Bernoulli block의 국소 분산은:

\[
 V_W=\sum_j(1/p_j-1)\|X^\top R_j\|_F^2,
 \qquad
 V_X=\sum_j(1/p_j-1)\|R_jW_j^\top\|_F^2.
\]

동일한 확률 p에서는 `V_W=(1/p−1)||XᵀR||²`다. 이는 현재 선택한 부분공간이 실제 가중치 미분에서 놓치는 양을 직접 측정하는 식이다. 고정 개수 비복원 표집의 입력 VJP 분산에는 block 사이 공분산도 포함한다.

`C_W=WQ`는 같은 step의 여러 microbatch에서 공유할 수 있다. SGD로 한 번 갱신할 때 `Z_H=XᵀH`, `Z_C=Xᵀ(R_C/p)`라 두면:

\[
 C_W' = C_W-\eta\{Z_H(Q^\top Q)+Z_CQ_C\}.
\]

즉 큰 W를 다시 투영하는 비용을 작은 coefficient 갱신으로 바꾸는 확장도 가능하다. 이 항등식은 같은 Q, 선형 SGD 업데이트를 조건으로 한다. Adam의 원소별 상태 변환을 적용할 때는 그 변환 이후의 projection 비용을 추가한다. 실제 tensor의 8×8 부분행렬에서 cache 갱신 항등식의 최대 절대 오차는 2.08×10⁻¹⁷ 이하였다.

다른 입력에서 기저를 재사용한 결과는 다음과 같다. 두 trace의 대응하는 가중치는 bit 단위로 대조했다. 기저는 offset 0에서 얻고 offset 1,024의 미분을 처리했다.

| 기저 구성 | 미분 잔차 에너지 중앙값 | 비용에 따라 혼합 VJP를 택한 연산 | 선형 forward·재계산·backward 비용비 |
| --- | ---: | ---: | ---: |
| 이전 입력의 rank 16 | 67.28% | 125 / 164 | 0.8214 |
| 이전 rank 16 + 현재 잔차의 16방향, power iteration 1회 | 12.90% | 6 / 164 | 0.9843 |

두 번째 후보는 현재 잔차의 range를 `DᵀΩ`로 얻고 기존 Q에 직교화한 뒤 한 번의 부분공간 반복을 수행했다. 추가 GEMM, 새 `WQ`, 두 번 직교화하는 QR의 산술 예산을 포함했다. 현재 기저를 따라가면 오차는 크게 줄며, 그 기저를 만드는 연산이 비용의 대부분을 사용한다.

## 5. 순전파 출력에서 기저를 얻는 파라미터 VJP

위 결과를 바탕으로 현재 forward의 `Y=XW`를 기저 재료로 사용했다. Y의 r개 열은 이미 계산된 `X W_C`다. 이 열들을 직교화해 token 기저 `Q∈R^(N×r)`를 만들면 별도의 `XΩ` GEMM을 줄일 수 있다. 같은 X를 사용하는 Q/K/V나 gate/up은 기저와 `XᵀQ`를 공유한다.

\[
 A=X^\top Q,\quad B=Q^\top D,\quad E=X-QA^\top.
\]

가중치 미분은 다음처럼 나뉜다.

\[
 G_W=AB+E^\top D.
\]

입력 feature block C를 표집해 잔차의 가중치 행만 계산한다.

\[
 \widehat G_W=AB+
   \operatorname{scatterRows}_C(E_{:,C}^\top D/p),
 \quad E_{:,C}=X_{:,C}-QA_{C,:}^\top.
\]

Q와 현재 X·D를 조건으로 `E[G_W_hat]=G_W`다. Q의 정규직교성은 근사 품질과 수치 안정성을 개선한다. 이 분해의 평균 항등식은 계산한 잔차 E를 통해 성립한다. 입력 VJP `DWᵀ`는 현재 정확한 연산을 수행한다. 따라서 이 위치에서 추가된 파라미터 미분 오차가 이전 층의 cotangent로 전파되는 경로를 분리할 수 있다.

동일 확률의 block 표집에서:

\[
 \mathbb E\|\widehat G_W-G_W\|_F^2
 =(1/p-1)\|E^\top D\|_F^2.
\]

이 식은 활성값의 재구성 오차에 더해 **그 오차가 현재 loss 미분과 수축한 양**을 측정한다. [활성값·기울기 공동 압축 연구](https://arxiv.org/html/2605.01255v1)와 연결되는 지점이다. 여기서는 현재 forward의 Y를 기저로 사용하고, 정확한 잔차의 feature block으로 평균을 교정한다.

Offset 1,024의 164개 실제 연산에서 rank 16과 feature block 1/4을 분석했다. 기저 QR은 두 번 직교화하는 산술 예산, GEMM은 실제 타일을 적용했다. 같은 입력을 쓰는 연산 사이의 기저·투영 공유를 포함했다.

| 경로 | 파라미터 VJP 비용비 | 정확한 입력 VJP 포함 backward 비용비 | 선형 forward·재계산 포함 비용비 | 연산별 파라미터 미분 상대 MSE 합산값 |
| --- | ---: | ---: | ---: | ---: |
| Encoder | 0.5721 | 0.7861 | 0.8574 | 0.3411 |
| Decoder | 0.5721 | 0.7861 | 0.8574 | 0.1390 |
| Global 공유 | 0.8380 | 0.9190 | 0.9595 | 0.00324 |
| Global 전문가 | 0.8974 | 0.9447 | 0.9712 | 0.05941 |
| 전체 선형 연산 | 0.7823 | 0.8873 | 0.9376 | 0.1565 |

비용 선택은 164개 중 109개에 압축을 적용했다. 나머지 형상에는 원래 정확한 연산의 비용을 넣었다. MSE는 각 연산의 정확한 D를 고정한 조건부 표집 분산을 합산한 값이다. 같은 가중치를 여러 번 사용하는 경우 모델 gradient의 합과 연산별 norm 합을 따로 측정한다. GPU packing·직교화 wall time·optimizer를 연결하는 단계에서는 다음 절의 실제 비용을 함께 적용한다.

이 후보에서 global 공유 경로는 비교적 작은 추가 미분 분산으로 파라미터 VJP 연산을 줄였다. Encoder는 더 큰 분산을 보였으므로 같은 rank·표집률을 모든 층에 일괄 적용하는 정책보다 층별 예산을 계산하는 정책이 적합하다.

## 6. 분산 전파와 보정의 조건

정확한 순전파에서 고정된 전체 gradient를 g라 하고, 각 미분 압축 지점의 조건부 평균 0인 오차를 η_l이라 하자. 그 지점의 오차를 모델 parameter 공간으로 옮기는 정확한 선형 사상을 H_l이라 두면:

\[
 \widehat g-g=\sum_l H_l\eta_l.
\]

역전파 순서의 새 표집이 각 지점의 조건부 평균을 유지하면 이 오차들은 martingale difference를 구성한다. 따라서 전체 제곱오차는 `Σ E||H_l η_l||²`로 합산된다. 초기 층까지 전달되는 Jacobian의 영향이 이 식에 들어 있다. [2026년 VJP 연구](https://arxiv.org/html/2602.14701v1)의 국소 분산·이전 지점 분산 전파 분석을 Sejong의 DAG와 공유 parameter에 적용하는 기준이다.

학습 중인 모델의 자료 표집 분산을 σ², 미분 근사로 더한 분산을 V, step 비용비를 γ라 두면, 분산이 지배하는 SGD 상한에서 선별 지표는:

\[
 \gamma(1+V/\sigma^2).
\]

σ²는 같은 가중치에서 여러 관측의 gradient로 추정한다. V에는 전체 parameter에 도달한 오차를 사용한다. 이번 국소 trace 분석은 그 V를 구성할 재료와 비용 후보를 제공한다. 실제 목표 BPB까지의 판정은 같은 데이터·optimizer·초기값의 누적 시간과 FLOPs로 수행한다.

기저가 주어진 채 orthogonal residual의 k차원 공간에서 s개 방향을 균등 선택하면:

\[
 \mathbb E\|\widehat D-D\|_F^2=(k/s-1)\|D_{\perp}\|_F^2.
\]

실제 cotangent의 8×8 부분행렬에서 가능한 잔차 부분집합을 전부 열거해 평균과 분산을 검산했다. 평균 상대 제곱오차는 1.03×10⁻³⁰ 이하, 분산식 상대 오차는 1.25×10⁻¹⁵ 이하였다. 같은 projector를 직렬 두 지점에서 재사용한 반례는 `E[P²]` 때문에 상대 제곱 편향 9를 보였다. FFN 한 지점의 DG·DU·가중치 기여를 같은 한 번의 표집으로 계산하는 경우에는 각 경로에 역확률을 한 번 적용한다.

국소 미분 잔차를 다음 optimizer step으로 넘길 때도 Jacobian을 추적해야 한다. `e_(t+1)=D_t+e_t−C(D_t+e_t)`와 `D_hat_t=D_t+e_t−e_(t+1)`이면:

\[
\sum_{t=0}^{T-1}J_t^\top\widehat D_t
=\sum_tJ_t^\top D_t+J_0^\top e_0-J_{T-1}^\top e_T
 +\sum_{t=1}^{T-1}(J_t^\top-J_{t-1}^\top)e_t.
\]

입력과 가중치가 바뀌는 만큼 마지막 drift 항이 생긴다. 실제 tensor 부분행렬에서 이 항등식을 대조한 최대 절대 오차는 2.12×10⁻²²였다. [PowerSGD](https://arxiv.org/abs/1905.13727)의 parameter gradient 통신 압축과 시간에 따라 Jacobian이 달라지는 내부 cotangent 누적을 각각의 좌표계에서 다룬다.

## 7. SwiGLU 블록 미분의 GPU 측정

SwiGLU FFN의 중간 channel j는 해당 j의 gate 열·up 열·down 행으로 연결된다. 정확한 forward 이후 이 묶음의 일부를 골라 역확률을 한 번 적용하면, 모든 token에 대한 FFN 미분을 선택 channel에서 계산할 수 있다. 이 구조는 원소별 비선형이 channel 지지 집합을 유지하는 성질을 사용한다.

전체 hidden 폭 f, 실행 비율 p에서 일반 SwiGLU의 주요 산술은 forward `6Ndf`, backward `12Ndf`이고, 이 후보의 backward는 `12Ndfp`다. 전체 FFN 비용비는 재계산을 해제하면 `(1+2p)/3`, forward를 한 번 재계산하면 `(1+p)/2`다.

가중치 gradient에서 channel 묶음들의 지지 집합은 서로 겹치지 않는다. 균등한 p의 상대 MSE는 정확히 `1/p−1`이다. p=1/4이면 3이다. 현재 encoder·decoder FFN의 현재 block 에너지를 알고 최적 Bernoulli 중요도 표집을 적용한 경우에도 대략 2.7–2.9였다. 큰 성분을 먼저 보존하는 보정이 필요한 수치적 이유다.

실제 학습 모델의 FFN 입력을 고정하고, hidden 384 중 96개를 선택해 RX 6600에서 2회 준비 후 32회씩 실행했다.

| 실제 FFN 형상 N×d×f | 정확 VJP µs | 선택 입력 준비 완료·compact gradient µs | 매회 packing·full gradient 복원 µs |
| --- | ---: | ---: | ---: |
| 512×128×384 | 471.39 | 190.64 | 496.29 |
| 64×512×384 | 266.13 | 109.40 | 408.07 |
| 1×512×384 | 131.63 | 80.66 | 382.09 |

정확 VJP는 8 launch, 준비된 compact VJP는 9 launch, packing과 full 복원 경로는 35 launch였다. 측정 구간의 신규 컴파일 수는 0이었다. 위 값은 GPU 동기화를 포함한 해당 FFN VJP의 wall time이다. Forward는 수집 시 수행했고, optimizer와 새로운 표집 인덱스 생성은 별도 비용이다. 준비된 입력 경로의 2.4배 안팎 속도와 매회 준비·복원 경로의 시간을 함께 사용해 구현 우선순위를 정한다.

각 GPU 결과는 같은 선택 인덱스와 역확률을 적용한 NumPy FP64 참조식과 대조했다. `dX,dW_gate,dW_up,dW_down`의 상대 제곱오차는 모든 경우 2.84×10⁻¹³ 이하였다. 이 오차는 선택한 확률적 연산의 GPU 수치 오차이며, 원래 전체 gradient에 대한 표집 분산은 위의 별도 항이다.

## 8. 미분 표현과 optimizer를 함께 구현하는 계약

다음 구현의 gradient 형식은 `LowRankAndIndexedRows { A, B, rows, tail }`이다. `G=AB+scatterRows(tail)`을 표현한다. 입력 VJP를 압축할 때는 cotangent에도 같은 형태의 열 인덱스와 인수를 둔다.

1. Forward의 같은 입력을 쓰는 연산들을 묶고, donor 출력·rank·현재 QR의 유효 방향을 기록한다. Q·XᵀQ와 선택 feature 목록을 공유한다.
2. 정확한 incoming cotangent에서 입력 VJP를 계산한다. Parameter VJP의 head와 선택 잔차를 계산한다.
3. 인덱스 GEMM은 원래 tensor에서 선택한 행·열을 직접 읽는다. Residual kernel은 필요한 `X_C−Q A_Cᵀ`만 만든다.
4. Optimizer는 head의 GEMM과 선택 tail을 결합해 W를 갱신한다. Full gradient 복원에 들어간 35-launch 경로를 compact gradient의 소비 경로로 교체한다.
5. RNG·기저 정책·예산 통계·step을 checkpoint에 저장한다. 같은 초기값·자료·RNG에서 재개 결과를 대조한다.

Norm도 인수에서 계산할 수 있다. `G=AB+S_C T`이면:

\[
 \|G\|_F^2
 =\operatorname{tr}[(A^\top A)(BB^\top)]
 +\|T\|_F^2+2\langle A_C B,T\rangle.
\]

Gram·tail·교차항에서 clipping과 finite 검사에 필요한 전체 gradient norm을 얻는다. 여러 microbatch를 합친 경우에는 공통 parameter의 인수·tail 사이 교차항도 포함한다. 저장과 optimizer 갱신에 실제로 적용하는 추정 gradient를 기준으로 계산한다.

예산은 현재 shape의 타일 후보와 관측한 비용·분산으로 정한다. 독립 Bernoulli block에서 분산 항 v_j와 비용 c_j가 주어지면 Lagrange 조건은 `p_j=min(1,sqrt(v_j/(lambda*c_j)))`다. lambda는 전체 계산 예산으로 정한다. 과거 통계의 불확실성을 포함한 양의 포함 확률을 사용하고, 현재 gradient를 구성하는 동안 정책을 고정한다. 고정 개수 표집에서는 joint inclusion 확률과 실제 타일 비용을 사용한다.

현재 QR 구현은 열마다 한 kernel을 실행한다. Rank 16이면 16 launch다. GPU 구현의 다음 핵심은 얇은 행렬 전체를 한 번에 처리하는 QR·선택 load GEMM·인수 norm·SGD 갱신을 연결해, 기저와 표현을 만드는 비용을 줄이는 일이다. 형상별 비용 선택은 N=1 전문가처럼 launch와 타일 비율이 큰 경우 원래 경로의 비용도 함께 평가한다.

## 9. 선행연구와 Sejong의 다음 실험

| 선행연구 | 이번 연구에서 참조한 지점 |
| --- | --- |
| [RAD, 2021](https://arxiv.org/abs/2007.10412) | 계산 그래프 안에서의 확률적 미분과 분산 |
| [Approximate Tensor Operations, NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/file/eaa1da31f7991743d18dadcf5fd1336f-Paper.pdf) | Backward 행렬곱의 표집과 GPU 비용. MLP·CNN의 평가 조건 |
| [Unbiased Approximate VJPs, 2026](https://arxiv.org/html/2602.14701v1) | 국소 조건부 평균, 분산 전파, 데이터별 표집 확률 |
| [Activation–Gradient Co-Compression, 2026](https://arxiv.org/html/2605.01255v1) | 저장 활성값과 선형층 parameter gradient의 공동 인수 사용 |
| [Optimal Low-Rank Stochastic Gradient Estimation, 2026](https://arxiv.org/html/2603.20632v1) | 무작위 부분공간의 분포와 지연 갱신을 함께 다루는 접근 |
| [GaLore, 2024](https://arxiv.org/abs/2403.03507) | 이미 계산한 gradient의 부분공간과 optimizer 상태 저장 |
| [CoLA, 2025](https://arxiv.org/html/2502.10940) | 활성값의 저랭크 구조를 모델 구조에 반영한 pretraining 비교 |

Sejong에서 연구할 결합은 **현재 forward 출력의 재사용, 입력·parameter VJP 분리, 정확한 국소 잔차 표집, compact gradient를 소비하는 Rust·Triton 학습기**다.

다음 실험은 동일한 본체·optimizer·관측 수에서 정확 역전파, 순전파 기저 parameter VJP, 혼합 cotangent VJP를 비교한다. 먼저 기저·packing·norm·optimizer를 포함한 실제 비용과 여러 관측의 parameter gradient 분산을 잰다. 그 다음 같은 검증 BPB에 도달하는 누적 FLOPs와 시간으로 판단한다. 현재 분석에서 global 공유 경로는 작은 국소 추가 분산을 보였고, encoder·작은 전문가에는 서로 다른 예산이 필요했다.

핵심 연구 과제는 **큰 성분의 현재 방향을 저렴하게 얻고, 나머지 미분을 적은 분산으로 계산하며, 압축한 표현을 다음 연산까지 유지하는 것**이다. 이번 결과는 그 세 항을 실제 모델 신호와 RX 6600 비용으로 분해했다.

재현 입력은 앞선 측정의 저장 모델과 validation snapshot이다. 아래 분석은 rank·에너지 목표·표집률·타일·공유 횟수·현재 방향 보충 예산을 모두 명시한다.

```bash
source scripts/env.sh
cargo build --release -p sejong-model --example adjoint_trace --example adjoint_replay --locked
target/release/examples/adjoint_trace configs/sejong.toml \
  docs/measurements/2026-11-01/validation.txt .cache/adjoint-initial 512 0
target/release/examples/adjoint_trace configs/sejong.toml \
  docs/measurements/2026-11-01/validation.txt .cache/adjoint-trained 512 0 \
  runs/compute-rx6600-learning-window/run-000/model
target/release/examples/adjoint_trace configs/sejong.toml \
  docs/measurements/2026-11-01/validation.txt .cache/adjoint-trained-next 512 1024 \
  runs/compute-rx6600-learning-window/run-000/model
target/release/examples/adjoint_replay .cache/adjoint-trained decoder.1 \
  .cache/adjoint-replay-local 96 32 2 32
target/release/examples/adjoint_replay .cache/adjoint-trained layers.4.experts.1 \
  .cache/adjoint-replay-global 96 32 2 32
target/release/examples/adjoint_replay .cache/adjoint-trained layers.4.experts.3 \
  .cache/adjoint-replay-tiny-expert 96 32 2 32
OPENBLAS_NUM_THREADS=1 python scripts/research-adjoint.py \
  --trace .cache/adjoint-initial --trace .cache/adjoint-trained \
  --out docs/research/adjoint-results.json \
  --ranks 4,8,16,32,64 --energies 0.9,0.99 --fractions 0.25,0.5,1 \
  --block 32 --tile 16,32,32 --projection-reuses 8 \
  --hybrid-reference .cache/adjoint-trained --hybrid-target .cache/adjoint-trained-next \
  --fresh-rank 16 --power-steps 1 --detail-rank 16 --detail-fraction 0.25 \
  --replay .cache/adjoint-replay-local .cache/adjoint-trained \
  --replay .cache/adjoint-replay-global .cache/adjoint-trained \
  --replay .cache/adjoint-replay-tiny-expert .cache/adjoint-trained
```
