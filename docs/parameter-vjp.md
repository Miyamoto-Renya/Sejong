# 순전파 기저와 잔차 보정을 사용하는 실제 학습

Sejong의 `training.parameter_vjp`는 선형층의 가중치 미분을 현재 순전파 기저와 선택한 잔차 행으로 계산한다. 입력 미분은 `D Wᵀ`를 실행한다. Rust Graph가 압축 gradient를 parameter leaf와 checkpoint 재계산 경계를 지나 전달하고, SGD는 인수와 선택 행을 직접 읽어 원래 FP32 가중치를 갱신한다.

`configs/sejong.toml`과 `configs/sejong-parameter-vjp.toml`은 같은 계층·GQA·전문가 본체에 이 미분을 적용한다. `configs/sejong-sparse.toml`은 기존 정확 SGD 비교군이고, `configs/sejong-parameter-exact.toml`은 새 전역 SGD 경로에서 전체 잔차를 관측하는 정확 미분 비교군이다.

[준비 비용을 줄인 RX 6600의 66-step 비교](measurements/2026-11-01-parameter-vjp-overhead/README.md)에서 기존 B1/A8 설정의 평균 step 시간은 초기 압축 구현의 0.278→0.259초로 줄었다. 같은 실행의 기존 SGD는 0.263초, 전역 타일 SGD의 정확 미분은 0.257초였다. 당시 압축 설정은 주요 issued FLOPs 107.19 GFLOPs, 검증 BPB 7.934876을 기록했다. [초기 구현 측정](measurements/2026-11-01-parameter-vjp/README.md)에 입력 묶음마다 기저를 생성하던 비용을 보관했다.

현재 기본 구성은 활성값 보존·B8/A1·F16 dot2를 사용한다. [생산 경로의66-step 실행](measurements/2026-11-01-architecture-fit/README.md)에서 평균111.436ms, issued77.793GF/step, 검증 BPB7.934923을 기록했다. Global 행 수512에서 QKV의 공유 압축 VJP를 실행한다.

```toml
[training.parameter_vjp]
rank = 16
residual_fraction = 0.25
block_rows = 32
seed = 124
```

Rank 16과 잔차 1/4은 [실제 미분 연구](research/adjoint-compression.md)의 비교 예산이다. Block 32는 기존 GEMM의 reduction·output 타일과 함께 비교한 행 크기다. 실행기는 `N≥입력 폭`인 긴 입력과 여러 parameter 행렬의 공유 입력 묶음에 압축을 적용한다. 그 후보의 실제 폭·공유 사용 수·타일 패딩·QR 산술을 비교한다. 짧은 입력과 단독 행렬은 정확 미분을 실행한다. 비교 프로파일의 seed는 자료 표집 RNG와 미분 표집 RNG를 분리해 재현하는 값이다.

`Y=XW`의 현재 출력 열을 비복원 균등 선택해 token 기저 Q를 만든다. 같은 입력을 사용하는 Q/K/V 또는 gate/up은 투영과 잔차를 공유한다. 같은 관측에서 `(행 수, rank)`가 같은 묶음은 Q를 공유한다. 첫 묶음의 가장 넓은 parameter 행렬곱 출력을 donor로 사용한다. Rank는 설정값과 실제 token 수·입력 폭·donor 폭의 최솟값이다. `basis_donor_width`와 `selected_columns`는 해당 Q를 생성한 donor의 열 정보를 보존한다.

\[
A=X^\top Q,\quad B=Q^\top D,\quad E=X-QA^\top,
\qquad G_W=AB+E^\top D.
\]

입력 feature를 `block_rows`씩 묶고 전체 block B개에서 `ceil(B × residual_fraction)`개를 비복원 균등 선택한다. 포함 확률 p는 선택 수 / 전체 block 수다. 선택 C에 대해 `E_C=X_C−Q A_Cᵀ`만 계산하고, `T=E_CᵀD/p`를 만든다. 실제 gradient는 `AB+scatterRows_C(T)`다. 현재 X·D·Q를 조건으로 표집 평균이 `XᵀD`와 일치한다. 마지막 partial block의 빈 위치는 -1 인덱스와 zero 행으로 표현한다. GPU SplitMix64와 rejection sampling은 Rust RNG의 비복원 선택 결과와 대조했다.

같은 관측의 입력 폭이 같은 층은 feature 선택표와 row map을 공유한다. 각 입력의 A와 선택 잔차는 개별 계산한다. 공유 선택은 층 사이의 표집 상관을 만들고, 각 parameter의 표집 평균은 같은 포함 확률 보정식을 따른다. 다음 관측의 Graph는 기저와 선택표를 새로 생성한다. 현재 B8/A1 기본 구성의66-step 측정에서 step당 기저는 localN4096·globalN512의2개, 투영 묶음은16개였다. Expert routing에 따라 추가 압축 묶음을 선택한다. `basis_groups`와 `projection_groups`가 두 수를 구분한다.

Triton 커널의 연결은 다음과 같다.

| 구현 | 실행 |
| --- | --- |
| `parameter_indices` | GPU에서 donor 열·잔차 block 선택, 원래 feature 행에서 compact 행으로 가는 map 작성 |
| `parameter_basis_register` | 작은 기저 행렬을 register 표현으로 처리하는 CGS2 |
| `parameter_basis` | Row·rank tile을 순회하는 streaming CGS2 |
| `parameter_prepare` | 타일과 정렬된 feature block·작은 rank의 입력 투영과 선택 잔차를 한 커널에서 계산 |
| `parameter_residual` | 원래 X의 선택 열을 직접 읽고 잔차를 계산하며 Qᵀ와 연결된 projection 작성 |
| GEMM | 연결된 projection과 D를 한 번 곱해 B·T 생성 |
| `parameter_gradient_update` | 인수·tail·dense 항을 타일 안에서 합산하고 원래 가중치에 SGD 적용 |
| `parameter_gradient_materialize` | Adam이 사용할 실제 합산 gradient를 출력 |
| `parameter_gradient_norm` | clipping·진단에서 실제 합산 gradient의 FP64 제곱합을 계산 |

Register 실행은 GPU가 보고한 register 예산과 실제 기저의 작업량을 사용한다. 컴파일된 shared-memory 크기는 HIP의 block당 한도와 대조한다. 큰 기저와 shared-memory 한도를 넘는 형상은 streaming 실행을 사용한다. 선택 결과와 보정 확률은 같은 RNG 상태에서 구성한다. Resource 판정은 커널 형상별로 보관한다.

작은 rank가 GEMM의 BN·BK 안에 들어가고 feature block과 입력 폭이 BM에 정렬되면 `parameter_prepare`가 `XᵀQ`를 register에 보존해 선택 잔차까지 계산한다. 나머지 형상은 별도 입력 투영과 `parameter_residual`을 실행한다. 두 경로는 같은 수식과 표집 결과를 사용한다.

기저의 두 번 직교화와 작은 방향의 제외 기준은 FP32 unit roundoff와 N·rank의 연산 수에서 계산한다. 선택 행·열·grid·descriptor·allocation 크기는 실제 tensor shape와 checked 크기 계산을 사용한다. GEMM 타일과 비용 선택기는 같은 BM16·BN32·BK32 상수를 참조한다. 후보 비용에 QR과 공유 투영을 포함하고, 비용이 줄어드는 parameter 행렬에 압축을 적용한다.

Graph는 `ParameterGradient { dense, terms }`로 미분을 전달한다. 각 term은 `left`, `right`, `tail`, `row_map`, `scale`을 가진다. Cast와 checkpoint 재계산은 이 항들을 parameter leaf까지 보존한다. 누적 배치의 평균과 loss scale은 term scale에 적용한다. Optimizer는 여러 관측의 rank 축을 연결한 인덱스로 읽어 같은 BK 타일을 채우고, 선택 잔차와 dense 항을 합산한다. 가중치 미분의 전체 행렬을 별도 buffer에 만드는 단계가 SGD 갱신에서 줄어든다.

QKV·gate/up의 결합 순전파도 원래 Matmul node와 출력 순서를 사용한다. 기존 input 묶음·donor 동률 선택·RNG 진행과 FactoredRows를 같은 leaf에 전달한다. [실제 본체의 통합 결과](measurements/2026-11-01-fused-applied/README.md)는 step당 기저8개·투영묶음64개·압축행렬160개와 학습·재개를 확인했다.

SGD 갱신 커널은 합산 gradient와 weight decay를 적용해 가중치를 저장한다. Clipping이나 `gradient_diagnostics_every`를 선택한 step은 실제 합산 gradient의 FP64 norm을 계산한다. Log의 `gradient_norm`은 해당 step의 값이고, 나머지 step은 null이다. SGD는 clipping 배율과 분리된 원래 학습률로 weight decay를 적용한다. [검사 비용 정리](research/check-overhead-changes.md)에 norm 상한·갱신 후 scan 제거와 준비 경계 변경을 정리했다.

모든 관측은 기존 언어모델 순전파·loss·입력 미분 경로를 실행한다. `observed_bytes`는 읽은 관측량, `backward_bytes`는 역전파 관측량이다. 가중치 미분을 압축한 step의 `exact_bytes`는 null이며, 전체 행렬을 정확하게 미분한 step은 실제 byte 수를 기록한다. `benchmark`가 활성화하는 FLOPs 계측은 GEMM·attention 내적·기저 QR 산술과 optimizer 안의 인수 행렬곱을 집계한다. Kernel launch 수는 실제 GPU launch 수다. 일반 학습의 계측 상태는 `execution.json`의 `metrics_collected`에 기록한다.

Byte 학습은 `optimizer="sgd"`와 이 정책을 함께 사용한다. 체크포인트는 기존 모델 가중치와 학습 설정, corpus RNG, 미분 표집 RNG를 저장한다. 데이터·정책·배치·schedule을 대조한 뒤 재개한다. 추론은 저장한 원래 가중치를 사용한다.

구간 broadcast의 입력 미분은 각 수신 구간이 담당하는 토큰을 고정 순서로 합산한다. BOS 미분은 모든 배치의 첫 구간을 같은 순서로 합산하고, 마지막 구간의 미사용 입력 위치에는 zero gradient를 기록한다. 이 합산 방식은 압축 미분과 정확 미분의 공통 실행 경로에 적용된다.

```bash
source scripts/env.sh
cargo build --release -p sejong-cli --locked
target/release/sejong train --config configs/sejong-parameter-vjp.toml \
  --data data/corpus --out runs/sejong-parameter
target/release/sejong generate --model runs/sejong-parameter/model \
  --prompt 'Sejong' --max-bytes 128
```

NativeValue의 `ValueTraining.parameter_vjp`도 같은 Graph와 잔차 연산을 사용한다. 기존 NativeValue Adam 경로는 optimizer 경계에서 해당 미분을 dense buffer로 합산한 뒤 원소별 moments를 갱신한다. `GradientMaterializer`가 gradient buffer를 별도로 소유하고, 원래 가중치는 Adam 갱신에서 사용한다. Native checkpoint의 `parameter_rng`가 미분 표집 상태를 보존한다. 기존 spec의 생략 필드는 기존 학습 설정을 따른다.

자료 생성기에 TOML을 전달하면 model 표와 `training.parameter_vjp`가 `native-b.json`에 들어간다.

```bash
python scripts/create-native-data.py --out data/native-parameter \
  --value-model-config configs/sejong-parameter-vjp.toml
target/release/sejong train-value --spec data/native-parameter/native-b.json \
  --out runs/native-parameter
```

검증 진입점은 [미분·SGD 커널](../crates/sejong-gpu/examples/parameter_vjp_check.rs), [전체 모델 학습·재개](../crates/sejong-train/examples/parameter_vjp_contract.rs), [NativeValue 학습·봉인·재개](../crates/sejong-train/examples/native_parameter_vjp.rs)다. 커널 예제는 서로 다른 rank의 두 인수 항과 dense 항, 비정렬 행렬, 정확한 norm과 weight decay·clipping을 FP64 수식으로 대조한다. 전체 모델 계약은 설정한 실제 본체를 학습하고 checkpoint 재개 결과와 전체 잔차 관측의 정확 SGD 결과를 대조한다.
