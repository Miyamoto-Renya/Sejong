# 순전파 기저 parameter VJP의 RX 6600 측정

이 문서는 입력 묶음마다 기저를 생성한 초기 구현의 측정이다. 현재 구현의 자원 공유와 고정 경로는 [준비 비용을 줄인 후속 측정](../2026-11-01-parameter-vjp-overhead/README.md)에 기록한다.

2026-11-01. 계층·GQA·전문가 본체의 파라미터 26,488,192개를 학습했다. 모델·초기값·학습률·학습 자료·검증 자료·관측량을 맞추고 정확 SGD, 전역 타일 SGD의 정확 미분, 기본 설정의 순전파 기저·잔차 미분을 비교했다. [원본 측정](benchmark.json)에 모든 측정 step과 실행 파일·소스·자료 fingerprint를 저장했다.

GPU는 RX 6600, 실행은 Rust release + Triton 3.8.0/ROCm SDK 10.1.0이다. Python은 3.14.4이며 `torch_installed`의 측정값은 false다. 각 구성은 66번 가중치를 갱신했고, 첫 2 step 뒤 64 step을 집계했다. 관측량과 역전파량은 각각 step당 4,096 bytes다. 측정 구간의 신규 커널 컴파일 수는 세 구성 모두 0이다. 별도 validation 자료의 512-byte 배치 8개로 초기·최종 BPB를 계산했다.

| 구성 | 평균 초/step | Logical GFLOPs/step | Issued GFLOPs/step | Launch/step | Peak MiB | 최종 검증 BPB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 기존 정확 SGD `sejong-sparse.toml` | 0.263702 | 97.42949 | 109.71519 | 14,728.63 | 465 | 7.935028 |
| 전역 타일 SGD·정확 미분 `sejong-parameter-exact.toml` | 0.256943 | 97.42949 | 109.71578 | 14,334.89 | 465 | 7.934972 |
| 기본 설정·rank 16·잔차 1/4 `sejong.toml` | 0.278218 | 86.90080 | 99.32839 | 15,064.47 | 577 | 7.935099 |

초기 검증 BPB는 세 구성 모두 8.596439다. 새 미분은 기존 정확 SGD 대비 logical FLOPs를 10.81%, issued FLOPs를 9.47% 줄였고 평균 step 시간은 5.50% 늘었다. 전역 타일 SGD의 정확 미분과 비교한 시간 증가는 8.28%다. 기저·잔차 준비, 인수 읽기와 전체 optimizer 갱신이 측정 시간에 함께 들어간다. 최종 모델의 full/cached 마지막 logit 최대 차이는 세 구성 모두 0이었다.

FLOPs는 실행한 GEMM·attention 내적·기저 QR의 산술과 optimizer 내부의 인수 행렬곱을 센다. FMA는 2 FLOPs다. Logical 값은 유효 형상, issued 값은 실제 커널 타일과 QR 패딩의 산술을 사용한다. Pointwise·표집·norm·routing·host 전송은 wall time에 포함한다. 누적 배치의 rank 축을 한 BK 타일에 연결하고, 정렬된 feature block에서는 입력 투영·잔차 준비를 한 커널로 실행했다. 구간 broadcast 역전파는 수신 구간별 고정 순서로 합산한다.

새 미분은 모든 관측에서 원래 loss와 입력 미분을 계산한다. 가중치 미분은 현재 순전파 기저의 성분과 포함 확률로 보정한 선택 잔차를 합산한다. 기본 설정의 `exact_bytes`는 null이며, `backward_bytes`는 실제 역전파 관측량이다. 비교한 학습·검증 범위는 위의 66번 갱신과 validation 4,096 bytes다.

실행 명령:

```bash
source scripts/env.sh
cargo build --release -p sejong-cli --locked
target/release/sejong benchmark \
  --config configs/sejong-sparse.toml \
  --config configs/sejong-parameter-exact.toml \
  --config configs/sejong.toml \
  --data docs/measurements/2026-11-01/train.txt \
  --validation docs/measurements/2026-11-01/validation.txt \
  --out runs/parameter-vjp-stable-rx6600 \
  --warmup-steps 2 --measure-steps 64 \
  --evaluation-batches 8 --decode-bytes 16
```

[검증 원본](verification.json)은 다음 실행 결과를 담는다.

| 검증 | 실제 결과 |
| --- | --- |
| 비정렬 VJP·인수 2개·dense 합산·clipping·weight decay | 세 형상의 VJP relative squared error 3.12×10⁻¹⁴–1.09×10⁻¹³, SGD 최대 오차 2.39×10⁻⁸ 이하 |
| Register·streaming 기저 | 직교성 최대 오차 1.94×10⁻⁷ 이하 |
| 구간 broadcast·partial patch·여러 배치의 미분 | 참조식 최대 차이 0 |
| 전체 byte 모델 6-step·checkpoint 3에서 재개 | 최종 가중치·optimizer RNG 최대 차이 각각 0 |
| 전체 잔차 관측의 전역 SGD와 기존 정확 SGD | 가중치 최대 차이 4.66×10⁻¹⁰ |
| NativeValue 전체 본체·Adam·2-step·checkpoint 1에서 재개 | 미분 RNG 일치, 가중치 최대 차이 1.19×10⁻⁷ |
| NativeValue 봉인·재로딩·32개 요청의 재개 출력 확률 | 재로딩 가중치·출력 확률 최대 차이 각각 0 |
| Rust | workspace tests 5개 통과, all-targets check 통과 |

NativeValue는 [실행한 spec](native-b.json)의 전체 본체와 32-example 배치를 사용했다. Step 1은 119개, step 2는 116개 parameter 행렬에서 압축 미분을 실행했다. NativeValue의 Adam은 optimizer 경계에서 gradient를 합산해 moments에 전달한다. Byte SGD는 인수와 선택 행을 직접 소비한다.

```bash
cargo run --release -p sejong-train --example parameter_vjp_contract -- \
  configs/sejong.toml docs/measurements/2026-11-01/train.txt \
  .cache/parameter-vjp-contract-new
cargo run --release -p sejong-train --example native_parameter_vjp -- \
  docs/measurements/2026-11-01-parameter-vjp/native-b.json \
  .cache/native-parameter-vjp-contract-new
```

구현의 수식·자원 선택·gradient·norm·저장 계약은 [parameter VJP](../../parameter-vjp.md)에 기록했다.
