# 현재 Sejong 구조에 맞는 학습 가속 실험

측정일: 2026-11-01. **현재 기본 경로는 활성값 보존, 실행 batch 통합, F16 dot2를 사용한다.** RX 6600에서 전체 26,488,192-parameter byte 모델을 실행했다. 독립520-step 비교에서 평균 시간은 **222.499→114.034ms/step**, 검증 BPB는 **6.769966→6.769952**였다. 처리량은 **1.951×**, peak GPU 할당량은 **472→632MiB**다. 생산 최소안의66-step 결과는 아래에 기록했다.

연구 실험은 `.cache/fit-research`의 독립 소스 복사본에서 수행했다. 이 디렉터리에 패치·설정·측정 원자료를 보관한다. 연구 실험을 마칠 때 생산 소스·설정·실행파일69개의 SHA256을 시작 시점과 대조했고 모두 일치했다. 이후 아래 최소 조합을 생산 경로에 적용했다.

## 생산 적용

`ops/kernels.py`와 `ops/projection.py`에 측정한 dot2 패치를 적용했다. `configs/sejong.toml`과 `configs/sejong-parameter-vjp.toml`은 `activation_checkpointing=false`, `batch=8`, `accumulation=1`을 사용한다. Dtype 선택은 `constexpr`로 컴파일 시 확정한다. 실행 루프에 추가한 검사·예외 처리·진단 호출은0개다.

적용한 생산 경로에서 같은66-step 학습과 cached 추론을 한 번 실행했다. 마지막64 step은 모두 compile0회이며, 평균 **111.436ms/step**, logical **74.727GF**, issued **77.793GF**, launch **1,401회**, peak **629MiB**, 최종 검증 BPB **7.934923**을 기록했다. Cached/full 최종 logits 최대 차이는0이었다. 실험 최소안과 kernel SHA256·전체 소스 fingerprint·검증 BPB가 일치한다. [applied.json](/home/renya/Sejong/docs/measurements/2026-11-01-architecture-fit/applied.json)에 실제 설정과 step별 결과를 보관했다.

```bash
SEJONG_ROOT=/home/renya/Sejong PYTHONPATH=/home/renya/Sejong/python \
SEJONG_PYTHON=/home/renya/Sejong/.venv/bin/python \
target/release/sejong benchmark --config configs/sejong.toml \
  --data docs/measurements/2026-11-01/train.txt \
  --validation docs/measurements/2026-11-01/validation.txt \
  --out runs/fit-applied --warmup-steps 2 --measure-steps 64 \
  --evaluation-batches 1 --decode-bytes 32
```

## 비교 조건

- GPU: AMD Radeon RX 6600, gfx1032. Rust release 실행파일과 Python/Triton/HIP으로 학습했다.
- 본체: local encoder/decoder 폭128·각2층·window128, patch8, global 폭512·8층·GQA8/4, expert4/top2·내부 폭384.
- 학습: sequence512, effective batch8, **4,096 bytes/step**, FP32 master·F16 compute, SGD, parameter VJP rank16·잔차 표집1/4·feature block32. 입력 VJP는 정확 역전파를 실행한다.
- 기존 기준: physical batch1 × accumulation8, global activation checkpointing. Local 활성값은 보존한다.
- 자료: 기존 [학습 파일](/home/renya/Sejong/docs/measurements/2026-11-01/train.txt) 16,939 bytes와 [검증 파일](/home/renya/Sejong/docs/measurements/2026-11-01/validation.txt) 7,633 bytes. 같은 seed와 step별 8개 학습 sequence를 사용했다. 검증도 같은 8개 sequence이며 평가 호출 수는 `8 / physical_batch`다.
- 66-step 실험은 처음2 step 뒤64 step, 520-step 실험은 처음8 step 뒤512 step을 측정했다. 모든 측정 step에서 새 kernel compilation은0회였다. Learning-rate warmup은100 step이며 520-step 실험은 그 이후420 step을 포함한다.
- FLOPs 범위는 GEMM·attention 내적·parameter-basis QR이며 FMA=2다. Logical은 실제 행렬 형상, issued는 실행 tile의 padding을 포함한다. Pointwise·전송·호스트 실행 비용은 step 시간에 포함한다.

## 전체 모델에서 분리한 효과

다음은 66-step 학습 중64-step 평균이다. `보존`은 global 활성값 보존, `mirror`는 optimizer step 안에서 compute weight 공유를 뜻한다.

| 실행 | Batch × 누적 | ms/step | Issued GF/step | Launch/step | Peak MiB | 최종 검증 BPB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 기존 | 1×8 | 226.089 | 106.499 | 12,908.125 | 472 | 7.934876 |
| 보존 | 1×8 | 190.817 | 87.796 | 11,044.625 | 483 | 7.934876 |
| 보존 + mirror | 1×8 | 181.145 | 87.796 | 9,644.625 | 483 | 7.934876 |
| 보존 + dot2 | 1×8 | 179.906 | 87.802 | 11,045.219 | 483 | 7.934891 |
| 보존 + mirror + dot2 | 1×8 | 170.272 | 87.802 | 9,645.219 | 483 | 7.934891 |
| 보존 | 4×2 | 129.478 | 81.618 | 2,826.625 | 579 | 7.934958 |
| 보존 + mirror + dot2 | 4×2 | 118.450 | 81.617 | 2,626.625 | 579 | 7.934998 |
| 보존 | 8×1 | 121.682 | 77.793 | 1,401 | 629 | 7.934994 |
| 보존 + mirror + dot2 | 8×1 | 114.086 | 77.793 | 1,401 | 629 | 7.934923 |
| **보존 + dot2** | **8×1** | **111.498** | **77.793** | **1,401** | **629** | **7.934923** |

마지막 두 행은 별도 순차 실행이다. 두 구성의 launch 수·논리 FLOPs·issued FLOPs가 같고, 66-step 최종 가중치의 상대 L2 차이는5.53e−11이었다. 8×1에서는 반복 cast 절감량이0이므로 **최소 적용안은 두 Triton 파일과 학습 설정**으로 구성한다.

## 1. 활성값 보존: 현재 메모리 여유로 재계산을 줄인다

Global8층의 checkpoint 재실행을 줄이면 logical FLOPs는 **94.126→77.730GF/step**, issued FLOPs는 **106.499→87.796GF/step**이 된다. 같은 batch1×누적8에서 시간은15.60% 감소하고 peak는11MiB 증가했다.

[Adacc](https://arxiv.org/abs/2508.00806)의 접점은 메모리 예산에 따라 활성값 보존·재계산·압축을 선택하는 방식이다. 현재 형상에서는 보존만으로 위 효과를 얻었다. 확대 형상은 persistent tensor·activation·scratch의 실제 수명과 allocator 단위를 합산해 보존할 block을 정한다.

## 2. 실행 batch 통합: 계층 구조와 압축 VJP를 함께 활용한다

`batch1×누적8 → batch8×누적1`은 동일한8개 sequence의 실행을 묶는다. Sequence별 인과 경계와 global 길이64는 유지하고, 선형층이 처리하는 global 행 수는 **64→512**가 된다.

이 변화가 현재 구조에서 두 비용을 함께 줄인다.

1. 작은 GEMM의 빈 tile과 관측별 launch·factor 복원·BOS projection의 반복이 줄어든다. 활성값 보존 상태의 B1→B4에서 logical은77.730→77.615GF, issued는87.796→81.618GF다.
2. Global 행 수512가 기존 parameter VJP의 `N≥입력 폭512` 조건을 충족한다. 공유 기저와 비용 선택기가 global QKV24개에도 압축을 선택한다. B4→B8에서 logical은77.615→74.727GF, issued는81.618→77.793GF로 줄었다.

66-step 실험의 Batch8 압축 호출44개는 local20개와 global QKV24개다. Token 기저는 localN4096·globalN512의2개, 공유 projection group은16개다. 모델의200개 parameter tensor와26,488,192개 학습가능 가중치를 유지한다.

**이 효과는 실행 형상과 기존 미분 비용 선택기의 결합에서 나온다.** Batch 변경은 기저를 만드는 관측 집합과 잔차 표집 횟수도 바꾼다. 이에 따라 parameter RNG는 서로 다른 경로로 진행하며, 아래520-step 비교에서 같은 자료량 이후의 검증 BPB를 대조했다.

## 3. Dot2: 결합 projection까지 F16 operand를 유지한다

일반 GEMM과 QKV·gate/up 결합 kernel은 F16×F16 operand를 그대로 `tl.dot`에 전달하고 FP32 accumulator를 사용한다. F32 operand가 포함된 곱은 F32 연산을 선택한다. Dtype 선택은 Triton의 `constexpr`로 컴파일 시 처리한다.

[RDNA2 ISA](https://docs.amd.com/v/u/en-US/rdna2-shader-instruction-set-architecture)의 packed FP16 dot 명령을 활용했다. gfx1032 컴파일9개 signature에서 F16 경로의 `v_dot2c_f32_f16` 생성을 확인했다. F16 gate/up의 VGPR은192→163, LDS는4,096→5,120 bytes였다. QKV의 LDS는4,096→7,168 bytes로 증가했다. 실제 전체 모델 시간으로 이 자원 교환의 결과를 판단했다.

보존 B1에서 dot2만 더하면190.817→179.906ms, 보존 B8에서121.682→111.498ms였다. Dot2는 같은 논리 FLOPs의 실행 시간을 줄인다. 누산 순서 변화는 expert routing과 padding에도 반영되며, issued FLOPs의 작은 차이가 여기에 포함된다.

## 520-step 학습 비교

각 구성은 **2,129,920 byte positions**를 학습했다. 시간은 마지막512 step의 평균이다.

| 항목 | 기존 B1/A8 | 보존 + dot2 + mirror B8/A1 |
| --- | ---: | ---: |
| ms/step | 222.499 | 114.034 |
| Logical GF/step | 94.126 | 74.675 |
| Issued GF/step | 105.296 | 77.721 |
| Launch/step | 12,624.533 | 1,399.988 |
| Peak MiB | 472 | 632 |
| 초기 검증 BPB | 8.596439 | 8.596391 |
| 최종 검증 BPB | 6.769966 | 6.769952 |
| 초기화·학습·저장을 포함한 초 | 116.487 | 60.033 |

시간은48.75%, logical FLOPs는20.66%, issued FLOPs는26.19% 감소했다. 최종 가중치의 상대 L2 차이는2.34e−5, 최대 절댓값 차이는1.14e−4였다. 두 실행의 data RNG 상태가 같고, 저장 가중치 전체가 유한값이었다. 같은 held-out 입력의 cached/full 최종 logits 최대 차이는 두 구성 모두0이었다.

결합 구성의512개 측정 step에서 압축 VJP 호출 수는44회가423 step, 46회가78 step, 48회가11 step이었다. 일부 expert의 행 수가 커지면 비용 선택기가 추가 gate/up 쌍을 압축한다. 이 변화로 장기 평균 logical FLOPs는 단기74.727GF에서74.675GF가 됐다.

520-step 결과는 mirror Rust 경로를 포함한 구성의 측정값이다. 최소안의 별도66-step 결과는111.498ms이며, 같은 길이의 mirror 포함 구성과 BPB·optimizer RNG 상태가 같았다. 품질 비교의 관측 범위는 위 학습·검증 파일, seed1개, 검증8개 sequence다.

## 누적 학습에서 남길 선택: step별 mirror

Mirror 공유는 `parameter tensor 수 × (accumulation−1)`개의 cast를 줄인다. 현재200개 tensor에서 A8은1,400회, A2는200회, A1은0회다. 보존 B1/A8에서 **190.817→181.145ms**, 추가5.07% 감소를 측정했다.

메모리 예산으로 물리 batch가 작아져 A>1이 되는 형상에 이 구현을 연결한다. 준비된 F16 값을 원래 master leaf와 cast 미분에 연결하고, optimizer 갱신 전에 step 수명을 끝낸다. 패치는 [mirror.patch](/home/renya/Sejong/docs/measurements/2026-11-01-architecture-fit/mirror.patch)에 보관했다.

## 확대 시 선택 규칙과 다음 적용 범위

Physical batch는 effective batch의 약수에서 고른다. 현재 effective8의 후보는1·2·4·8이며, 측정한 결합 구성의 peak는 B1 483MiB, B4 579MiB, B8 629MiB였다. B8의520-step peak는632MiB다. 각 형상의 실제 GPU 가용량·tensor 수명·측정 시간을 이용해 batch와 checkpoint 계획을 준비 시 정한다. B2는 추가 측정 후보로 남는다.

적용한 최소 변경과 누적 학습용 후속 선택은 다음과 같다.

- [dot2.patch](/home/renya/Sejong/docs/measurements/2026-11-01-architecture-fit/dot2.patch): `python/sejong_kernels/ops/kernels.py`와 `projection.py`의 dtype 전달.
- [retain-b8.toml](/home/renya/Sejong/docs/measurements/2026-11-01-architecture-fit/retain-b8.toml): 현재 형상의 `activation_checkpointing=false`, `batch=8`, `accumulation=1`.
- A>1을 사용하는 확대 형상: step별 mirror 패치와 해당 형상의 메모리·시간 비교.

후속 grouped MoE의 기준은 이 구성에서 **8회/step**인 expert counts CPU 회수다. 기존 checkpoint B1/A8에서는128회였다. 후속 연구는 이 새 비용과 커진 expert GEMM을 출발점으로 삼는다. Query tile·입력 VJP 변경·layer dropout의 평가는 같은 검증 BPB에 도달하는 전체 시간을 별도로 측정하는 순서로 둔다.

## 원자료와 재현

[summary.json](/home/renya/Sejong/docs/measurements/2026-11-01-architecture-fit/summary.json)은12개 전체 모델 실행을 요약한다. `<실행 이름>.json`에는 step별 시간·FLOPs·config·소스/실행파일 fingerprint를 보관했다. [weight-summary.json](/home/renya/Sejong/docs/measurements/2026-11-01-architecture-fit/weight-summary.json)은6개 전체 가중치 대조를 요약한다.

`plan*.json`과 `run_benchmarks.py`는 실행 순서·환경·CLI 인자를 기록한다. 생산 소스를 복사한 실험 root에 해당 패치를 적용하고 `SEJONG_ROOT`, `PYTHONPATH`, `SEJONG_PYTHON`을 그 root와 기존 venv로 지정했다. Mirror 실행파일은 복사본에서 release 빌드했다. Dot2 최소안은 기존 release 실행파일을 사용했다. GPU 실행은 순차로 진행했다.

`source-sha256.json`은 기준 소스와 실행파일, `dot2-source-sha256.json`과 `mirror-variant.json`은 실험 변경을 식별한다. `production-source-comparison.json`은 기준69개 파일의 대조 결과다. `dot2-compile/`에 ISA·자원 비교 요약을 보관했다. 전체 weights와 checkpoint는 `runs/fit-*/run-000/`에 있다.
