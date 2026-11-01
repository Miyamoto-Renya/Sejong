# 저수준 최적화 실험

2026-11-01, AMD Radeon RX 6600 / gfx1032. [조사와 구현 우선순위](../../research/low-level-optimization.md)에 코드 근거를 정리했다. 생산 코드의 검사 정리는 [앞선 측정](../2026-11-01-check-pruning/README.md)에 연결된다.

전체 모델 비교는 release 빌드, 26,488,192 parameters, sequence 512 × accumulation 8, warmup 2 + 측정 64 step, 평가 8 batch 조건이다. 기본 구성과 같은 자료·seed·schedule에서 `training.activation_checkpointing`만 false로 바꿔 global 활성값을 보존했다. [전체 원본](retain-activations.json), [요약](summary.json), [가중치 대조](activation-weight-comparison.json)를 저장했다.

| 항목 | 기본 checkpoint | 활성값 보존 |
| --- | ---: | ---: |
| 평균 ms/step | 259.633 | 217.728 |
| 주요 issued GFLOPs/step | 107.1864 | 88.4838 |
| launches/step | 14,259.625 | 11,784.375 |
| peak GPU MiB | 472 | 483 |
| 최종 검증 BPB | 7.9348758879 | 7.9348758879 |

최종 26,488,192개 가중치 중 14개 원소에서 차이가 있었고 최대 절대차는 3.73×10⁻⁹였다. 시간 16.14%, 주요 FLOPs 17.45% 감소는 이 전체 본체·학습 예산의 결과다.

[HIP event 원본](kernel-profile.json)은 별도 [3-step 설정](profile-config.json)의 초기화·학습·저장 구간이다. 총 43,005 launches, 연산별 event 구간 비중은 matmul 57.4%, add 11.4%, copy 6.46%, RMSNorm backward 4.51%다. 각 launch 앞뒤의 event와 host 제출 간격이 이 계측에 포함된다.

[dot2 컴파일](dot2-compile.json)은 F16×F16→F16, FP32 누산, tile 16×32×32, warps 4, wave32 조건이다. [기존 NN ISA](matmul-0-0.amdgcn)와 [변경 NN ISA](native_half_matmul-0-0.amdgcn)에 실제 FP32 FMA와 packed dot2를 저장했다. 컴파일 결과의 정적 명령 수는 해당 loop의 실제 반복 수와 별도로 읽는다.

[독립 GEMM 측정](dot2-microbench.json)은 CPU `ctypes`에서 HIP module을 직접 실행했다. Variant를 ABBA 순서로 동일하게 예열하고, 256개 launch 묶음을 ABBA 8회 측정했다. HIP event는 묶음 앞뒤에 기록했다. 할당·전송·컴파일은 시간 구간 밖에서 실행했고 GPU 측정은 전체 모델 실행과 순차로 진행했다. CPU enqueue 중앙값은 약 1.9µs/launch였다.

| M,N,K | 기존 최소/중앙값 µs | dot2 최소/중앙값 µs | 중앙값 배속 | 출력 상대 L2 차이 |
| --- | ---: | ---: | ---: | ---: |
| 512,384,128 | 73.176 / 80.340 | 64.355 / 65.075 | 1.235× | 9.48×10⁻⁶ |
| 512,128,384 | 85.637 / 87.668 | 63.381 / 66.046 | 1.327× | 1.19×10⁻⁵ |
| 64,512,512 | 69.912 / 73.888 | 54.245 / 56.684 | 1.304× | 1.09×10⁻⁵ |

출력의 최대 절대차는 각각 0.00390625·0.00390625·0.0078125다. 형상별 FP64 CPU dot 64개 표본에서 두 커널의 오차가 같았다. 이 수치는 개별 GEMM의 실행 범위에 해당한다. [컴파일 probe](dot2-compile-probe.py)와 [직접 HIP 측정 코드](dot2-hip-microbench.py)를 함께 보관했다. [소스에서 계산한 작업량](derived-counts.json)은 step mirror·전송·attention Delta 후보의 계산 근거다.

```bash
source scripts/env.sh
python docs/measurements/2026-11-01-low-level/dot2-compile-probe.py
python docs/measurements/2026-11-01-low-level/dot2-hip-microbench.py
```

첫 명령은 현재 compiler로 각 HSACO와 IR·ISA를 만들고, 두 번째 명령은 생성한 NN module 두 개를 직접 HIP으로 실행한다. 결과 JSON과 binary fingerprint는 실행한 환경에 맞춰 기록한다.
