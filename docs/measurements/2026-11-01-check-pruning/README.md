# 검사 비용 정리 후 RX 6600 측정

2026-11-01. [38개 항목의 구현 대응표](../../research/check-overhead-changes.md)는 반복 검사·전체 해시·메모리 재합산과 상시 진단을 삭제하거나 준비 경계·선택한 진단으로 옮긴 내용을 담는다.

RX 6600, release 빌드, 26,488,192 parameters, sequence 512 × accumulation 8, warmup 2 + 측정 64 step, 평가 8 batch를 사용했다. 자료·seed·배치·학습률 일정은 [직전 측정](../2026-11-01-parameter-vjp-overhead/benchmark.json)과 같다. [원본 결과](benchmark.json)에 소스·실행 파일 fingerprint와 각 step을 저장했다.

| 경로 | 이전 ms/step | 현재 ms/step | 이전 → 현재 launches/step | 현재 issued GFLOPs/step | 현재 검증 BPB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 기존 정확 SGD | 263.385 | 257.910 | 14,728.625 → 14,537.625 | 109.7152 | 7.935028 |
| 전역 타일 정확 SGD | 256.955 | 257.225 | 14,334.891 → 14,340.891 | 109.7158 | 7.934972 |
| 기본 parameter VJP | 259.391 | 259.633 | 14,253.625 → 14,259.625 | 107.1864 | 7.934876 |

기존 정확 SGD의 평균 시간은 2.08% 감소했다. 전역 타일 정확 SGD와 기본 압축 경로는 각각 0.10%, 0.09% 증가했다. 기본 압축 경로의 주요 FLOPs와 검증 BPB는 직전 값과 같았다. 이번 측정에서 확인한 기본 경로의 시간 변화는 +0.243ms/step이다.

위 비교는 `benchmark`가 FLOPs 계측과 매 step 기록을 켠 실행이다. 일반 `train`의 같은 66-step 구성에서 계측을 끈 [실행](unmetered.json)은 warmup 뒤 평균 257.483ms/step을 기록했다. `metrics_collected=false`, `log_every=1`이며 이 값은 현재 구현의 계측 비용을 함께 살피기 위한 별도 관측이다.

[검증 결과](verification.json)에는 GPU 커널, 전체 byte 본체, NativeValue, Memory, 교정 SGD와 파서의 학습·재개 대조를 담았다. CPU 단위 테스트 5개와 전체 target·선택적 kernel-profile 빌드가 통과했다. 큰 gradient 반례의 실제 norm은 FP64에서 6.80565×10³⁸, SGD 결과 원소는 FP32 유한값이었다. Byte 재개 가중치 최대 차이는 2.84×10⁻¹⁴, NativeValue는 5.96×10⁻⁸이었다. 가중치가 0인 midpoint·consistency·contrast 경로와 해당 checkpoint 재개도 실행했다.

```bash
target/release/sejong benchmark \
  --config configs/sejong-sparse.toml \
  --config configs/sejong-parameter-exact.toml \
  --config configs/sejong.toml \
  --data docs/measurements/2026-11-01/train.txt \
  --validation docs/measurements/2026-11-01/validation.txt \
  --out runs/check-pruning-rx6600 \
  --warmup-steps 2 --measure-steps 64 --evaluation-batches 8 --decode-bytes 16
```
