# 순전파 결합과 RMSNorm·attention 공유 적용

2026-11-01. RX 6600에서 26,488,192 parameters의 실제 본체를 학습했다. Sequence512 × accumulation8, 같은 자료·seed·학습률·parameter VJP 정책으로 warmup2 + 측정64 step, 평가8 batch, prefill480 bytes와 decode32 bytes를 실행했다. 두 측정의 64개 step은 모두 compile 없이 실행됐다.

| 항목 | 적용 전 | 적용 후 |
| --- | ---: | ---: |
| 평균 ms/step | 259.834 | 226.306 |
| 중앙값 ms/step | 259.399 | 226.396 |
| logical GFLOPs/step | 94.7326 | 94.1260 |
| 주요 issued GFLOPs/step | 107.1864 | 106.4986 |
| 평균 launches/step | 14,259.625 | 12,908.125 |
| peak GPU MiB | 472 | 472 |
| 최종 검증 BPB | 7.9348758879 | 7.9348758879 |
| prefill ms | 7.070 | 5.072 |
| byte decode 중앙값 µs | 464.42 | 407.28 |
| full/cached 최종 logits 최대차 | 0 | 0 |

평균 step 시간은 **12.90%**, launch는 **9.48%**, prefill 시간은 **28.26%**, decode 중앙값은 **12.30%** 감소했다. 주요 FLOPs 감소율은 0.64%다. Projection 결합은 같은 GEMM 산술에서 공유 입력 읽기·epilogue·dispatch를 줄이고, RMSNorm은 atomic 경합을 partial reduction으로 교체한다. Attention은 반복 Delta 내적을 공유한다.

최종 [가중치 대조](weights.json)는 전체 26,488,192개 원소 중 17개 차이, 최대 절대차 1.49×10⁻⁸을 기록했다. 이 수치와 BPB는 위 66-step 실행의 결과다.

적용한 실행 경로는 다음과 같다.

- [RMSNorm](../../../crates/sejong-gpu/src/norm_ops.rs): 최초 `(rows,width)`에서 연구의 BR 후보를 실제 HIP event로 비교해 cache에 보관한다. 이후 partial·streaming final 두 launch를 실행한다. 큰 group 수는 정해진 타일을 순회한다.
- [Attention](../../../python/sejong_kernels/ops/hierarchy.py): dQ가 FP32 Delta를 저장하고 dKV가 읽는다. 기존 두 launch를 유지하고 dKV의 반복 O 읽기·내적을 줄인다.
- [QKV·gate/up](../../../python/sejong_kernels/ops/projection.py): 다중 accumulator로 X tile을 공유한다. GQA는 실제 KV 폭에 해당하는 tile에서 K/V를 계산한다. Gate/up은 기존 중간 rounding 뒤 SwiGLU를 계산한다.
- [Graph](../../../crates/sejong-gpu/src/graph.rs): 결합된 순전파 출력으로 원래 Matmul·SwiGlu node를 같은 순서에 둔다. 원래 parameter leaf·VJP donor·표집 RNG·checkpoint와 역전파를 연결한다. Cached prefill·decode에도 같은 projection 커널을 사용한다.

RMS 준비는 첫 형상에서 후보별 컴파일·예열과 4회 왕복 측정을 수행한다. 세 후보의 준비 작업은 형상당 774 launches이며 기본 세 norm 형상에서는 2,322회다. 반복 경로는 캐시한 계획을 사용한다. 전체 초기화·저장 포함 시간은 적용 전21.367초, 적용 후15.536초였고, 이 구간의 compile 시간은 각각3.612초·0.032초였다. 위 표의 성능 비교는 warmup 뒤 compile 없는 64개 step을 사용한다.

[검증 원본](verification.json)은 CPU 단위 테스트5개, 전체 target와 kernel-profile 빌드, 계층/GQA/MoE의 미분·cache·저장, 전체 byte 본체6-step 학습·재개·정확 SGD 대조, 전체 NativeValue2-step Adam·봉인·재개, CV 교정·재개를 담는다. Byte 재개 가중치는 원소별로 일치했고 NativeValue 재개 최대차는 7.45×10⁻⁹이었다. 비정렬 F16/F32 projection 전체 출력은 기존 연산과 일치했다. RMS streaming의 R1031,D257,G258 조건에서 FP64 대비 상대 L2는 dx5.84×10⁻⁸·dw9.89×10⁻⁸이었다.

[적용 전 결과](before.json), [적용 후 결과](after.json), [요약](summary.json), [적용 소스 SHA256](source-sha256.json)을 보관했다. 적용 전 기준은 기존 실행파일과 원본 커널 snapshot을 별도 compiler root에서 실행했다. 실행파일의 BLAKE3는 이전 측정과 같은 `f229e94057fba7113d4dbcd4b056963659a4314bc4d143a2c3a4a4e6016f8597`이다. `before.json`의 source fingerprint는 해당 root에 복사한 소스트리, executable fingerprint는 실제 실행파일을 식별한다. [기준 실행 정보](baseline-provenance.json)가 별도 root와 원본 snapshot 위치를 기록한다.

```bash
target/release/sejong benchmark \
  --config configs/sejong.toml \
  --data docs/measurements/2026-11-01/train.txt \
  --validation docs/measurements/2026-11-01/validation.txt \
  --out runs/fused-applied \
  --warmup-steps 2 --measure-steps 64 --evaluation-batches 8 --decode-bytes 32
```
