# Attention·RMSNorm·projection·arena·HIP graph 연구

2026-11-01. 현재 Sejong의 계층 본체·수동 역전파·parameter VJP를 기준으로 다섯 항목을 분석하고 별도 Triton·직접 HIP 커널을 실행했다. [원본 결과와 재현 코드](../measurements/2026-11-01-fused-execution/README.md)는 독립 커널과 dispatch chain의 측정이다. 아래 `ProjectionFamily`, arena IR, `StepState`는 생산 통합을 위한 제안 인터페이스다.

후속 [생산 적용](../measurements/2026-11-01-fused-applied/README.md)은 RMSNorm partial·Delta 공유·QKV/gate-up 순전파를 연결했다. Graph는 원래 논리 node와 역전파를 유지한다. 같은 실제 본체의 평균 step 시간은 259.834→226.306ms였다. 아래 family owner·arena·HIP graph는 확장 설계다.

가장 먼저 연결할 대상은 RMSNorm의 partial reduction과 attention Delta 공유다. 기존 Graph의 입력·출력·launch 수 안에서 교체할 수 있다. QKV·gate/up 순전파 결합은 원래 master parameter와 logical 미분 관계를 유지하는 family 연산으로 연결한다. Arena는 이 연산들의 실제 buffer 수명을 준비하고, HIP graph는 그 고정 주소·연산 순서를 재실행한다.

| 대상 | 실험 범위 | 확인한 결과 |
| --- | --- | --- |
| RMSNorm backward | 기존 fill+atomic 두 launch → partial+final 두 launch | global 16.207→6.932µs, local 167.478→7.682µs, final 144.743→11.300µs |
| Attention backward | dQ+dKV 두 launch에서 Delta 저장·재사용 | local 233.066→223.449µs, global 129.016→123.966µs |
| Gate/up 순전파 | 두 GEMM+SwiGLU → 두 accumulator와 activation epilogue | local 85.894→65.805µs, expert 64.992→35.658µs |
| QKV 순전파 | 세 GEMM → K/V 폭을 반영한 다중 출력 GEMM | local 중앙값 157.881→73.026µs, GQA 124.748→58.017µs |
| Gate/up 입력 미분 | 두 GEMM+add → 두 accumulator와 최종 합산 | local 237.537→231.997µs, expert 115.095→113.674µs |
| HIP graph | 256원소 Sejong add 14,260개 chain의 direct 제출 → replay | CPU enqueue 27.944→0.116ms, 동기 완료 43.194→36.120ms |

QKV local의 기존 경로 표본은 123.761–160.525µs, 결합 BN32는 71.283–73.771µs였다. 각 배속은 표에 명시한 연산 묶음과 형상에 대응한다. 전체 학습의 다음 측정은 통합한 경로의 compile·준비·메모리·step 시간을 함께 수집한다.

Attention에서 query별로 공유할 값은

\[
\Delta_{b,t,h}=\sum_d \mathrm{DO}_{b,t,h,d}\,\mathrm{O}_{b,t,h,d}
\]

이다. 현재 [dQ](../../python/sejong_kernels/ops/hierarchy.py#L56)는 이 값을 query당 한 번 계산하고, [dKV](../../python/sejong_kernels/ops/hierarchy.py#L99)는 key마다 같은 내적을 계산한다. 제안 커널은 dQ의 register 값을 `Delta[B,T,HQ]` FP32 buffer에 저장하고 뒤의 dKV가 읽는다. 기존 두 launch와 같은 stream 순서가 쓰기·읽기 의존성을 연결한다. dKV의 O load와 Delta reduction을 공유하고, dQ의 계산식은 유지한다.

유효 key 폭을 `L=min(T,window)`로 두고 global은 L=T로 두면 causal pair 수는 `P=TL−L(L−1)/2`다. 기본 local은 P=57,408, global은 P=2,080이다. Local 32회·global 64회 backward와 실제 query heads를 합하면 반복 내적은 **601,862,144개의 scalar 곱·합/step**이다. FMA=2 계측 규칙에서는 **606,601,216 FLOPs/step**이며 dKV의 per-pair dot 계수는 10D→8D로 바뀐다. FP16 O read를 FP32 Delta read와 저장으로 바꾼 논리적 load/store 감소량은 **587,382,784 bytes/step**이다. 이 값은 source-level traffic이며 DRAM cache traffic은 hardware counter로 측정할 대상이다.

Delta는 local 4KiB, global 2KiB다. 기본 두 형상의 dQ/dK/dV는 기존 결과와 bitwise 일치했다. Batch2·T33·GQA4/2·D32·window17에서는 dK 최대 절대차가 2.98×10⁻⁸이었다. 기존 O(F16)·LSE(F32)를 조건으로 CPU FP64 backward 수식도 대조했다. 이후의 query tile·GQA group 재사용은 같은 K/V tile을 여러 query가 공유하는 단계로 확장한다. Tile의 causal/window 경계와 query head→KV head mapping을 계획에 담는다. [FlashAttention의 backward](https://papers.nips.cc/paper_files/paper/2022/file/67d57c32e20fd0a7a302cb81d36e40d5-Supplemental-Conference.pdf)는 row Delta와 tile별 gradient 합산을 설명한다.

RMSNorm에서 행 r의 inverse를 `s_r=(mean(x_r²)+epsilon)^(-1/2)`, `u_r=dy_r*w`로 두면 미분은

\[
dx_r=s_r\bigl(u_r-x_r\operatorname{mean}(u_rx_r)s_r^2\bigr),
\qquad dw_j=\sum_r dy_{rj}x_{rj}s_r
\]

이다. 현재 [backward](../../python/sejong_kernels/ops/kernels.py#L61)는 각 token이 같은 dW 주소를 atomic add한다. 실제 gfx1032 컴파일 산출물의 [CAS loop](../measurements/2026-11-01-fused-execution/attention-rms/rms-base-128.amdgcn#L170)가 경합 시 재시도한다. 기본 본체의 backward는 global 136회 `(R64,D512)`, local 80회 `(R512,D128)`, final 8회 `(R512,D512)`다. 합계 224회·**11,796,480개 논리적 atomic 갱신/step**이며 CAS 재시도가 추가된다.

첫 커널은 BR개 행의 dx와 `partial[group,j]=sum_group(dy*x*s)`를 계산한다. 두 번째 커널은 partial의 group 축을 합산해 dw를 저장한다. 기존 `fill(dw)+atomic` 두 launch를 `partial+final` 두 launch로 교체한다. Temporary는 `4*ceil(R/BR)*D` bytes이며 실험에서 global BR4와 local BR8은 각각 32KiB, final BR8은 128KiB였다. Temporary slot은 각 norm의 마지막 reduction 뒤 재사용한다.

실험의 dx는 기존 결과와 bitwise 일치했다. dw는 합산 순서가 바뀌며 FP64 기준 상대 L2가 약 8–9×10⁻⁸이었다. R129·D257도 실행해 행·열의 tail mask와 실제 D 분모를 대조했다. BR4·8·16의 전체 시간과 자원을 원본에 기록했다. BR16·D512는 VGPR256, BR4·D512는 VGPR67, BR8·D128은 VGPR60이었다. BR은 실제 형상·컴파일 자원·측정값에서 선택하고, 큰 R의 final은 group tile streaming 또는 다단계 reduction으로 확장한다. [Triton의 weight-gradient partial reduction 설명](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)을 참고해 현재 RMSNorm 수식의 독립 partial 저장을 구현했다.

현재 norm backward는 FP32 산술·출력을 사용한다. FP64 forward의 확률 계산 경로는 해당 정밀도 경로로 유지한다. Forward inverse 저장은 현재 forward의 `1/sqrt`와 backward의 `rsqrt` 사이 산술 차이까지 함께 평가하는 다음 후보다.

Projection 결합은 같은 입력을 사용하는 원래 parameter들의 논리적 family로 정의한다. 아래 필드는 제안한 실행 계획의 정보다.

```text
ProjectionFamily {
  input: ValueId,
  members: [(ParameterId, OutputId, output_width, correction)],
  output_layouts: [contiguous spans],
  backward_members: [logical matmul identity],
  basis_group, donor_member, donor_columns, feature_selection,
  epilogue: plain | rounded_gate_up_swiglu | rounded_qkv_rope,
  kernel_plan: tile, precision, resource usage
}
```

현재 [Tensor](../../crates/sejong-gpu/src/lib.rs#L23)는 contiguous shape·offset을 사용하고 [column slice](../../crates/sejong-gpu/src/common_ops.rs#L115)는 복사를 실행한다. Family는 원래 master와 leaf를 유지하고 별도의 contiguous 출력 span에 직접 저장한다. Kernel 내부에서 input tile과 accumulator를 공유해 packing·split을 수행할 위치를 정한다. Step별 packed mirror를 선택하는 경우 준비 비용은 accumulation 전체와 optimizer 갱신까지 포함해 계산한다.

Gate/up의 순전파는 같은 X tile로 두 FP32 accumulator를 만들고, 각각 기존 G/U dtype(F16)으로 round한 뒤 SwiGLU를 계산한다. 학습에 필요한 G·U·H를 모두 저장한다. 이렇게 하면 두 GEMM과 activation의 세 launch를 한 launch로 연결하고 X의 중복 load·G/U 재독출을 줄인다. FP32 IEEE operand 연산을 유지한 실험에서 세 형상의 전체 G/U/H가 bitwise 일치했다. Local `(M512,N384,K128)`은 1.305×, expert `(M32,N384,K512)`는 1.823×, 비정렬 M47은 1.609×였다. VGPR은 111→192, LDS는 4KiB, private/global scratch는 0이었다. 이 실험은 fusion에 따른 효과를 측정했다.

QKV의 논리적 폭을 Q=D, K=V=J로 두면 유효 GEMM FLOPs는 `2MD(D+2J)`다. GQA에서 J=D/2일 때 전 영역에서 세 accumulator를 실행하면 타일 연산량이 50% 증가한다. 측정한 커널은 `n_tile_start<J` 영역에서 Q/K/V를 함께 계산하고 나머지 영역은 Q만 계산한다. 따라서 같은 tile을 사용할 때 issued K/V 연산은 각 실제 폭의 타일 수를 따른다. 기본 local `(M512,D128,J128)`과 global `(M64,D512,J256)`에서 전체 Q/K/V가 bitwise 일치했다. BN32는 각각 중앙값 2.162×·2.150×였고, BN16은 각각 2.071×·1.610×였다. 두 tile의 자원·모든 시간 표본을 보관했다.

미분은 원래 family member마다 `dW_i=XᵀD_i`, 공통 입력에 `dX=sum_i D_iW_iᵀ`를 적용한다. 모든 output cotangent가 준비되면 family owner가 backward를 한 번 실행한다. 현재의 단일-output `Some(dy)` 분기 앞에 이 완료 경계를 연결하고, 기존 입력 미분의 fold 순서 QKV `(V+K)+Q`, gate/up `U+G`를 precision 계획에 둔다.

Parameter VJP의 입력 묶음은 새 family의 `logical matmul identity`에서 구성한다. 같은 donor member·열·방문 순서·seed 소비를 보존하면 기존 표집을 재현할 수 있다. 현재 동률 donor는 local QKV의 Value와 gate/up의 Up이다. A·선택 잔차는 family에서 공유하고 각 member의 B·tail을 원래 leaf에 전달한다. 결합된 QKV의 전체 열에서 donor를 새로 선택하는 정책은 별도 표집 정책으로 저장한다.

RoPE epilogue는 기존 projection 출력의 F16 rounding 뒤 회전하는 순서를 따른다. Parameter VJP가 사용하는 pre-RoPE donor는 raw Q/K span 또는 준비한 donor 열 저장으로 유지한다. CV의 `W_base+UAVᵀ`는 member별 correction과 projected gradient 경로를 보존한다. 현재 low-rank forward의 dense·correction rounding과 최종 add 순서도 family의 precision 계획에 담는다.

Gate/up dX를 두 accumulator로 결합한 실험은 3→1 launch와 bitwise 동일한 출력을 얻었지만 시간 이득은 local 1.024×·expert 1.013×였다. VGPR127→249, LDS6→8KiB가 늘었다. 입력 미분의 다음 후보는 목적 gradient에 beta epilogue로 누적하거나 tile을 줄이는 계획이다. 각 GEMM의 결과를 별도 FP32 accumulator에서 합치는 순서와 기존 gradient alias를 함께 준비한다. QKV dX도 같은 비용·수명 기준으로 선택한다.

Arena의 입력은 forward·backward·checkpoint 재계산을 펼친 연산 DAG와 saved-for-backward 목록이다. 현재 [Node](../../crates/sejong-gpu/src/graph.rs#L47)가 value를 tape 전체 수명 동안 보유하므로, 실행 계획이 실제 마지막 read를 계산해 slot 수명을 정한다. `rows`·`reshape`가 공유하는 allocation과 [Add backward의 공유 dy](../../crates/sejong-gpu/src/graph.rs#L791)를 같은 storage alias로 추적한다.

```text
ValueSpec   = ValueId, StorageId, layout, byte_span, alignment
KernelCall  = KernelId, reads, writes, saved_for_backward, arg_layout, grid
StorageLife = StorageId, born, last_read, arena_kind, capacity
ArenaSlot   = StorageId, arena_kind, offset, capacity
```

Storage의 생성 순서로 interval을 배정하고, 마지막 reader가 끝난 slot을 free 목록에 합친다. 실제 kernel의 alignment를 만족하는 호환 block을 사용하며 남는 span을 분할·병합한다. Arena 크기는 배정한 `offset+capacity`의 최댓값이다. 동시에 살아 있는 고유 storage byte 합의 최댓값을 하한으로 기록하면 계획의 단편화를 비교할 수 있다. Alias의 모든 reader가 끝난 입력 span은 같은 layout의 출력 목적지 후보가 된다. 이 판정은 준비 단계에서 gradient beta epilogue와 in-place 연산에 전달한다.

Persistent arena에는 master·FP16 mirror·optimizer/CV state·static descriptor를 둔다. Step arena에는 관측별 dense gradient와 `FactoredRows.left/right/tail/row_map`, loss summary를 둔다. Temporary arena에는 activation·checkpoint scratch·VJP Q/projection scratch를 둔다. Shared left·row map은 마지막 optimizer 소비까지 pin하고, 공유 Q는 마지막 VJP 소비까지 pin한다. 각 관측의 factor를 보존하면서 다음 관측의 temporary slot은 재사용한다. 실제 GPU budget·persistent 크기·slot 비용으로 activation 보존과 재계산 구간을 선택한다.

HIP graph는 이 고정 주소와 module을 소유하는 executable에서 재실행한다. 설치된 HIP runtime에서 capture·instantiate·launch·node enable·kernel parameter update를 실행했다. 128-node graph의 한 node를 끄면 출력127, 다시 켜고 alpha를2로 바꾸면 출력129였다. 14,260-node add chain의 준비 비용은 capture21.804ms+instantiate60.259ms였다. 동일 chain의 동기 완료 절감량으로 계산하면 약12 replay에서 준비 비용을 상쇄한다. 이 시간은 해당 dispatch chain의 결과다.

변하는 값은 고정 device buffer인 `StepState {step,lr,decay,control_weights,seeds}`에서 읽는다. Capture가 host scalar를 복사하므로 현재 seed·LR를 이 buffer 또는 소수의 node 갱신으로 연결한다. 기본 parameter-indices seed는 8회/step이다. Rust SplitMix64의 seed schedule K회를 준비하면 state를 `K*0x9e3779b97f4a7c15 (mod 2^64)`만큼 진전시켜 같은 순서를 보존한다. 데이터 표집 RNG는 원래 자료 표집 순서로 별도 보존한다. Checkpoint는 graph 완료 경계에서 weights·optimizer·RNG·step을 저장하고 restore에서 arena 주소·descriptor·graph를 다시 bind한다.

현재 expert counts의 host 회수는 GPU queue로 연결할 경계다. 선택 slot 수 `S=R*K`, 활성 expert 수 상한 `a=min(E,S)`라면 row tile 수의 상한은 `a+floor((S−a)/BM)`이다. 각 양수 M_e에 대해 `ceil(M_e/BM)<=1+(M_e−1)/BM`를 합쳐 얻는다. GPU는 실제 `sum_e ceil(M_e/BM)`개의 task만 queue에 넣고 grouped GEMM이 읽는다. 고정 주소·capacity는 S와 형상에서 계산하고 실제 실행량은 device counts에서 정한다.

첫 replay는 local encoder/decoder와 정적 dense segment에 적용한다. 관측별 Q·feature map·factor는 segment 경계를 넘는 slot에 둔다. GPU grouped experts를 연결하면 관측 graph, 이어 전체 step graph로 묶을 수 있다. Compiler warmup·module load·arena 할당·descriptor 준비를 capture 전에 수행한다. 입력·라벨은 고정 device buffer와 pinned staging의 비동기 복사로 연결하고, 완료 event까지 staging을 보존한다. 기본 갱신·clipping·진단·CV refresh는 필요한 plan variant를 준비한다. [HIP graph 규격](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/hipgraph.html)은 capture 중 stream 연산의 기록과 재실행·메모리 수명을 설명한다.

통합 순서는 **RMSNorm partial·Delta 공유 → 논리적 projection family와 QKV·gate/up 순전파 → gradient 목적지·arena 수명 계획 → 정적 segment HIP replay → GPU grouped experts와 전체 step replay**다. 각 단계의 준비 비용·커널 시간·메모리·학습 결과를 같은 실제 본체에서 연결해 측정한다.
