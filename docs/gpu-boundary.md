# Rust–Triton GPU 경계

Rust Engine은 HIP 장치·stream·메모리 풀·컴파일 서비스와 커널을 소유한다. Tensor는 allocation 참조, offset, shape과 FP16·FP32·FP64·I32 dtype을 가진다. 마지막 참조를 해제하면 allocation을 같은 stream의 풀로 반환한다.

## 컴파일과 실행

영속 Python 서비스에 연산 이름·인자 signature·constexpr·장치 architecture·wavefront를 JSON-lines로 전달한다. Triton은 HSACO와 symbol·thread 수·shared memory·scratch 크기를 반환한다. Rust는 HIP module API로 결과를 로드하고 장치 주소·형상·scalar를 전달해 실행한다.

캐시 키는 연산 모듈·컴파일 서비스·Triton 버전·컴파일 설정을 포함한다. 커널은 `python/sejong_kernels/ops/`의 `kernels.py`, `common.py`, `parser.py`에 배치한다.

## 메모리와 연산

GPU 예산은 HIP가 보고한 가용량과 `gpu.memory_mb` 설정으로 정한다. 기본값 0은 가용량을 사용한다. Engine은 tensor 할당량과 재사용 풀을 관리하며 학습기는 실제 작업 tensor·gradient·optimizer state의 비용으로 배치를 계산한다.

일반 행렬곱·embedding·gather·RMSNorm·LayerNorm·GELU·ReLU·SwiGLU·RoPE·attention·cross-entropy·Adam을 제공한다. 관계 attention은 거리의 방향·로그 크기와 같은 identity 여부의 학습 bias를 사용한다. Concat·slice·reshape·곱·affine bias와 관계 bias의 gradient도 역전파 그래프에서 누적한다.

동결 MiniLM encoder는 공식 설정의 post-LayerNorm BERT를 FP32로 실행한다. 파서는 입력 개체·술어 수에 따른 tensor를 만들고, 질문 접두사별 조건부 선택을 계산한다. Query와 어휘 relational encoder의 폭·층·head는 모델 설정을 따른다. 입력 길이는 해당 encoder의 위치 규격과 계산할 attention tensor의 메모리로 결정한다.

준비된 memory는 query 본체·입구·K/V·readout을 FP64로 실행한다. 준비와 추론이 같은 `forward_row`를 사용해 선언된 요청의 full/direct 법칙을 계산한다. 학습 gradient와 optimizer state는 FP32다. Byte Transformer checkpoint는 backward에서 블록 내부 활성값을 재계산한다.

## 호스트 계산과 저장

Rust FP64는 Givens QR 입구 피팅, 행 법칙의 보정, 의미·사건 확률 합성, equality partition tensor를 계산한다. QR은 관측을 행 단위로 받고 좌표 차원에 따른 행렬을 보관한다. 경험적 행 보정은 두 FP64 실행의 query 차이와 total variation을 방향 반올림으로 누적한다.

호스트 추론은 `MemAvailable`, cgroup 잔여량과 명시한 바이트 예산을 적용한다. 접두사·행 캐시·공유 사건 그래프·결합 tensor의 형상에서 작업 공간을 계산한다. `sejong-core::numeric`의 정밀도·연산량 기반 오차는 결과의 `numerical_error`에 들어간다. 선언된 full 행에 대한 direct 읽기 보정은 `read_error`로 전파한다.

Safetensors는 원본 dtype을 보존한다. Native snapshot은 원본 파일·입출력·encoder identity·경로·보정값과 서명을 함께 저장한다. 같은 HIP context와 같은 digest의 동결 encoder는 GPU 가중치를 공유한다.

`parser_system`은 인원·술어·경로·어휘 길이 확장과 파서 미분을 대조한다. 공식 ONNX 참조 출력은 `scripts/parser-reference.py`로 생성한다. `memory_contract`는 실제 사실 학습·선언 행 보정·개정·재개를 연결한다. `native_system`은 학습한 원본의 질의·정답·저장 흐름을 실행한다.
