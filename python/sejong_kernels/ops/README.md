# GPU 연산

`kernels.py`에 byte·query Transformer, K/V 읽기, FP64 연산, 역전파와 AdamW의 Triton 커널을 둔다. Rust `Engine`이 signature와 compile-time 설정을 전달하고 장치 메모리를 관리한다.
