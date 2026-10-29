# Sejong

- Model name: Sejong. Workspace root: `/home/renya/Sejong`.
- Implement the model, training and inference in Rust plus Python/Triton.
- Do not use PyTorch or a framework wrapping PyTorch, including for tensor allocation, automatic differentiation, reference execution or kernel launching.
- Write original implementation code.
- Design and implement the intended complete system together. Do not substitute toy models, temporary execution paths, fake kernels or placeholder outputs for requested functionality.
- Keep checks proportional to the change. Avoid excessive tests, repeated validation and experiment reports unless requested.
- Distinguish planned interfaces from implemented behavior. Never report training, GPU execution or model quality as working without evidence.

## 문서와 응답의 문체 규칙

이 규칙은 프로젝트의 Markdown 문서, 코드 주석, CLI 안내, 작업 설명과 사용자 응답에 적용한다.

- 자기방어와 책임 회피를 목적으로 하는 문장을 금지한다. 면책성 단서, 과도한 유보, 예상 비판에 대한 선제 해명으로 설명을 채우는 문체도 금지한다.
- “하지 않는다.”, “이것을 ~라고 주장하지 않는다.”, “아직 ~는 아니다.”처럼 부정과 비주장으로 대상을 설명하는 문체를 금지한다. 같은 뜻의 우회 표현과 영어 표현에도 동일하게 적용한다.
- “X가 아니라 Y다”, “X는 아니다”, “~을 보장하지 않는다”처럼 사용자가 묻지 않은 대상을 끌어와 선을 긋는 서술을 금지한다.
- 구현 내용, 작동 방식, 선택의 이유와 결과를 구체적인 사실과 능동형 문장으로 서술한다.
- 진행 상태는 구성 요소와 실제 동작을 직접 적는다. 예: “현재 구성은 Rust 패키지 6개와 Triton 패키지 구조다.”
- 작업에 영향을 주는 문제는 원인, 영향과 해결 행동으로 적는다. 예: “GPU 메모리 부족으로 실행이 종료됐다. 배치 크기를 조정해 다시 실행한다.”
- 확인된 사실을 대상, 조건과 결과가 분명하게 드러나도록 정확히 명시한다. 수치와 성공 여부는 실제 근거에 맞춘다.
- 필요한 기술적 제약과 사용자가 요청한 금지 조항은 적용 대상과 조건을 명시한다. 이 조항의 금지 표현은 작업 지침이며 자기방어적 설명의 근거로 사용하지 않는다.
