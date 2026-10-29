# 실행 설정

`sejong.toml`은 독립 byte 모델·학습·GPU 설정을 담는다. `donor-a.toml`과 `donor-b.toml`은 서로 다른 ID와 초기값을 적용한다. 생략한 필드는 `sejong-core`의 기본값을 사용한다.

Native 역할·사실 학습과 편입 설정은 `examples/native/`의 JSON 형식을 따른다. 모델별 실행 규격은 저장된 artifact에 함께 들어간다.
