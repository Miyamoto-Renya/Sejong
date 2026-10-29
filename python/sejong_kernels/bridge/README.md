# 컴파일 서비스

`python -m sejong_kernels.bridge.compiler`는 JSON-lines 요청을 받아 HSACO와 호출 metadata를 반환한다. Rust가 이 서비스를 시작하고 HIP 모듈을 로드한다. 컴파일 결과는 소스·설정·컴파일러 버전별 캐시로 저장한다.
