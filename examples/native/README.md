# Native 입력 예제

기본 자료는 인원 6명·술어 5개로 생성한다. `query.json`, `story.json`, `story-two.json`, `joint.json`은 문장·사건 경로·공유 사실 질의다. `lexical.json`과 `renamed.json`은 등록 어휘와 개체 이름을 사용한다. `assembly.json`은 원본 편입, `weights.json`은 연속 읽기, `revision.json`은 K/V 개정 입력이다.

`python scripts/create-native-data.py --out data/native-v6`로 해당 원본의 학습 spec과 gold 답을 생성한다. `--people`, `--predicates`, `--max-depth`로 자료를 조정한다. 질의 학습은 `Program` 구조와 STOP을 사용한다. 모델 설정과 학습 정책은 생성한 `recipe.json` 및 각 학습 spec에 저장한다.
