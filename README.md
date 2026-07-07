# 딥러닝 + LLM 이미지 분류/설명 서비스

딥러닝(ResNet18 이미지 분류)과 LLM(Claude API)을 결합한 간단한 Streamlit 서비스 예제입니다.

## 파일 구조

```
dl_llm_app/
├── app.py            # Streamlit 메인 앱 (UI)
├── model.py          # 딥러닝 이미지 분류 클래스 (ImageClassifier)
├── llm_service.py     # LLM 결과 설명 클래스 (LLMExplainer)
├── requirements.txt   # 필요 패키지 목록
└── README.md
```

## 동작 흐름

1. 사용자가 이미지를 업로드
2. `model.py`의 `ImageClassifier`가 **딥러닝(ResNet18, ImageNet 사전학습)** 으로 이미지를 분류하여 Top-3 클래스와 확률 반환
3. `llm_service.py`의 `LLMExplainer`가 **Claude API**를 호출하여 예측 결과를 사람이 이해하기 쉬운 문장으로 설명
4. 사용자가 추가 질문을 입력하면 그 질문에 대한 답변도 함께 생성

## 로컬 실행 방법

```bash
pip install -r requirements.txt
streamlit run app.py
```

실행 후 사이드바에 본인의 **Anthropic API Key**를 입력하면 LLM 설명 기능을 사용할 수 있습니다.
(API Key가 없으면 딥러닝 예측 결과까지만 확인 가능합니다.)

## Streamlit Cloud 배포 방법

1. 이 폴더(`app.py`, `model.py`, `llm_service.py`, `requirements.txt`)를 GitHub 저장소에 업로드
2. [share.streamlit.io](https://share.streamlit.io) 접속 후 GitHub 계정 연동
3. "New app" → 저장소/브랜치 선택 → Main file path에 `app.py` 지정
4. Deploy 클릭 → 배포 완료 후 URL 생성
5. 앱 실행 화면 사이드바에서 Anthropic API Key를 입력해 사용

> API Key를 코드에 직접 넣지 말고, 실행 시 사이드바에서 입력하도록 되어 있어 안전합니다.
> (원한다면 Streamlit Cloud의 "Secrets" 기능에 `ANTHROPIC_API_KEY`로 등록해 자동 입력되도록 확장할 수도 있습니다.)

## 과제 관점에서의 포인트

- **딥러닝 파트**: `model.py` — 사전학습 CNN(ResNet18)을 이용한 이미지 분류 (전이학습/추론)
- **LLM 서비스 파트**: `llm_service.py` — 딥러닝 결과를 자연어로 해설해주는 LLM 연동 서비스
- **클래스 단위로 파일 분리**: `ImageClassifier`, `LLMExplainer` 각각 별도 `.py` 파일로 구현
- **배포**: Streamlit Cloud를 통한 웹 서비스화
